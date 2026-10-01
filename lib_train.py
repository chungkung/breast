"""Training loop, losses, and the multi-loss objective of P1/P2/P3.

Primary early-stopping metric is the DEV-SET AUPRC (breast-level), matching the
protocol; the RSNA pF1 is kept only as a reporting parity metric, never for
selection.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

import config as CFG
import lib_common as C
from lib_models import build_model, GroupDRO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def focal_bce(logits, y, gamma: float = CFG.FOCAL_GAMMA, reduction: str = "mean"):
    """Focal binary cross-entropy. reduction in {'mean','none'}."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pt = torch.where(y > 0.5, p, 1 - p)
    fl = (1 - pt) ** gamma * ce
    return fl.mean() if reduction == "mean" else fl


def binary_kl_distill(t_logits, s_logits, T: float = CFG.KD_T):
    """KL(teacher||student) for Bernoulli outputs, scaled by T^2."""
    t = torch.sigmoid(t_logits / T)
    s = torch.sigmoid(s_logits / T)
    kl = t * torch.log((t + 1e-7) / (s + 1e-7)) + \
         (1 - t) * torch.log((1 - t + 1e-7) / (1 - s + 1e-7))
    return kl.mean() * (T * T)


# --------------------------------------------------------------------------- #
# Epoch + training
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, opt, method, machine_to_group=None, groupdro=None,
              train=True):
    """One epoch. Returns (mean_loss, ys, ps)."""
    model.train(train)
    ys, ps = [], []
    tot = 0.0
    n_seen = 0

    for batch in loader:
        cc = batch["cc"].to(DEVICE)
        mlo = batch["mlo"].to(DEVICE)
        pcc = batch["pcc"].to(DEVICE)
        pmlo = batch["pmlo"].to(DEVICE)
        y = batch["y"].to(DEVICE)
        B = cc.size(0)

        with torch.set_grad_enabled(train):
            logits = model(cc, mlo, pcc, pmlo)

            # ---- full-view classification ----
            if method in ("B0_CC", "B0_MLO"):
                m = (pcc > 0.5) if method == "B0_CC" else (pmlo > 0.5)
                loss = focal_bce(logits[m], y[m], CFG.FOCAL_GAMMA) if m.any() \
                    else torch.zeros((), device=DEVICE)
            else:
                loss = focal_bce(logits, y, CFG.FOCAL_GAMMA)

            # ---- P1/P2/P3: random single-view masking (+ masked loss) ----
            drop = None
            s_logits = None
            if method in ("P1", "P1_zero", "P1_dup", "P2", "P2_zero", "P2_dup", "P3") and train:
                both = (pcc > 0.5) & (pmlo > 0.5)
                drop = both & (torch.rand(B, device=DEVICE) < CFG.MASK_PROB)
                drop_cc = drop & (torch.rand(B, device=DEVICE) < 0.5)
                drop_mlo = drop & (~drop_cc)
                s_pcc = pcc * (1 - drop_cc.float())
                s_pmlo = pmlo * (1 - drop_mlo.float())
                if drop.any():
                    s_logits = model(cc, mlo, s_pcc, s_pmlo)
                    loss = loss + CFG.LAMBDA_MASKED * focal_bce(s_logits[drop], y[drop],
                                                                CFG.FOCAL_GAMMA)

            # ---- P2/P3: full-view teacher -> single-view student KD ----
            # Teacher == full-view logits (100% view completeness => the teacher pass
            # is exactly the full-view forward already computed above). Reusing
            # logits.detach() removes a redundant 3rd backbone pass -> halves P2 GPU
            # peak (was OOM at 768/batch16) and cuts P2 compute by ~1/3.
            if method in ("P2", "P2_zero", "P2_dup", "P3") and train and drop is not None and drop.any():
                loss = loss + CFG.LAMBDA_KD * binary_kl_distill(logits.detach()[drop],
                                                                s_logits[drop], CFG.KD_T)

            # ---- P3: GroupDRO over machine groups (additive term) ----
            if method == "P3" and train and groupdro is not None \
                    and machine_to_group and groupdro.n_groups > 1:
                fl_full = focal_bce(logits, y, CFG.FOCAL_GAMMA, reduction="none")
                mach = batch["machine"]
                groups = torch.tensor(
                    [machine_to_group.get(int(mi), -1) for mi in mach.cpu().tolist()],
                    device=DEVICE)
                valid = groups >= 0
                if valid.any():
                    loss = loss + CFG.LAMBDA_GROUP * groupdro.forward(fl_full[valid],
                                                                      groups[valid])

        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        tot += float(loss.item()) * len(y)
        n_seen += len(y)
        ys.append(y.detach().cpu().numpy())
        ps.append(torch.sigmoid(logits).detach().cpu().numpy())

    avg_loss = tot / max(1, n_seen)
    return avg_loss, np.concatenate(ys), np.concatenate(ps)


def train_model(train_tbl, val_tbl, method, seed, machine_to_group=None):
    """Train one study variant with early stopping on dev AUPRC.

    Returns the best model (weights restored from the best AUPRC checkpoint).
    """
    C.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = build_model(method, CFG).to(DEVICE)
    # Layered LR: slow fine-tune of the pretrained backbone, faster head training.
    backbone_p = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    head_p = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([
        {"params": backbone_p, "lr": CFG.BACKBONE_LR},
        {"params": head_p, "lr": CFG.HEAD_LR},
    ], weight_decay=CFG.WD)
    # cosine LR schedule with linear warmup (single schedule over all groups)
    if getattr(CFG, "USE_COSINE", True):
        warm = max(1, int(getattr(CFG, "WARMUP_EPOCHS", 3)))
        total = max(1, int(CFG.EPOCHS))

        def _lr_lambda(ep):
            if ep < warm:
                return (ep + 1) / warm
            t = (ep - warm) / max(1, total - warm)
            return 0.5 * (1 + math.cos(math.pi * t))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    else:
        sched = None
    tl = C.make_loader(train_tbl, train=True, balanced=True)
    vl = C.make_loader(val_tbl, train=False)

    n_groups = len(machine_to_group) if machine_to_group else 0
    groupdro = GroupDRO(n_groups).to(DEVICE) if (method == "P3" and n_groups > 1) else None

    best_auprc = -1.0
    best_state = None
    best_epoch = 0
    patience = 0
    for ep in range(CFG.EPOCHS):
        loss, _, _ = run_epoch(model, tl, opt, method, machine_to_group, groupdro, train=True)
        _, yv, pv = run_epoch(model, vl, None, method, machine_to_group, groupdro, train=False)
        auprc = average_precision_score(yv, pv) if len(np.unique(yv)) > 1 else 0.0
        tag = "" if auprc <= best_auprc else "  *"
        print(f"[{method}] seed={seed} epoch {ep+1}/{CFG.EPOCHS}  loss={loss:.4f}  "
              f"val_AUPRC={auprc:.4f}{tag}")
        if auprc > best_auprc:
            best_auprc = auprc
            best_epoch = ep + 1
            patience = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
        if patience >= CFG.EARLY_STOP_PATIENCE:
            print(f"[{method}] early stop at epoch {best_epoch} (patience {patience})")
            break
        if sched is not None:
            sched.step()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best_state is not None:
        model.load_state_dict(best_state)
    model._best_epoch = best_epoch
    model._best_val_auprc = float(best_auprc)
    return model
