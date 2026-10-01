"""Shared utilities: DICOM preprocessing, dataset/loaders, metrics, statistics.

Deliberately kept import-light (no torch) where possible so that
``01_audit_split.py`` can reuse the view-mapping / cohort logic without a GPU
stack.  The torch-dependent parts (dataset, loaders) are guarded so that a
CPU-only metadata audit still runs.
"""
from __future__ import annotations

import math
import os
import random
import warnings

import numpy as np
import pandas as pd

import config as CFG

warnings.filterwarnings("ignore")


# ============================================================================ #
# Cohort assembly & view mapping
# ============================================================================ #
def map_channel(view: object) -> str:
    """Normalize a raw `view` string to one of two channels: 'CC' or 'MLO'."""
    v = str(view).upper().strip()
    if v in CFG.VIEW_CC:
        return "CC"
    if v in CFG.VIEW_MLO:
        return "MLO"
    # Unknown views fall back to CC channel (extremely rare in RSNA).
    return "CC"


def build_breast_table(train_csv: str) -> pd.DataFrame:
    """Build a breast-level table (one row per patient_id + laterality).

    Columns: prediction_id, patient_id, laterality, cancer, cc_image_id,
    mlo_image_id, plus every optional clinical covariate that is present
    (machine_id, site_id, age, implant, density, biopsy, invasive, BIRADS,
    difficult_negative_case).

    The label is OR-ed across the breast's images (== g.cancer.max()), which is
    the standard RSNA breast-level definition.
    """
    df = pd.read_csv(train_csv)
    df["chan"] = df["view"].map(map_channel)

    # Optional covariates that may or may not exist in this dump.
    optional_cols = [c for c in [
        "age", "implant", "density", "machine_id", "site_id",
        "biopsy", "invasive", "BIRADS", "difficult_negative_case",
    ] if c in df.columns]

    rows = []
    for (pid, lat), g in df.groupby(["patient_id", "laterality"], sort=False):
        cc   = g[g.chan == "CC"]
        mlo  = g[g.chan == "MLO"]
        row = {
            "prediction_id": f"{pid}_{lat}",
            "patient_id": int(pid),
            "laterality": lat,
            "cancer": int(g.cancer.max()) if "cancer" in g.columns else int(-1),
            "cc_image_id":  int(cc.image_id.iloc[0]) if len(cc) else None,
            "mlo_image_id": int(mlo.image_id.iloc[0]) if len(mlo) else None,
        }
        for c in optional_cols:
            row[c] = g[c].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def patient_level_split(tbl: pd.DataFrame, frac: float = 0.2, seed: int = 42):
    """Patient-level split.  frac = fraction reserved for the LOCKED test set.

    No patient ever crosses a split: all of a patient's breasts (L/R) and all
    of their views move together.  Returns a copy with a 'split' column.
    """
    rng = np.random.RandomState(seed)
    pats = np.asarray(tbl.patient_id.unique())
    rng.shuffle(pats)
    k = int(len(pats) * (1 - frac))
    train_pats = set(pats[:k].tolist())
    out = tbl.copy()
    out["split"] = np.where(out.patient_id.isin(train_pats), "train", "test")
    return out


# ============================================================================ #
# DICOM preprocessing (used only on the real data path)
# ============================================================================ #
def load_dicom_image(path: str) -> np.ndarray:
    """DICOM -> float32 HxW in [0,1]: VOI LUT, MONOCHROME1 invert, 0.5-99.5% clip."""
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    ds = pydicom.dcmread(path)
    arr = ds.pixel_array
    try:
        arr = apply_voi_lut(arr, ds)                 # Rescale/VOI LUT windowing
    except Exception:
        pass
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr                        # invert: bright = dense tissue
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [0.5, 99.5])         # intensity clipping
    arr = np.clip(arr, lo, hi)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    return arr


def crop_breast_roi(img: np.ndarray) -> np.ndarray:
    """Otsu-like threshold -> largest connected component -> bbox crop.

    Removes black borders and burned-in device annotations (prevents the model
    from latching onto border/marker pseudo-correlations).
    """
    from scipy import ndimage
    t = img > max(0.05, float(img.mean()) * 0.5)
    lbl, n = ndimage.label(t)
    if n == 0:
        return img
    sizes = ndimage.sum(np.ones_like(lbl), lbl, index=range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    mask = lbl == keep
    ys, xs = np.where(mask)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def flip_to_common_side(img: np.ndarray) -> np.ndarray:
    """Standardize orientation: put the breast mass on the LEFT for all images."""
    left_mass = img[:, : img.shape[1] // 2].sum()
    right_mass = img[:, img.shape[1] // 2:].sum()
    if right_mass > left_mass:
        img = img[:, ::-1]
    return np.ascontiguousarray(img)


def resize_pad(img: np.ndarray, size: int) -> np.ndarray:
    """Aspect-preserving resize + zero pad to (size, size)."""
    import cv2
    h, w = img.shape
    s = size / max(h, w)
    nh, nw = max(1, int(h * s)), max(1, int(w * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), np.float32)
    out[:nh, :nw] = r
    return out


def preprocess_real(path: str, size: int) -> np.ndarray:
    img = load_dicom_image(path)
    img = crop_breast_roi(img)
    img = flip_to_common_side(img)
    img = resize_pad(img, size)
    return img.astype(np.float32)


def dcm_path(image_id, patient_id) -> str:
    return os.path.join(CFG.TRAIN_IMAGES_DIR, str(int(patient_id)), f"{int(image_id)}.dcm")


_IMG_CACHE = {}  # (image_id, size) -> uint8 array; in-RAM to avoid repeated disk reads


def cached_image(image_id, patient_id, size: int) -> np.ndarray:
    """Decode a DICOM once, cache the preprocessed uint8 on disk AND in RAM."""
    key = (int(image_id), int(size))
    arr = _IMG_CACHE.get(key)
    if arr is not None:
        return arr.astype(np.float32) / 255.0
    cp = os.path.join(CFG.CACHE_DIR, f"{int(image_id)}_{size}.npy")
    if os.path.exists(cp):
        try:
            arr = np.load(cp)
        except Exception:
            arr = None
    if arr is None:
        img = preprocess_real(dcm_path(image_id, patient_id), size)
        arr = np.clip(img * 255, 0, 255).astype(np.uint8)
        try:
            np.save(cp, arr)
        except Exception:
            pass
    _IMG_CACHE[key] = arr
    return arr.astype(np.float32) / 255.0


def _augment_view(img: np.ndarray) -> np.ndarray:
    """Random small affine (rotate/scale/translate) + contrast/brightness.

    img: float32 HxW in [0,1] (background = 0).  Never vertical-flip.
    """
    import cv2
    h, w = img.shape
    ang = random.uniform(-10, 10)
    scale = random.uniform(0.9, 1.1)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale)
    m[0, 2] += random.uniform(-0.06, 0.06) * w
    m[1, 2] += random.uniform(-0.06, 0.06) * h
    img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    if random.random() < 0.5:
        alpha = random.uniform(0.85, 1.15)      # contrast
        beta = random.uniform(-0.06, 0.06)      # brightness
        img = np.clip(alpha * img + beta, 0.0, 1.0)
    return img


# ============================================================================ #
# torch dataset / loaders (imported lazily so metadata audit needs no torch)
# ============================================================================ #
def make_breast_dataset(tbl: pd.DataFrame, train: bool = True):
    import torch
    from torch.utils.data import Dataset

    class BreastDataset(Dataset):
        def __init__(self, t, train):
            self.tbl = t.reset_index(drop=True)
            self.train = train

        def __len__(self):
            return len(self.tbl)

        def _img(self, image_id, y):
            present = image_id is not None and not (isinstance(image_id, float) and np.isnan(image_id))
            if present:
                try:
                    return cached_image(image_id, self._pid, CFG.IMG_SIZE), 1
                except Exception:
                    return np.zeros((CFG.IMG_SIZE, CFG.IMG_SIZE), np.float32), 0
            return np.zeros((CFG.IMG_SIZE, CFG.IMG_SIZE), np.float32), 0

        def __getitem__(self, i):
            r = self.tbl.iloc[i]
            self._pid = r.patient_id
            y = int(r.cancer)
            cc, pcc   = self._img(r.cc_image_id, y)
            mlo, pmlo = self._img(r.mlo_image_id, y)
            if self.train and random.random() < 0.5:
                cc, mlo = cc[:, ::-1].copy(), mlo[:, ::-1].copy()
            if self.train and getattr(CFG, "AUGMENT", True):
                cc = _augment_view(cc)
                mlo = _augment_view(mlo)
            cc  = torch.from_numpy(cc).unsqueeze(0).repeat(3, 1, 1)
            mlo = torch.from_numpy(mlo).unsqueeze(0).repeat(3, 1, 1)
            out = dict(
                cc=cc, mlo=mlo,
                pcc=torch.tensor(pcc, dtype=torch.float32),
                pmlo=torch.tensor(pmlo, dtype=torch.float32),
                y=torch.tensor(y, dtype=torch.float32),
                machine=torch.tensor(int(r.get("machine_id", -1)) if pd.notna(r.get("machine_id", -1)) else -1),
            )
            return out

    return BreastDataset(tbl, train)


def make_loader(tbl: pd.DataFrame, train: bool = True, balanced: bool = True, batch: int | None = None):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    batch = batch or CFG.BATCH
    ds = make_breast_dataset(tbl, train=train)
    if train and balanced:
        y = tbl.cancer.values.astype(float)
        npos = max(1, float(y.sum())); nneg = max(1, float((1 - y).sum()))
        pf = float(getattr(CFG, "POS_SAMPLE_FRAC", 0.5))
        w = np.where(y == 1, pf / npos, (1.0 - pf) / nneg)
        sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double),
                                        num_samples=len(tbl), replacement=True)
        # drop_last=True prevents a size-1 final batch from breaking BatchNorm.
        return DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=0,
                          pin_memory=False, drop_last=True)
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0, pin_memory=False)


def _precache_one(job):
    iid, pid = job
    try:
        cached_image(iid, pid, CFG.IMG_SIZE)
        return 1
    except Exception:
        return 0


def precache(tbl: pd.DataFrame, workers: int = 8) -> int:
    """Pre-decode every DICOM referenced by `tbl` into the on-disk cache.

    DICOM decode is CPU-bound (~1.4 s/img single-threaded), so it is parallelised
    with a process pool.  Pass workers=1 to run serially.  Returns decoded count.
    """
    jobs = []
    for _, r in tbl.iterrows():
        for col in ("cc_image_id", "mlo_image_id"):
            iid = r[col]
            if iid is not None and not (isinstance(iid, float) and np.isnan(iid)):
                jobs.append((int(iid), int(r.patient_id)))
    jobs = list(dict.fromkeys(jobs))
    from tqdm.auto import tqdm
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for ok in tqdm(ex.map(_precache_one, jobs, chunksize=32),
                           total=len(jobs), desc="Pre-caching DICOMs"):
                done += ok
        return done
    done = 0
    for j in tqdm(jobs, desc="Pre-caching DICOMs"):
        done += _precache_one(j)
    return done


# ============================================================================ #
# Metrics
# ============================================================================ #
def pfbeta(y_true, y_prob, beta: float = 1.0) -> float:
    """RSNA official probabilistic F1 (threshold-free). Kept for reporting parity."""
    y_true = np.asarray(y_true, float)
    y_prob = np.clip(np.asarray(y_prob, float), 0, 1)
    ctp = (y_prob * y_true).sum()
    cfp = (y_prob * (1 - y_true)).sum()
    if (ctp + cfp) == 0:
        return 0.0
    prec = ctp / (ctp + cfp)
    rec = ctp / (y_true.sum() + 1e-9)
    if prec + rec == 0:
        return 0.0
    b2 = beta * beta
    return float((1 + b2) * prec * rec / (b2 * prec + rec + 1e-9))


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1])
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def _is_binary(y) -> bool:
    return len(np.unique(np.asarray(y))) > 1


def all_metrics(y, p) -> dict:
    """Primary = AUPRC; secondary = AUROC, pF1, Brier, ECE, + operating points."""
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 brier_score_loss)
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    out = dict(
        AUPRC=average_precision_score(y, p) if _is_binary(y) else float("nan"),
        AUROC=roc_auc_score(y, p) if _is_binary(y) else float("nan"),
        pF1=pfbeta(y, p),
        Brier=brier_score_loss(y, p) if _is_binary(y) else float("nan"),
        ECE=expected_calibration_error(y, p),
    )
    out.update(operating_point_metrics(y, p))
    return out


def operating_point_metrics(y, p) -> dict:
    """Sensitivity at 90% specificity, specificity at 80% sensitivity, PPV/NPV."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if not _is_binary(y):
        return dict(Sens_at_90Spec=float("nan"), Spec_at_80Sens=float("nan"),
                    PPV=float("nan"), NPV=float("nan"))

    # Sensitivity at a fixed (90%) specificity
    thr = _threshold_for_specificity(y, p, CFG.FIXED_SPECIFICITY)
    pred = (p >= thr).astype(int)
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    sens_at_90spec = tp / (tp + fn + 1e-9)

    # Specificity at a fixed (80%) sensitivity
    thr_s = _threshold_for_sensitivity(y, p, CFG.FIXED_SENSITIVITY)
    pred_s = (p >= thr_s).astype(int)
    tp_s = ((pred_s == 1) & (y == 1)).sum()
    fp_s = ((pred_s == 1) & (y == 0)).sum()
    fn_s = ((pred_s == 0) & (y == 1)).sum()
    tn_s = ((pred_s == 0) & (y == 0)).sum()
    spec_at_80sens = tn_s / (tn_s + fp_s + 1e-9)

    ppv = tp / (tp + fp + 1e-9)
    npv = tn / (tn + fn + 1e-9)
    return dict(Sens_at_90Spec=float(sens_at_90spec), Spec_at_80Sens=float(spec_at_80sens),
                PPV=float(ppv), NPV=float(npv))


def _threshold_for_specificity(y, p, target_spec: float) -> float:
    """Threshold achieving ~target specificity (first threshold crossing it when
    walking probabilities from high to low)."""
    order = np.argsort(-p)                      # high -> low
    n_neg = int((y == 0).sum())
    if n_neg == 0:
        return 0.0
    fp = 0
    for idx in order:
        if y[idx] == 0:
            fp += 1
        spec = (n_neg - fp) / n_neg
        if spec <= target_spec:
            return float(p[idx])
    return 0.0


def _threshold_for_sensitivity(y, p, target_sens: float) -> float:
    order = np.argsort(-p)
    n_pos = (y == 1).sum()
    tp = 0
    for idx in order:
        if y[idx] == 1:
            tp += 1
            sens = tp / n_pos if n_pos else 0.0
            if sens >= target_sens:
                return p[idx]
    return 0.0


# ============================================================================ #
# Statistics
# ============================================================================ #
def bootstrap_ci(y, p, patients, fn, n: int = 2000, seed: int = 42):
    """Patient-level (cluster) bootstrap 95% CI for metric fn(y,p)."""
    y = np.asarray(y); p = np.asarray(p)
    patients = np.asarray(patients)
    rng = np.random.RandomState(seed)
    uniq = np.unique(patients)
    idx_by_p = {u: np.where(patients == u)[0] for u in uniq}
    stats = []
    for _ in range(n):
        samp = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_by_p[u] for u in samp])
        try:
            stats.append(fn(y[idx], p[idx]))
        except Exception:
            pass
    stats = np.asarray(stats)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(stats.mean()), float(lo), float(hi)


def paired_delta_ci(y, pA, pB, patients, fn, n: int = 2000, seed: int = 42):
    """Patient-level bootstrap 95% CI of the paired difference fn(B) - fn(A)."""
    y = np.asarray(y); pA = np.asarray(pA); pB = np.asarray(pB)
    patients = np.asarray(patients)
    rng = np.random.RandomState(seed)
    uniq = np.unique(patients)
    idx_by_p = {u: np.where(patients == u)[0] for u in uniq}
    d = []
    for _ in range(n):
        samp = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_by_p[u] for u in samp])
        try:
            d.append(fn(y[idx], pB[idx]) - fn(y[idx], pA[idx]))
        except Exception:
            pass
    d = np.asarray(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def delong_auc_ci(y, p):
    """DeLong 95% CI for a single AUROC (fast analytic variance)."""
    y = np.asarray(y); p = np.asarray(p)
    pos = p[y == 1]; neg = p[y == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan"), float("nan"), float("nan")

    def midrank(x):
        order = np.argsort(x)
        r = np.empty(len(x)); s = x[order]
        i = 0
        while i < len(x):
            j = i
            while j < len(x) and s[j] == s[i]:
                j += 1
            r[order[i:j]] = 0.5 * (i + j - 1) + 1
            i = j
        return r

    allp = np.concatenate([pos, neg]); r = midrank(allp)
    rpos, rneg = r[:m], r[m:]
    auc = (rpos.sum() - m * (m + 1) / 2) / (m * n)
    v01 = (rpos - midrank(pos)) / n
    v10 = 1 - (rneg - midrank(neg)) / m
    var = np.var(v01, ddof=1) / m + np.var(v10, ddof=1) / n
    se = math.sqrt(max(var, 0.0)); z = 1.959964
    return float(auc), float(max(0.0, auc - z * se)), float(min(1.0, auc + z * se))


def mcnemar_test(y, predA, predB):
    """McNemar test (with continuity correction) on paired binary predictions.

    Returns (chi2_stat, p_value).  Discordant pairs are the two cells where the
    models disagree.
    """
    from scipy.stats import chi2
    y = np.asarray(y, bool); a = np.asarray(predA, bool); b = np.asarray(predB, bool)
    b_cell = int(((a == 1) & (b == 0)).sum())
    c_cell = int(((a == 0) & (b == 1)).sum())
    denom = b_cell + c_cell
    if denom == 0:
        return 0.0, 1.0
    stat = (abs(b_cell - c_cell) - 1) ** 2 / denom
    return float(stat), float(chi2.sf(stat, 1))


def holm_correction(pvals):
    """Holm-Bonferroni step-down adjusted p-values (ascending-order version)."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    for rank, idx in enumerate(order):
        adj[idx] = min(1.0, p[idx] * (n - rank))
    for rank in range(1, n):
        adj[order[rank]] = max(adj[order[rank]], adj[order[rank - 1]])
    return adj


# ============================================================================ #
# Small helpers
# ============================================================================ #
def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    except Exception:
        pass
