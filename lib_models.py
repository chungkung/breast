"""Model zoo: shared backbone + the B0/B1/B2/P1/P2/P3 study variants.

Architecture (B2/P1/P2/P3): a shared ImageNet-pretrained backbone encodes each
view -> 512-d view token -> 2-layer / 8-head Transformer over [CLS, CC, MLO] ->
breast-level cancer logit.  A missing view is replaced by a learnable missing
token.  P1/P2/P3 differ only in training objective (masking / distillation /
GroupDRO), not in architecture, which keeps ablations clean.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import config as CFG


# --------------------------------------------------------------------------- #
# Backbone factory
# --------------------------------------------------------------------------- #
def _conv3x3(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_c, out_c, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(in_c, out_c, stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_c, out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.downsample = downsample

    def forward(self, x):
        i = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.downsample is not None:
            i = self.downsample(i)
        return self.relu(x + i)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_c, out_c, stride=1, downsample=None):
        super().__init__()
        width = out_c
        self.conv1 = nn.Conv2d(in_c, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = _conv3x3(width, width, stride)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_c * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        i = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.downsample is not None:
            i = self.downsample(i)
        return self.relu(x + i)


class _ResNet(nn.Module):
    """Minimal torchvision-compatible ResNet (state_dict keys match torchvision)."""

    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make(block, 64, layers[0])
        self.layer2 = self._make(block, 128, layers[1], stride=2)
        self.layer3 = self._make(block, 256, layers[2], stride=2)
        self.layer4 = self._make(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


_TORCH_RESNET_URLS = {
    "resnet18": "resnet18-f37072fd.pth",
    "resnet34": "resnet34-b627a593.pth",
    "resnet50": "resnet50-0676ba61.pth",
    "resnet101": "resnet101-63fe2227.pth",
}


def _torch_resnet(name, pretrained):
    """Pure-torch ResNet (no timm/torchvision).  Downloads ImageNet V1 weights."""
    import torch.hub
    arch = {"resnet18": [2, 2, 2, 2], "resnet34": [3, 4, 6, 3],
            "resnet50": [3, 4, 6, 3], "resnet101": [3, 4, 23, 3]}[name]
    block = _BasicBlock if name in ("resnet18", "resnet34") else _Bottleneck
    model = _ResNet(block, arch, num_classes=1000)
    if pretrained:
        try:
            sd = torch.hub.load_state_dict_from_url(
                "https://download.pytorch.org/models/" + _TORCH_RESNET_URLS[name],
                progress=False)
            model.load_state_dict(sd)
        except Exception:
            pass
    feat = model.fc.in_features
    model.fc = nn.Identity()
    return model, int(feat)


def get_backbone(name: str, pretrained: bool = True):
    """Return (backbone_module, feature_dim).  Tries timm, then torchvision, then
    a pure-torch ResNet fallback (used on Hygon DCU, where the PyPI torchvision
    build is ABI-incompatible with the DCU torch build)."""
    try:
        import timm
        try:
            m = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
        except Exception:
            m = timm.create_model(name, pretrained=False, num_classes=0, global_pool="avg")
        return m, int(m.num_features)
    except Exception:
        pass

    try:
        import torchvision
        try:
            m = getattr(torchvision.models, name)(weights="IMAGENET1K_V1" if pretrained else None)
        except Exception:
            m = getattr(torchvision.models, name)(weights=None)
        if hasattr(m, "fc"):
            feat = m.fc.in_features
            m.fc = nn.Identity()
        elif hasattr(m, "classifier"):
            feat = m.classifier[-1].in_features if hasattr(m.classifier[-1], "in_features") else 512
            m.classifier = nn.Identity()
        else:
            feat = 512
        return m, int(feat)
    except Exception:
        pass

    if name.startswith("resnet"):
        return _torch_resnet(name, pretrained)
    raise RuntimeError(
        f"Backbone '{name}' unavailable (no timm/torchvision and no pure-torch "
        f"fallback); install a compatible backbone library.")


# --------------------------------------------------------------------------- #
# B0: single-view independent model
# --------------------------------------------------------------------------- #
class SingleViewNet(nn.Module):
    """Single-view encoder (B0).  Trained and evaluated on one view at a time."""

    def __init__(self, cfg, view: str = "CC"):
        super().__init__()
        self.view = view
        self.backbone, feat = get_backbone(cfg.BACKBONE, cfg.PRETRAINED)
        self.proj = nn.Linear(feat, cfg.D_MODEL)
        self.head = nn.Linear(cfg.D_MODEL, 1)

    def encode(self, img):
        return self.proj(self.backbone(img))

    def forward(self, cc, mlo, pcc, pmlo):
        img = cc if self.view == "CC" else mlo
        return self.head(self.encode(img)).squeeze(1)


# --------------------------------------------------------------------------- #
# B1: probability-average fusion (late fusion)
# --------------------------------------------------------------------------- #
class ProbAvgNet(nn.Module):
    """Shared backbone + per-view heads; final prob = mean over present views."""

    def __init__(self, cfg):
        super().__init__()
        self.backbone, feat = get_backbone(cfg.BACKBONE, cfg.PRETRAINED)
        self.proj = nn.Linear(feat, cfg.D_MODEL)
        self.cc_head = nn.Linear(cfg.D_MODEL, 1)
        self.mlo_head = nn.Linear(cfg.D_MODEL, 1)

    def forward(self, cc, mlo, pcc, pmlo):
        cc_t = self.proj(self.backbone(cc))
        mlo_t = self.proj(self.backbone(mlo))
        cc_p = torch.sigmoid(self.cc_head(cc_t)).squeeze(1)   # [B]
        mlo_p = torch.sigmoid(self.mlo_head(mlo_t)).squeeze(1)  # [B]
        num = pcc * cc_p + pmlo * mlo_p
        den = pcc + pmlo + 1e-6
        p = torch.clamp(num / den, 1e-5, 1 - 1e-5)
        return torch.logit(p)


# --------------------------------------------------------------------------- #
# B2 / P1 / P2 / P3: shared backbone + Transformer view fusion + missing token
# --------------------------------------------------------------------------- #
class MissingViewFusion(nn.Module):
    def __init__(self, cfg, use_missing_token: bool = True, missing_mode: str = "learned"):
        super().__init__()
        self.backbone, feat = get_backbone(cfg.BACKBONE, cfg.PRETRAINED)
        self.proj = nn.Linear(feat, cfg.D_MODEL)
        self.missing_mode = missing_mode
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.D_MODEL))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if missing_mode == "dup":
            # Ablation: missing view token -> duplicate of the surviving view's token.
            self.register_buffer("missing_token", torch.zeros(1, 1, cfg.D_MODEL))
        elif missing_mode == "zero":
            # Ablation control: missing view -> fixed zero token (no learnable prior).
            # NOTE: must be a plain buffer, never nn.Parameter, or it would be a
            # second learnable token and be identical to the "learned" variant.
            self.register_buffer("missing_token", torch.zeros(1, 1, cfg.D_MODEL))
        elif use_missing_token:
            self.missing_token = nn.Parameter(torch.zeros(1, 1, cfg.D_MODEL))
            nn.init.trunc_normal_(self.missing_token, std=0.02)
        else:
            # Defensive fallback: fixed zero token.
            self.register_buffer("missing_token", torch.zeros(1, 1, cfg.D_MODEL))
        enc = nn.TransformerEncoderLayer(cfg.D_MODEL, cfg.N_HEADS, cfg.D_MODEL * 2,
                                         batch_first=True, dropout=cfg.DROP_RATE)
        self.tr = nn.TransformerEncoder(enc, cfg.N_LAYERS)
        self.head = nn.Linear(cfg.D_MODEL, 1)

    def encode(self, img):
        return self.proj(self.backbone(img))

    def forward(self, cc, mlo, pcc, pmlo):
        B = cc.size(0)
        cc_t = self.encode(cc)
        mlo_t = self.encode(mlo)
        if self.missing_mode == "dup":
            cc_t_orig, mlo_t_orig = cc_t, mlo_t
            cc_t = torch.where(pcc.unsqueeze(1) > 0.5, cc_t_orig, mlo_t_orig)
            mlo_t = torch.where(pmlo.unsqueeze(1) > 0.5, mlo_t_orig, cc_t_orig)
        else:
            miss = self.missing_token.expand(B, 1, -1).squeeze(1)
            cc_t = torch.where(pcc.unsqueeze(1) > 0.5, cc_t, miss)
            mlo_t = torch.where(pmlo.unsqueeze(1) > 0.5, mlo_t, miss)
        cls = self.cls_token.expand(B, 1, -1)
        tok = torch.cat([cls, cc_t.unsqueeze(1), mlo_t.unsqueeze(1)], dim=1)
        out = self.tr(tok)[:, 0]
        return self.head(out).squeeze(1)


# --------------------------------------------------------------------------- #
# GroupDRO adversarial reweighting (P3 / E5)
# --------------------------------------------------------------------------- #
class GroupDRO:
    """Group Distributionally Robust Optimization over machine_id groups.

    Maintains an adversarial distribution q over groups; at each step q is
    updated toward groups with higher loss, then the group losses are combined
    as sum_g q_g * L_g.
    """

    def __init__(self, n_groups: int, eta: float = CFG.GROUP_DRO_STEP_SIZE):
        self.n_groups = n_groups
        self.eta = eta
        self.q = torch.ones(n_groups) / n_groups

    def to(self, device):
        self.q = self.q.to(device)
        return self

    def forward(self, losses, groups):
        """losses: (B,) per-sample; groups: (B,) contiguous group ids in [0,n)."""
        device = losses.device
        self.q = self.q.to(device)
        group_loss = torch.zeros(self.n_groups, device=device)
        for g in range(self.n_groups):
            m = groups == g
            if m.any():
                group_loss[g] = losses[m].mean()
        with torch.no_grad():
            self.q = self.q * torch.exp(self.eta * group_loss)
            self.q = self.q / (self.q.sum() + 1e-9)
        return (self.q * group_loss).sum()


# --------------------------------------------------------------------------- #
# Factory + machine->group index mapping
# --------------------------------------------------------------------------- #
def build_model(method: str, cfg) -> nn.Module:
    """Return the nn.Module for a study variant.

    method in {'B0_CC','B0_MLO','B1','B2','P1','P2','P3'}.
    B2/P1/P2/P3 share the identical architecture (differences are in training).
    """
    if method in ("B0_CC", "B0_MLO"):
        return SingleViewNet(cfg, view="CC" if method == "B0_CC" else "MLO")
    if method == "B1":
        return ProbAvgNet(cfg)
    if method in ("P1_zero", "P2_zero"):
        return MissingViewFusion(cfg, missing_mode="zero")
    if method in ("P1_dup", "P2_dup"):
        return MissingViewFusion(cfg, missing_mode="dup")
    return MissingViewFusion(cfg)


def build_machine_groups(train_tbl):
    """Map each machine_id to a contiguous group index [0, n_groups).

    Returns (machine_to_group, n_groups).  Unknown/missing machine_id -> -1 and
    is excluded from GroupDRO reweighting (but still trained normally).
    """
    vals = []
    if "machine_id" in train_tbl.columns:
        vals = [v for v in train_tbl.machine_id.dropna().unique().tolist()]
    machine_to_group = {v: i for i, v in enumerate(sorted(vals))}
    return machine_to_group, len(machine_to_group)
