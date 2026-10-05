import os, sys, gc
sys.path.insert(0, "/root/breast_sci_out")
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score, roc_auc_score
import config as CFG
import lib_common as C
import lib_models as M
from lib_train import DEVICE

CKPT = "/root/breast_sci_out/checkpoints"
OUT = "/root/breast_sci_out/results/m5_token_sensitivity.txt"
print("start M5 final", flush=True)

p2 = M.build_model("P2", CFG).to(DEVICE)
p2.load_state_dict(torch.load(f"{CKPT}/model_P2_seed42.pt", map_location=DEVICE))
p2tok = p2.missing_token.data.detach().cpu().clone()
del p2; gc.collect(); torch.cuda.empty_cache()
print("extracted P2 token", flush=True)

test_tbl = pd.read_csv("/root/breast_sci_out/splits/test_breast.csv")
b2 = M.build_model("B2", CFG).to(DEVICE)
b2.load_state_dict(torch.load(f"{CKPT}/model_B2_seed42.pt", map_location=DEVICE))
b2.eval()

loader = C.make_loader(test_tbl, train=False)
n_batches = len(loader)
ys = []
acc = {"cc_zero":[], "mlo_zero":[], "cc_p2":[], "mlo_p2":[]}
with torch.no_grad():
    for bi, b in enumerate(loader):
        cc = b["cc"].to(DEVICE); mlo = b["mlo"].to(DEVICE)
        pcc = b["pcc"].to(DEVICE); pmlo = b["pmlo"].to(DEVICE)
        y = b["y"].numpy()
        zc = torch.zeros_like(pcc); zm = torch.zeros_like(pmlo)
        b2.missing_token.data.zero_()
        acc["cc_zero"].append(torch.sigmoid(b2(cc, mlo, pcc, zm)).cpu().numpy())
        acc["mlo_zero"].append(torch.sigmoid(b2(cc, mlo, zc, pmlo)).cpu().numpy())
        b2.missing_token.data.copy_(p2tok.to(DEVICE))
        acc["cc_p2"].append(torch.sigmoid(b2(cc, mlo, pcc, zm)).cpu().numpy())
        acc["mlo_p2"].append(torch.sigmoid(b2(cc, mlo, zc, pmlo)).cpu().numpy())
        ys.append(y)
        if (bi + 1) % 25 == 0:
            print("  batch %d/%d" % (bi + 1, n_batches), flush=True)

y = np.concatenate(ys)
lines = ["=== B2 single-view under zero / P2-learned missing token (seed 42) ===",
         "(reference: default random token cc=0.1323 mlo=0.0752)"]
for k in ["cc_zero","mlo_zero","cc_p2","mlo_p2"]:
    p = np.concatenate(acc[k])
    lines.append("  %-9s AUPRC=%.4f  AUROC=%.4f" % (k, average_precision_score(y, p), roc_auc_score(y, p)))
lines.append("M5 DONE")
txt = "\n".join(lines)
open(OUT, "w").write(txt)
print(txt, flush=True)
print("saved to", OUT, flush=True)
