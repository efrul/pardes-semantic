"""Can the student be trained on Kaggle's TPU (PyTorch/XLA), and how fast? A Kaggle kernel: a few steps of
MultipleNegativesRankingLoss on one chip of the TPU v5e-8, every text padded to a fixed length (XLA compiles once for
each shape), against tune12's 11.6 rows a second (anchor, positive and negative) on the 2 T4s.

Run 9 (2026-11-27, RUN5_PLAN.md, 3): 165 rows a second with a batch of 32, 239 with a batch of 64; each new shape
compiles in 22 s; the whole run took 6 minutes of the TPU quota. Kaggle's TPU queue is long: run 9 waited 9 hours,
run 8 20.

    mkdir -p build/kaggle/tpuprobe && cp tpu_probe.py build/kaggle/tpuprobe/probe.py
    python3 tpu_probe.py --metadata build/kaggle/tpuprobe     # its owner from KAGGLE_USERNAME, never written here
    kaggle kernels push -p build/kaggle/tpuprobe
    kaggle kernels output "$KAGGLE_USERNAME/pardes-semantic-tpuprobe" -p build/kaggle-output/tpuprobe

It trains the model of the tune kernel's output (kernel_sources: pardes-semantic-tune12) and prints its numbers on a
RESULT line.
"""
import glob, json, os, subprocess, sys, time

if sys.argv[1:2] == ["--metadata"]:
    user = os.environ["KAGGLE_USERNAME"]
    with open(os.path.join(sys.argv[2], "kernel-metadata.json"), "w") as f:
        json.dump({"id": f"{user}/pardes-semantic-tpuprobe", "title": "pardes-semantic-tpuprobe", "code_file": "probe.py",
                   "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False,
                   "enable_tpu": True, "machine_shape": "TpuV5E8", "enable_internet": True, "dataset_sources": [],
                   "competition_sources": [], "kernel_sources": [f"{user}/pardes-semantic-tune12"]}, f, indent=1)
    sys.exit(0)

# The TPU image's TensorFlow does not import (an undefined symbol), and transformers imports it when it is there
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"
os.environ["USE_FLAX"] = "0"

def sh(c):
    print("$", c, flush=True)
    subprocess.run(c, shell=True, check=False)

# Is there a TPU at all? The devices, the names of the TPU's variables (not their values), and JAX's view of it
sh("ls /dev | grep -i -E 'accel|vfio' ; env | grep -i -E 'tpu|pjrt|xla' | sed 's/=.*//'")
sh("timeout 300 python -c \"import jax; print('jax devices', jax.devices())\"")
sh("python -V; pip list 2>/dev/null | grep -i -E '^(torch|jax|jaxlib|libtpu|tensorflow|flax|transformers) '")
# Kaggle's TPU image has no PyTorch/XLA: the newest torch_xla with the torch it needs, then the pinned stack
# torch_xla 2.9.0 needs libtpu 0.0.21, whose wheels are for Python 3.10 to 3.12 (the image has 3.13); 2.8.1 needs libtpu
# 0.0.17, a wheel for any Python 3, and torch 2.8.0 exactly (the image has 2.11)
sh("pip -q install 'torch==2.8.0' 'torch_xla[tpu]==2.8.1' --extra-index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -4")
# The image's datasets 5.2 needs huggingface-hub 1.31+, which transformers 4.57 refuses: one that takes the hub 0.36
sh("pip -q install 'transformers==4.57.6' 'tokenizers>=0.22,<0.23' 'sentence-transformers==5.1.2' 'datasets==3.6.0' "
   "'huggingface-hub>=0.34,<1.0' 2>&1 | tail -3")
sh("python -c \"import torch, torch_xla; print('torch', torch.__version__, 'torch_xla', torch_xla.__version__)\"")
if glob.glob("/dev/accel*") or glob.glob("/dev/vfio/*"):
    os.environ.setdefault("PJRT_DEVICE", "TPU")
print("PJRT_DEVICE", os.environ.get("PJRT_DEVICE"), flush=True)
import torch
import torch_xla
import torch_xla.core.xla_model as xm

dev = torch_xla.device()
print("device", dev, xm.xla_device_hw(dev), "devices", len(xm.get_xla_supported_devices()), flush=True)
if xm.xla_device_hw(dev) == "CPU":
    print("RESULT no TPU in this session", flush=True)
    sys.exit(0)
src = os.path.dirname(sorted(glob.glob("/kaggle/input/**/tuned/config.json", recursive=True))[0])
print("model", src.split("/")[-2:], flush=True)
from sentence_transformers import SentenceTransformer, losses

model = SentenceTransformer(src, device="cpu", trust_remote_code=True)
model.to(dev)
model.train()
tok = model.tokenizer
loss_fn = losses.MultipleNegativesRankingLoss(model)
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
out = {}
for B, LQ, LD in ((32, 64, 256), (64, 64, 256)):
    q = [f"query: מה הדין במי ששכח לומר יעלה ויבוא {i}" for i in range(B)]
    d = [("ואם שכח ולא הזכיר מחזירין אותו " * 30) + str(i) for i in range(B)]

    def feats(texts, L):
        f = tok(texts, padding="max_length", truncation=True, max_length=L, return_tensors="pt")
        return {k: v.to(dev) for k, v in f.items()}

    fq, fd, fn = feats(q, LQ), feats(d, LD), feats(d[::-1], LD)
    times = []
    try:
        for step in range(10):
            t = time.time()
            with torch.autocast("xla", dtype=torch.bfloat16):
                loss = loss_fn([fq, fd, fn], None)
            loss.backward()
            opt.step()
            opt.zero_grad()
            torch_xla.sync()
            value = loss.item()
            times.append(time.time() - t)
            print(B, step, round(times[-1], 2), "s", round(value, 4), flush=True)
        steady = sorted(times[3:])[len(times[3:]) // 2]
        out[f"batch {B}"] = {"step_s": round(steady, 3), "rows_per_s_one_chip": round(B / steady, 1),
                             "first_step_s": round(times[0], 1)}
    except Exception as e:
        out[f"batch {B}"] = {"error": repr(e)[:400]}
        print("FAILED", repr(e)[:2000], flush=True)
print("RESULT", json.dumps(out), flush=True)
