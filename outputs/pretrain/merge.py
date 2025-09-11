# make_index.py
import os, sys, glob, json
from safetensors import safe_open

os.chdir(os.path.dirname(__file__))
d = sys.argv[1] if len(sys.argv) > 1 else "."
parts = sorted(glob.glob(os.path.join(d, "model-*-of-*.safetensors")))
assert parts, f"No shards found under {d}"

weight_map = {}
total_size = 0
for p in parts:
    bn = os.path.basename(p)
    total_size += os.path.getsize(p)  # 直接用文件大小做总大小统计即可
    with safe_open(p, framework="pt", device="cpu") as f:
        for k in f.keys():            # 不会把 tensor 读进内存
            weight_map[k] = bn

index = {
    "metadata": {"total_size": total_size},
    "weight_map": weight_map,
}
out = os.path.join(d, "model.safetensors.index.json")
with open(out, "w") as f:
    json.dump(index, f)
print("Wrote", out, "with", len(weight_map), "tensors")
