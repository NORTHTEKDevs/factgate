"""Isolated dtype support check: usage `python bench_directml_dtype.py bf16|fp16`.
Run each in its OWN process since an unsupported dtype can hard-crash the DML native backend."""
import sys, json, traceback

dtype_name = sys.argv[1] if len(sys.argv) > 1 else "bf16"
import torch
dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
dtype = dtype_map[dtype_name]

result = {"dtype": dtype_name, "supported": None, "note": None}
try:
    import torch_directml
    dml = torch_directml.device()
    a = torch.randn(64, 64, dtype=dtype, device=dml)
    b = torch.randn(64, 64, dtype=dtype, device=dml)
    c = a @ b
    c_cpu = c.to("cpu").float()
    result["supported"] = True
except Exception as e:
    result["supported"] = False
    result["note"] = f"{type(e).__name__}: {str(e)[:300]}"

print(json.dumps(result))
