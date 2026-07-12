"""Core 10M-param DirectML forward+backward micro-benchmark. Run in its own process."""
import sys, time, json, traceback
import warnings as _w
_warn_log = []
def _warn_handler(message, category, filename, lineno, file=None, line=None):
    _warn_log.append(str(message))
_w.showwarning = _warn_handler
_w.simplefilter("always")

import torch
import torch.nn as nn

result = {"status": "unknown", "steps_s_10M": None, "param_count_10M_model": None,
          "op_fallback_warnings": [], "notes": []}

try:
    import torch_directml
    dml = torch_directml.device()
except Exception as e:
    result["status"] = "import_failed"
    result["notes"].append(f"torch_directml import failed: {e}")
    print(json.dumps(result))
    sys.exit(0)


class TinyTransformer(nn.Module):
    def __init__(self, vocab=8000, d_model=256, nhead=8, nlayers=4, dim_ff=1024, seq=512):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                            batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x):
        h = self.embed(x) + self.pos[:, : x.shape[1], :]
        h = self.enc(h)
        h = self.ln(h)
        return self.head(h)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def bench(device_obj, seq=512, batch=8, steps=15, warmup=5):
    torch.manual_seed(0)
    model = TinyTransformer(seq=seq).to(device_obj)
    nparams = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, 8000, (batch, seq), device=device_obj)
    y = torch.randint(0, 8000, (batch, seq), device=device_obj)
    lossf = nn.CrossEntropyLoss()

    def step():
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = lossf(out.view(-1, out.shape[-1]), y.view(-1))
        loss.backward()
        opt.step()
        return loss.item()

    for _ in range(warmup):
        step()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    t1 = time.perf_counter()
    return steps / (t1 - t0), nparams


try:
    dml_sps, nparams = bench(dml, seq=512, batch=8, steps=15, warmup=5)
    result["steps_s_10M"] = dml_sps
    result["param_count_10M_model"] = nparams
    result["status"] = "ok"
except Exception as e:
    result["status"] = "dml_bench_failed"
    result["notes"].append(f"DML 10M bench failed: {e}\n{traceback.format_exc()[-1500:]}")

result["op_fallback_warnings"] = list(set(_warn_log))[:30]
print(json.dumps(result))
