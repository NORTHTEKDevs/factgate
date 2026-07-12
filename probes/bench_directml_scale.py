"""0.5B-scale single-step DML test. Conservative: batch=1, single fwd+bwd step, fp32 (bf16 crashes DML).
Run in own process; keep short to avoid stressing the iGPU driver."""
import sys, time, json, traceback, os
import psutil

result = {"status": "unknown", "param_count": None, "step_time_s": None, "tok_s": None,
          "ram_before_gb": None, "ram_after_gb": None, "notes": []}

proc = psutil.Process(os.getpid())
result["ram_before_gb"] = round(psutil.virtual_memory().used / 1e9, 2)

import torch
import torch.nn as nn

try:
    import torch_directml
    dml = torch_directml.device()
except Exception as e:
    result["status"] = "import_failed"
    result["notes"].append(str(e))
    print(json.dumps(result))
    sys.exit(0)


class Model05B(nn.Module):
    def __init__(self, vocab=32000, d_model=1024, nhead=16, nlayers=24, dim_ff=4096, seq=512):
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


seq = 512
batch = 1

try:
    torch.manual_seed(0)
    model = Model05B(seq=seq).to(dml)
    nparams = count_params(model)
    result["param_count"] = nparams

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, 32000, (batch, seq), device=dml)
    y = torch.randint(0, 32000, (batch, seq), device=dml)
    lossf = nn.CrossEntropyLoss()

    t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    out = model(x)
    loss = lossf(out.view(-1, out.shape[-1]), y.view(-1))
    loss.backward()
    opt.step()
    loss_val = loss.item()  # forces sync
    t1 = time.perf_counter()

    result["step_time_s"] = t1 - t0
    result["tok_s"] = (batch * seq) / (t1 - t0)
    result["loss_value"] = loss_val
    result["status"] = "ok"
    result["ram_after_gb"] = round(psutil.virtual_memory().used / 1e9, 2)
except Exception as e:
    result["status"] = "scale_bench_failed"
    result["notes"].append(f"{e}\n{traceback.format_exc()[-1500:]}")
    result["ram_after_gb"] = round(psutil.virtual_memory().used / 1e9, 2)

print(json.dumps(result))
