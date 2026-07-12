"""Same 10M model on CPU (for apples-to-apples vs DML, and reused across venvs for CPU baseline)."""
import sys, time, json
import torch
import torch.nn as nn

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

torch.set_num_threads(32)
sps, nparams = bench(torch.device("cpu"), seq=512, batch=8, steps=8, warmup=2)
print(json.dumps({"steps_s_10M_cpu": sps, "param_count": nparams, "threads": torch.get_num_threads(), "torch_version": torch.__version__}))
