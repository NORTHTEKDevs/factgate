import sys, time
def p(m): print(m, flush=True)
import torch
p(f"[1] torch {torch.__version__}")
p(f"[2] cuda(ROCm) available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    p("GPU NOT visible to torch -- stopping"); sys.exit(2)
p(f"[3] device: {torch.cuda.get_device_name(0)}")
p(f"[4] capability/props: {torch.cuda.get_device_properties(0)}")
d = torch.device("cuda")
t0=time.time(); x = torch.ones(1024, device=d); torch.cuda.synchronize()
p(f"[5] tiny tensor ok ({(time.time()-t0)*1000:.0f}ms) sum={x.sum().item()}")
t0=time.time(); a=torch.randn(512,512,device=d); b=torch.randn(512,512,device=d)
c=a@b; torch.cuda.synchronize()
p(f"[6] matmul 512x512 ok ({(time.time()-t0)*1000:.0f}ms) mean={c.mean().item():.4f}")
t0=time.time()
w=torch.randn(256,256,device=d,requires_grad=True)
y=(w@torch.randn(256,256,device=d)).sum(); y.backward(); torch.cuda.synchronize()
p(f"[7] backward ok ({(time.time()-t0)*1000:.0f}ms) grad_norm={w.grad.norm().item():.2f}")
# bf16 test (DirectML crashed here; ROCm should be fine)
t0=time.time(); ab=torch.randn(512,512,device=d,dtype=torch.bfloat16)
cb=ab@ab; torch.cuda.synchronize()
p(f"[8] bf16 matmul ok ({(time.time()-t0)*1000:.0f}ms)")
# throughput probe: bigger matmul
t0=time.time()
for _ in range(20):
    m=torch.randn(2048,2048,device=d,dtype=torch.bfloat16); (m@m)
torch.cuda.synchronize()
gflop = 20*2*2048**3/1e9
p(f"[9] 20x2048 bf16 matmul: {(time.time()-t0):.2f}s = {gflop/(time.time()-t0):.0f} GFLOP/s")
p("ALL GPU STAGES PASSED")
