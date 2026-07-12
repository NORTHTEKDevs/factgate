"""Overnight autonomous watchdog for the 14B FACTGATE LoRA run.
Does nothing if training is healthy; auto-resumes on crash/stall; runs the
merge+gate evaluation when training finishes; writes a final report. Survives
detached across the whole ~20h run.
ACCEPTANCE: checkpoints/factgate-14b/adapter_model.safetensors exists AND
results/checkpoint_eval.json written with gate_leaked_verified == 0."""
import os, subprocess, sys, time, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT/".venv-rocm/Scripts/python.exe")
OUT = ROOT/"checkpoints/factgate-14b"
LOG = ROOT/"data/logs_train_14b.txt"
STATUS = ROOT/"data/overnight_status.json"
FINAL = ROOT/"FINAL-REPORT.md"
DONE_MARK = "[done] saved"
STALL_S = 900          # 15 min with no log growth = crashed/hung
CHECK_S = 300
MAX_RESUMES = 30

TRAIN = [PY, "scripts/train_lora.py", "--model", "Qwen/Qwen2.5-14B-Instruct",
         "--out", str(OUT), "--epochs", "1", "--bs", "2", "--accum", "8",
         "--maxlen", "1024", "--resume"]
ENV = {**os.environ, "HF_HUB_DISABLE_TELEMETRY": "1"}  # NO experimental flash-attn (native crash)

def log_done():
    return LOG.exists() and DONE_MARK in LOG.read_text(encoding="utf-8", errors="ignore")

def log_stale():
    if not LOG.exists(): return True
    return (time.time() - LOG.stat().st_mtime) > STALL_S

def write_status(**kw):
    kw["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATUS.write_text(json.dumps(kw, indent=2))

def resume():
    fh = open(LOG, "a", encoding="utf-8")
    fh.write(f"\n[watchdog] resume {time.strftime('%H:%M:%S')}\n"); fh.flush()
    subprocess.Popen(TRAIN, cwd=str(ROOT), env=ENV, stdout=fh, stderr=subprocess.STDOUT)

resumes = 0
write_status(phase="watching", resumes=0)
# wait out an initial grace so we don't fight the already-running launch
time.sleep(CHECK_S)
while True:
    if log_done() or (OUT/"adapter_model.safetensors").exists():
        write_status(phase="training_done"); break
    if log_stale():
        if resumes >= MAX_RESUMES:
            write_status(phase="STALLED", resumes=resumes); break
        resumes += 1
        write_status(phase="resuming", resumes=resumes)
        resume()
        time.sleep(600)   # let model reload (~8min) before re-checking
    else:
        write_status(phase="training", resumes=resumes,
                     last_log_age_s=int(time.time()-LOG.stat().st_mtime))
    time.sleep(CHECK_S)

# --- evaluation ---
if (OUT/"adapter_model.safetensors").exists():
    write_status(phase="evaluating")
    ev = subprocess.run([PY, "scripts/eval_checkpoint.py", "--ckpt", str(OUT), "--n", "60"],
                        cwd=str(ROOT), env=ENV, capture_output=True, text=True, timeout=7200)
    (ROOT/"data/logs_eval_final.txt").write_text(ev.stdout + "\n---STDERR---\n" + ev.stderr)
    res = {}
    ep = ROOT/"results/checkpoint_eval.json"
    if ep.exists(): res = json.loads(ep.read_text())
    tm = {}
    tmp = OUT/"train_meta.json"
    if tmp.exists(): tm = json.loads(tmp.read_text())
    FINAL.write_text(f"""# FACTGATE 14B - Overnight Training Final Report

Training complete: Qwen2.5-14B-Instruct LoRA on {tm.get('n','?')} FACTGATE examples,
{tm.get('seconds',0)/3600:.1f}h wall-clock on gfx1151 (native Windows ROCm bf16).
Adapter: `checkpoints/factgate-14b/adapter_model.safetensors`. Resumes used: {resumes}.

## Trained-model behavior on the unknowable split (should abstain, never leak)
- examples: {res.get('n','?')}
- abstention_rate: {res.get('abstention_rate','?')}
- claims asserted: {res.get('claims_asserted','?')}
- gate blocked (would-be hallucinations caught): {res.get('gate_blocked','?')}
- **gate leaked (false-VERIFIED): {res.get('gate_leaked_verified','?')}**  (target 0)

The trained model is the fluent generator; the RCK gate is the hard guarantee.
Serve via Hyperion VerifiedBackend (HYPERION_BACKEND=verified). See RESULTS.md.
""")
    write_status(phase="DONE", resumes=resumes, eval=res)
else:
    write_status(phase="NO_ADAPTER_ended", resumes=resumes)
print("watchdog finished:", STATUS.read_text())
