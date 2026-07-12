"""FACTGATE generator LoRA fine-tune on ROCm gfx1151 (bf16, no quantization)."""
import argparse, json, os, time
from pathlib import Path
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[1]

def load_sft(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit: break
            r = json.loads(line)
            rows.append({"messages": r["messages"]})
    return Dataset.from_list(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default=str(ROOT/"data/sft_full.jsonl"))
    ap.add_argument("--out", default=str(ROOT/"checkpoints/factgate-lora"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--maxlen", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    print(f"[load] {a.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    ds = load_sft(a.data, a.limit)
    print(f"[data] {len(ds)} examples ({time.time()-t0:.0f}s)", flush=True)
    cfg = SFTConfig(output_dir=a.out, num_train_epochs=a.epochs,
                    per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.accum,
                    learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
                    logging_steps=10, save_steps=200, save_total_limit=3,
                    bf16=True, max_length=a.maxlen, max_steps=a.max_steps,
                    gradient_checkpointing=True, report_to=[], dataset_num_proc=1,
                    completion_only_loss=False)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         peft_config=lora, processing_class=tok)
    trainer.model.print_trainable_parameters()
    print(f"[train] starting ({time.time()-t0:.0f}s)", flush=True)
    ckpt = None
    if a.resume:
        import glob
        cks = sorted(glob.glob(str(Path(a.out)/"checkpoint-*")), key=lambda p: int(p.split("-")[-1]))
        ckpt = cks[-1] if cks else None
        if ckpt: print(f"[resume] {ckpt}", flush=True)
    trainer.train(resume_from_checkpoint=ckpt)
    trainer.save_model(a.out); tok.save_pretrained(a.out)
    json.dump({"model": a.model, "n": len(ds), "epochs": a.epochs,
               "seconds": time.time()-t0}, open(Path(a.out)/"train_meta.json","w"), indent=2)
    print(f"[done] saved {a.out} ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
