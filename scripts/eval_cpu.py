"""Stable CPU verification of the trained FACTGATE adapter (no GPU -> no crash).
Loads base+LoRA on CPU, generates on a few held-out prompts, gates the outputs."""
import json, sys, re, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from factgate.gate import verify_claim
from factgate.kb_service import load_kb

BASE="Qwen/Qwen2.5-14B-Instruct"; CKPT=sys.argv[sys.argv.index("--ckpt")+1] if "--ckpt" in sys.argv else "checkpoints/factgate-14b/checkpoint-200"
N=int(sys.argv[sys.argv.index("--n")+1]) if "--n" in sys.argv else 8
t0=time.time(); print("[load base on CPU]", flush=True)
tok=AutoTokenizer.from_pretrained(BASE)
model=AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cpu")
model=PeftModel.from_pretrained(model, CKPT); model.eval()
print(f"[loaded {time.time()-t0:.0f}s] gating KB...", flush=True)
agent=load_kb(ROOT/"data/kb_train_90k.jsonl", ROOT/"kb_snapshot_eval90k"); kb=agent.knowledge
SYS=("You are a fact-grounded assistant backed by a relational knowledge-base (KB) tool. "
     "To answer a factual question emit <kb_q>{\"s\":..,\"r\":..,\"unknown\":\"O\"}</kb_q>, read <kb_r>, "
     "then answer citing [kb:s/r/o]. If the KB returns no confident answer, say you don't know rather than guessing.")
unk=[json.loads(l) for l in open(ROOT/"data/eval/unknowable.jsonl",encoding="utf-8")][:N]
tagre=re.compile(r"\[kb:([^/\]]+)/([^/\]]+)/([^\]]+)\]"); qre=re.compile(r"<kb_q>")
abre=re.compile(r"don't know|do not know|no confident|not sure|cannot confirm|unable|no.*answer",re.I)
samples=[]; ab=0; toolcall=0; leaked=0; asserted=0
for i,ex in enumerate(unk):
    q=ex.get("question") or (ex["messages"][1]["content"] if "messages" in ex else "")
    ids=tok.apply_chat_template([{"role":"system","content":SYS},{"role":"user","content":q}],add_generation_prompt=True,return_tensors="pt")
    with torch.no_grad():
        out=model.generate(ids,max_new_tokens=60,do_sample=False,pad_token_id=tok.eos_token_id)
    txt=tok.decode(out[0][ids.shape[1]:],skip_special_tokens=True)
    if qre.search(txt): toolcall+=1
    if abre.search(txt): ab+=1
    for m in tagre.finditer(txt):
        asserted+=1
        if verify_claim(kb,None,m.group(1),m.group(2),m.group(3)).status.value=="VERIFIED": leaked+=1
    samples.append({"q":q,"out":txt[:200]})
    print(f"[{i+1}/{N}] {time.time()-t0:.0f}s tool={bool(qre.search(txt))} abstain={bool(abre.search(txt))}", flush=True)
res={"ckpt":CKPT,"n":len(unk),"tool_call_rate":toolcall/len(unk),"abstention_rate":ab/len(unk),
     "claims_asserted":asserted,"gate_leaked_verified":leaked,"seconds":time.time()-t0,"samples":samples[:4]}
(ROOT/"results").mkdir(exist_ok=True); json.dump(res,open(ROOT/"results/checkpoint_eval.json","w"),indent=2)
print(json.dumps({k:v for k,v in res.items() if k!="samples"},indent=2))
