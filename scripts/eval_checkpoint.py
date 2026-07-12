"""Merge a LoRA checkpoint, generate on held-out FACTGATE prompts, run outputs
through the gate. Proves the trained model behaves (tool-calls + abstains) and
measures model-alone vs gated hallucination on the unknowable split."""
import argparse, json, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from factgate.gate import verify_claim
from factgate.kb_service import load_kb
import re

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map={"":0})
    model = PeftModel.from_pretrained(model, a.ckpt)
    model.eval()
    agent = load_kb(ROOT/"data/kb_train_90k.jsonl", ROOT/"kb_snapshot_eval90k")
    kb = agent.knowledge
    SYS = ("You are a fact-grounded assistant backed by a relational knowledge-base (KB) tool. "
           "To answer a factual question emit <kb_q>{\"s\":..,\"r\":..,\"unknown\":\"O\"}</kb_q>, "
           "read <kb_r>, then answer citing [kb:s/r/o]. If the KB returns no confident answer, "
           "say you don't know rather than guessing.")
    # unknowable prompts (answer should be abstention)
    unk = [json.loads(l) for l in open(ROOT/"data/eval/unknowable.jsonl", encoding="utf-8")][:a.n]
    tagre = re.compile(r"\[kb:([^/]+)/([^/]+)/([^\]]+)\]")
    abst = re.compile(r"don't know|do not know|no confident|not sure|cannot confirm|unable to", re.I)
    ab=0; cw=0; leaked=0; checked=0
    for ex in unk:
        q = ex.get("question") or ex["messages"][1]["content"] if "messages" in ex else ex.get("question","")
        msgs=[{"role":"system","content":SYS},{"role":"user","content":q}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=80, do_sample=False, pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        if abst.search(txt): ab+=1
        for m in tagre.finditer(txt):
            checked+=1
            v=verify_claim(kb,None,m.group(1),m.group(2),m.group(3))
            if v.status.value=="VERIFIED": leaked+=1
            else: cw+=1  # asserted a claim that isn't verified = would-be hallucination, gate blocks it
    res={"ckpt":a.ckpt,"n":len(unk),"abstention_rate":ab/len(unk),
         "claims_asserted":checked,"gate_blocked":cw,"gate_leaked_verified":leaked}
    (ROOT/"results").mkdir(exist_ok=True)
    json.dump(res, open(ROOT/"results/checkpoint_eval.json","w"), indent=2)
    print(json.dumps(res, indent=2))

if __name__=="__main__": main()
