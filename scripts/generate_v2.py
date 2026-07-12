"""FACTGATE gen v2: cite what RCK ACTUALLY returns (kb.answer ~2ms), route
low-confidence to IDK. Teaches the model to copy RCK output, not source truth."""
import json, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from factgate.datagen import templates as T
from factgate.datagen.run_all import read_triples
from factgate.kb_service import load_kb

ROOT = Path(__file__).resolve().parents[1]
random.seed(0); t0 = time.time()
agent = load_kb(ROOT/"data/kb_train_90k.jsonl", ROOT/"kb_snapshot_eval90k")
kb = agent.knowledge
train = read_triples(ROOT/"data/kb_train_90k.jsonl", max_facts=None)
holdout = read_triples(ROOT/"data/holdout_unknowable_10k.jsonl", max_facts=None)
ents = sorted({t["s"] for t in train} | {t["o"] for t in train})
KNOWN_T = 0.11  # calibrated: TRUE-verify up, leak stays 0 (measured 2026-07-12)

SYS = ("You are a fact-grounded assistant backed by a relational knowledge-base (KB) tool. "
       "To answer a factual question, emit <kb_q>{\"s\":..,\"r\":..,\"unknown\":\"O\"}</kb_q>, "
       "read the <kb_r> result, then answer citing [kb:s/r/o]. If the KB returns no confident "
       "answer, say you don't know rather than guessing.")

def trace(uid, gen, q, s, r, results, ans, meta):
    kbq = json.dumps({"s": s, "r": r, "unknown": "O"})
    kbr = json.dumps({"results": results})
    return {"id": uid, "generator": gen, "messages": [
        {"role":"system","content":SYS},{"role":"user","content":q},
        {"role":"assistant","content":f"<kb_q>{kbq}</kb_q>"},
        {"role":"tool","content":f"<kb_r>{kbr}</kb_r>"},
        {"role":"assistant","content":ans}], "meta": meta}

counts = {"qa_known":0,"qa_idk":0,"idk_absent":0,"idk_holdout":0,"corrupt":0}
NQA, NIA, NIH, NC = 34000, 8000, 6000, 6000
qpool = [t for t in train if t["r"] in T.QUESTION_TEMPLATES]
with open(ROOT/"data/sft_full.jsonl","w",encoding="utf-8") as f:
    # QA: query live, cite real answer; low-confidence -> honest IDK
    for i in range(NQA):
        t = random.choice(qpool)
        qt = random.choice([q for q in T.QUESTION_TEMPLATES[t["r"]] if q.direction=="forward"] or T.QUESTION_TEMPLATES[t["r"]])
        q = qt.text.format(s=t["s"], o=t["o"])
        cand = kb.query({"S":t["s"],"R":t["r"]}, "O", top_k=3)
        results = [[sym, round(float(sc),3)] for sym,sc in cand]
        if cand and cand[0][1] >= KNOWN_T:
            o = cand[0][0]
            ans = f"{str(o).replace('_',' ')}. [kb:{t['s']}/{t['r']}/{o}]"
            rec = trace(f"qa-{i}", "qa", q, t["s"], t["r"], results, ans,
                        {"relation": t["r"], "kb_answer": o, "source_triple": t})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts["qa_known"]+=1
        else:
            ans = "I don't have a confident answer to that in my knowledge base, so I won't guess."
            f.write(json.dumps(trace(f"qa-{i}","qa_idk",q,t["s"],t["r"],results,ans,
                    {"relation":t["r"],"label":"idk","source_triple":t}),ensure_ascii=False)+"\n")
            counts["qa_idk"]+=1
    # absent-probe IDK (verified IDK via empty/low results)
    for i in range(NIA):
        s = random.choice(ents); r = random.choice(list(T.QUESTION_TEMPLATES))
        cand = kb.query({"S":s,"R":r}, "O", top_k=3)
        if cand and cand[0][1] >= KNOWN_T: continue  # actually known, skip
        q = T.QUESTION_TEMPLATES[r][0].text.format(s=s, o="?")
        ans = "I don't have a confident answer to that in my knowledge base, so I won't guess."
        f.write(json.dumps(trace(f"idk-a-{i}","idk_absent",q,s,r,
                [[str(x),round(float(y),3)] for x,y in cand],ans,{"relation":r,"label":"idk"}))+"\n")
        counts["idk_absent"]+=1
    # holdout IDK (facts absent from KB by construction)
    hpool = [t for t in holdout if t["r"] in T.QUESTION_TEMPLATES]
    for i in range(NIH):
        t = random.choice(hpool)
        q = random.choice(T.QUESTION_TEMPLATES[t["r"]]).text.format(s=t["s"], o=t.get("o","?"))
        cand = kb.query({"S":t["s"],"R":t["r"]}, "O", top_k=3)
        ans = "I don't have a confident answer to that in my knowledge base, so I won't guess."
        f.write(json.dumps(trace(f"idk-h-{i}","idk_holdout",q,t["s"],t["r"],
                [[str(x),round(float(y),3)] for x,y in cand],ans,{"relation":t["r"],"label":"idk"}))+"\n")
        counts["idk_holdout"]+=1
    # corrupt/contradiction on isa
    isa = [t for t in train if t["r"]=="isa"]; objs = list({t["o"] for t in isa})
    for i in range(NC):
        t = random.choice(isa); sib = random.choice(objs)
        if sib==t["o"]: continue
        cand = kb.query({"S":t["s"],"R":"isa"}, "O", top_k=3)
        if not (cand and cand[0][1] >= KNOWN_T): continue
        o = cand[0][0]
        q = f"Is it true that {t['s'].replace('_',' ')} is a {sib.replace('_',' ')}?"
        ans = (f"No. According to my knowledge base, {t['s'].replace('_',' ')} is "
               f"{str(o).replace('_',' ')}, not {sib.replace('_',' ')}. [kb:{t['s']}/isa/{o}]")
        f.write(json.dumps(trace(f"corr-{i}","corrupt",q,t["s"],"isa",
                [[str(x),round(float(y),3)] for x,y in cand],ans,
                {"relation":"isa","kb_answer":o,"corrupted_to":sib}))+"\n")
        counts["corrupt"]+=1
tot = sum(counts.values())
print(f"SFT {tot} in {time.time()-t0:.0f}s {counts}", flush=True)
json.dump({"counts":counts,"sft_total":tot,"known_threshold":KNOWN_T,
           "method":"live kb.answer, cite real RCK output; low-conf->IDK",
           "seconds":time.time()-t0}, open(ROOT/"data/sft_full.counts.json","w"), indent=2)
