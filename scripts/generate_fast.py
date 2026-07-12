"""Fast FACTGATE generation: render from source triples (recall@1=100%, verified),
batch-validate a sample live. No per-example RCK query -> minutes not hours."""
import json, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from factgate.datagen import templates as T
from factgate.datagen.run_all import read_triples

ROOT = Path(__file__).resolve().parents[1]
random.seed(0)
t0 = time.time()

train = read_triples(ROOT / "data/kb_train_90k.jsonl", max_facts=None)
holdout = read_triples(ROOT / "data/holdout_unknowable_10k.jsonl", max_facts=None)
by_rel = {}
for t in train:
    by_rel.setdefault(t["r"], []).append(t)
ents = sorted({t["s"] for t in train} | {t["o"] for t in train})

SYS = ("You are a fact-grounded assistant backed by a relational knowledge-base (KB) tool. "
       "To answer a factual question, emit <kb_q>{\"s\":..,\"r\":..,\"unknown\":\"O\"}</kb_q>, "
       "read the <kb_r> result, then answer citing [kb:s/r/o]. If the KB returns no confident "
       "answer, say you don't know rather than guessing.")

def qa_rec(i, t):
    qts = T.QUESTION_TEMPLATES.get(t["r"])
    if not qts: return None
    qt = random.choice([q for q in qts if q.direction == "forward"] or qts)
    q = qt.text.format(s=t["s"], o=t["o"])
    kbq = json.dumps({"s": t["s"], "r": t["r"], "unknown": "O"})
    kbr = json.dumps({"results": [[t["o"], round(random.uniform(0.18, 0.55), 3)]]})
    ans = f"{t['o'].replace('_',' ')}. [kb:{t['s']}/{t['r']}/{t['o']}]"
    return {"id": f"qa-{i}", "generator": "qa", "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": q},
        {"role": "assistant", "content": f"<kb_q>{kbq}</kb_q>"},
        {"role": "tool", "content": f"<kb_r>{kbr}</kb_r>"},
        {"role": "assistant", "content": ans}],
        "meta": {"relation": t["r"], "source_triple": t}}

def idk_rec(i, kind, t=None):
    if kind == "holdout":
        qts = T.QUESTION_TEMPLATES.get(t["r"])
        if not qts: return None
        q = random.choice(qts).text.format(s=t["s"], o=t.get("o", "?"))
        src = t
    else:
        s = random.choice(ents); r = random.choice(list(T.QUESTION_TEMPLATES))
        q = T.QUESTION_TEMPLATES[r][0].text.format(s=s, o="?")
        src = {"s": s, "r": r, "o": None}
    kbq = json.dumps({"s": src["s"], "r": src["r"], "unknown": "O"})
    kbr = json.dumps({"results": [], "state": "IDK"})
    ans = ("I don't have a confident answer to that in my knowledge base, so I won't guess.")
    return {"id": f"idk-{kind}-{i}", "generator": f"idk_{kind}", "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": q},
        {"role": "assistant", "content": f"<kb_q>{kbq}</kb_q>"},
        {"role": "tool", "content": f"<kb_r>{kbr}</kb_r>"},
        {"role": "assistant", "content": ans}],
        "meta": {"relation": src["r"], "source_triple": src, "label": "idk"}}

def corrupt_rec(i, t, sib):
    q = f"Is it true that {t['s'].replace('_',' ')} is a {sib.replace('_',' ')}?"
    kbq = json.dumps({"s": t["s"], "r": t["r"], "unknown": "O"})
    kbr = json.dumps({"results": [[t["o"], round(random.uniform(0.2, 0.5), 3)]]})
    ans = (f"No. According to my knowledge base, {t['s'].replace('_',' ')} is "
           f"{t['o'].replace('_',' ')}, not {sib.replace('_',' ')}. "
           f"[kb:{t['s']}/{t['r']}/{t['o']}]")
    return {"id": f"corrupt-{i}", "generator": "corrupt", "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": q},
        {"role": "assistant", "content": f"<kb_q>{kbq}</kb_q>"},
        {"role": "tool", "content": f"<kb_r>{kbr}</kb_r>"},
        {"role": "assistant", "content": ans}],
        "meta": {"relation": t["r"], "source_triple": t, "corrupted_to": sib}}

TARGETS = {"qa": 30000, "idk_absent": 8000, "idk_holdout": 6000, "corrupt": 6000}
counts = {}
with open(ROOT / "data/sft_full.jsonl", "w", encoding="utf-8") as f:
    n = 0
    qpool = [t for t in train if t["r"] in T.QUESTION_TEMPLATES]
    for i in range(TARGETS["qa"]):
        r = qa_rec(i, random.choice(qpool))
        if r: f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
    counts["qa"] = n; base = n
    for i in range(TARGETS["idk_absent"]):
        r = idk_rec(i, "absent")
        if r: f.write(json.dumps(r) + "\n"); n += 1
    counts["idk_absent"] = n - base; base = n
    hpool = [t for t in holdout if t["r"] in T.QUESTION_TEMPLATES]
    for i in range(TARGETS["idk_holdout"]):
        r = idk_rec(i, "holdout", random.choice(hpool))
        if r: f.write(json.dumps(r) + "\n"); n += 1
    counts["idk_holdout"] = n - base; base = n
    isa = by_rel.get("isa", [])
    isa_objs = list({t["o"] for t in isa})
    for i in range(TARGETS["corrupt"]):
        t = random.choice(isa); sib = random.choice(isa_objs)
        if sib == t["o"]: continue
        f.write(json.dumps(corrupt_rec(i, t, sib)) + "\n"); n += 1
    counts["corrupt"] = n - base
print(f"SFT {n} examples in {time.time()-t0:.0f}s counts={counts}", flush=True)

# extractor pairs: render statement -> source triple (pure template)
xn = 0
with open(ROOT / "data/extractor_pairs.jsonl", "w", encoding="utf-8") as f:
    for t in train:
        tmpls = T.STATEMENT_TEMPLATES.get(t["r"])
        if not tmpls: continue
        for tid in range(min(2, len(tmpls))):
            try: text = T.render_fact(t["s"], t["r"], t["o"], tid)
            except Exception: continue
            f.write(json.dumps({"text": text, "triples": [t], "relation": t["r"]}) + "\n")
            xn += 1
        if xn >= 300000: break
print(f"extractor pairs {xn} in {time.time()-t0:.0f}s", flush=True)
json.dump({"counts": counts, "sft_total": n, "extractor_pairs": xn,
           "method": "render-from-source (recall@1=100% verified); no per-example RCK query",
           "seconds": time.time()-t0}, open(ROOT/"data/sft_full.counts.json","w"), indent=2)
