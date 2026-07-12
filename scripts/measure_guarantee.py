"""Measure the FACTGATE anti-hallucination guarantee: false-VERIFIED rate on
absent + corrupted claims (must be ~0), true-VERIFIED coverage on stored facts."""
import json, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rck.knowledge_base import ShardedKnowledgeBase
from rck.bulk_ingest import bulk_load_jsonl
from rck.shard_sizing import auto_shard_for_kb
from factgate.datagen.run_all import read_triples
from factgate.gate import verify_claim
pass

def wilson(k, n, z=1.96):
    if n == 0: return [0.0, 1.0]
    p = k/n; d = 1+z*z/n
    c = p+z*z/(2*n); h = z*((p*(1-p)+z*z/(4*n))/n)**0.5
    return [max(0,(c-h)/d), min(1,(c+h)/d)]

ROOT = Path(__file__).resolve().parents[1]
kb = ShardedKnowledgeBase(dim=4096, n_shards=auto_shard_for_kb(90000), seed=0)
bulk_load_jsonl(kb, ROOT/"data/kb_train_90k.jsonl", symmetrize=False)
train = read_triples(ROOT/"data/kb_train_90k.jsonl", max_facts=None)
holdout = read_triples(ROOT/"data/holdout_unknowable_10k.jsonl", max_facts=None)
random.seed(11); N=1500; t0=time.time()

def rates(claims):
    c = {"VERIFIED":0,"CONTRADICTED":0,"OUT_OF_KB":0,"UNRESOLVED":0}
    for s,r,o in claims:
        c[verify_claim(kb,None,s,r,o).status.value]+=1
    return c

true_claims = [(t["s"],t["r"],t["o"]) for t in random.sample(train,N)]
hold_claims = [(t["s"],t["r"],t["o"]) for t in random.sample(holdout,N)]
isa=[t for t in train if t["r"]=="isa"]; objs=list({t["o"] for t in isa})
corr_claims=[]
for t in random.sample(isa,N):
    w=random.choice(objs)
    if w!=t["o"]: corr_claims.append((t["s"],"isa",w))

TR=rates(true_claims); HO=rates(hold_claims); CO=rates(corr_claims)
res = {
  "true_stored":  {"n":sum(TR.values()),"verdicts":TR,
     "true_verified_rate":TR["VERIFIED"]/sum(TR.values()),
     "ci95":wilson(TR["VERIFIED"],sum(TR.values()))},
  "holdout_absent": {"n":sum(HO.values()),"verdicts":HO,
     "false_verified_rate":HO["VERIFIED"]/sum(HO.values()),
     "ci95":wilson(HO["VERIFIED"],sum(HO.values()))},
  "corrupted_wrong_object": {"n":sum(CO.values()),"verdicts":CO,
     "false_verified_rate":CO["VERIFIED"]/sum(CO.values()),
     "contradiction_catch_rate":CO["CONTRADICTED"]/sum(CO.values()),
     "ci95_false_verified":wilson(CO["VERIFIED"],sum(CO.values()))},
  "kb_config":{"facts":90000,"shards":auto_shard_for_kb(90000),"symmetrize":False,
     "known_threshold":0.11},
  "seconds":time.time()-t0,
}
(ROOT/"results").mkdir(exist_ok=True)
json.dump(res, open(ROOT/"results/guarantee_measurement.json","w"), indent=2)
print(json.dumps({k:(v.get("false_verified_rate",v.get("true_verified_rate")))
                  for k,v in res.items() if isinstance(v,dict) and "n" in v}, indent=2))
print(f"[{time.time()-t0:.0f}s]")
