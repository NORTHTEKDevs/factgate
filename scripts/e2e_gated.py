"""B4 serving-gateway wiring check: drive the serving gateway's VerifiedBackend on 3
diagnostic prompts using a SCRIPTED inner (see ScriptedInner below) plus the real RCK gate.

The inner is deliberately NOT an LLM: it returns hand-written strings that already contain
the `[kb:s/r/o]` tag, so the run is deterministic and fast. This proves the gate plumbing
is correctly wired end to end; it does NOT demonstrate a live model routing its own claims
through the gate, and it does not exercise claim extraction at all.

Requires a separate OpenAI-compatible serving gateway checked out alongside this
repo (not included here) that exposes `serving.factgate_backend.VerifiedBackend`.
Point SERVING_GATEWAY_PATH at that checkout before running.
"""
import json, sys, os
from pathlib import Path
FG = Path(__file__).resolve().parents[1]
SERVING_GATEWAY = Path(os.environ.get(
    "SERVING_GATEWAY_PATH", str(FG.parent / "serving-gateway")))
sys.path.insert(0, str(SERVING_GATEWAY)); sys.path.insert(0, str(FG))
os.environ["SERVING_GATEWAY_FACTGATE_MODE"] = "inproc"
os.environ["SERVING_GATEWAY_FACTGATE_INNER"] = "ollama"
os.environ["SERVING_GATEWAY_FACTGATE_REPO"] = str(FG)
os.environ.setdefault("OLLAMA_MODEL", "llama3.2:3b")

# point the inproc gate at the corrected eval-90k snapshot
import factgate.kb_service as ks
_orig = ks.load_kb
ks.get_kb = lambda **kw: _orig(FG/"data/kb_train_90k.jsonl", FG/"kb_snapshot_eval90k")

from serving.factgate_backend import VerifiedBackend
from serving.backends import Backend, Completion, Message

# use a scripted inner so the E2E is deterministic + fast (no 40s ollama tool loop);
# the inner emits a factual claim tag; the GATE (real RCK) decides its fate.
class ScriptedInner(Backend):
    name = "scripted"; default_model = "scripted"
    def __init__(self, text): self._t = text
    def complete(self, messages, model=None, **kw):
        return Completion(id="e2e", created=0, model="scripted",
                          message=Message(role="assistant", content=self._t),
                          finish_reason="stop")
    def list_models(self): return ["scripted"]

cases = [
    ("known fact (dog isa)", "A dog is an animal. [kb:dog/isa/animal]"),
    ("false premise", "The capital of france is berlin. [kb:france/capital/berlin]"),
    ("out-of-KB entity", "A glorptron is a kind of zibbler. [kb:glorptron/isa/zibbler]"),
]
out = []
for label, text in cases:
    vb = VerifiedBackend(inner=ScriptedInner(text))
    comp = vb.complete([{"role":"user","content":"?"}], model="verified")
    out.append({"case": label, "inner_said": text, "gated_output": comp.message.content})
    print(f"\n=== {label} ===\nINNER: {text}\nGATED: {comp.message.content}")
json.dump(out, open(FG/"results/e2e_gated_demo.json","w"), indent=2)
print("\n[e2e demo written]")
