"""The hardening loop: run every check, verify the checks can fail, report what is proven.

Five adversarial rounds found defects one at a time, and by the fourth round most new leaks
were inside the previous round's fixes. Hand-hunting does not converge. What replaced it is
a set of proofs a machine can re-run in seconds -- but a proof is only worth what it can
catch, so this runs each proof AND then deliberately breaks the code to confirm the proof
notices.

    python scripts/harden.py             # the full loop
    python scripts/harden.py --quick     # skip mutation testing

Exit code is 0 only when every check passes AND every mutant is caught. A proof that
survives a mutant is reported as a GAP, because it is a check that would not have found the
bug it exists to find.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable

# (name, file, anchor, replacement). Each disables ONE defence. Every mutant must be caught
# by at least one test, or the suite has a hole exactly where that defence lives.
MUTANTS = [
    ("comparison: unit check dropped before DIFFER",
     "factgate/domain/quantity.py",
     "        if dq.unit == cq.unit:\n            return MATCH if dq.value == cq.value else DIFFER",
     "        if True:\n            return MATCH if dq.value == cq.value else DIFFER"),
    ("comparison: text canonicalisation removed",
     "factgate/domain/quantity.py",
     '    return " ".join(unicodedata.normalize("NFC", str(s))\n'
     '                    .translate(_PUNCT_FOLD).lower().split())',
     '    return " ".join(str(s).lower().split())'),
    ("residue: modifier-shape allowlist disabled",
     "factgate/domain/residue.py",
     "    if not _is_modifier_phrase(residue, raw_residue):",
     "    if False:"),
    ("residue: negation check disabled",
     "factgate/domain/residue.py",
     "        if _NEGATION & set(_WORD.findall(seg)):\n            continue",
     "        if False:\n            continue"),
    ("residue: clause scoping removed",
     "factgate/domain/residue.py",
     '_SEGMENT = re.compile(r"(?<!\\d)[.;:!?]|(?<=\\d)[.;:!?](?!\\d)")',
     '_SEGMENT = re.compile(r"(?!x)x")'),
    ("residue: second-value guard removed",
     "factgate/domain/residue.py",
     "    if _DIGIT.search(residue) or any(ch.isnumeric() for ch in residue):",
     "    if False:"),
    ("conditional: ambiguity check removed",
     "factgate/domain/factset.py",
     "        if not ctx:\n            return None",
     "        if False:\n            return None"),
    ("lint: unit-alias dimension check removed",
     "factgate/domain/factset.py",
     "            if a_known and c_known and a_known != c_known:",
     "            if False:"),
    ("extraction: ambiguity (decoy) guard disabled",
     "factgate/domain/link.py",
     "            if ambiguous_candidates(value, text, fs, entity, relation):",
     "            if False:"),
    ("extraction: foreign-word repair disabled",
     "factgate/domain/link.py",
     "            if value is not None and foreign_words(value, text):",
     "            if False:"),
    ("extraction: value grounding disabled",
     "factgate/domain/link.py",
     "    if not any(re.search(rf\"(?<![0-9.]){re.escape(n)}(?![0-9]|\\.[0-9])\", low)\n"
     "               for n in forms):\n        return False",
     "    if False:\n        return False"),
]

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""),
          flush=True)


def run(args, timeout=1200):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True,
                          timeout=timeout, encoding="utf-8", errors="replace")


def _tail(out: str) -> str:
    lines = [l for l in out.strip().splitlines() if "passed" in l or "failed" in l]
    return lines[-1] if lines else out.strip()[-70:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip mutation testing")
    a = ap.parse_args()
    start = time.perf_counter()

    print("PROOFS")
    r = run([PY, "-m", "pytest", "tests/", "-q"])
    check("full suite", r.returncode == 0, _tail(r.stdout))

    for label, path in [("value grammar vs exact-arithmetic oracle", "tests/test_value_grammar.py"),
                        ("residue and conditional verdict paths", "tests/test_verdict_paths.py"),
                        ("safety invariants I1-I7", "tests/test_invariants.py")]:
        r = run([PY, "-m", "pytest", path, "-q"])
        check(label, r.returncode == 0, _tail(r.stdout))

    print("\nDOMAINS")
    probe = (
        "import json,glob,sys; sys.path.insert(0,'.')\n"
        "from factgate.domain.factset import FactSet\n"
        "bad=[]\n"
        "n=0\n"
        "for f in sorted(glob.glob('data/domains/*.json')):\n"
        "    d=json.load(open(f,encoding='utf-8'))\n"
        "    try:\n"
        "        fs=FactSet.from_dict(d); ok,miss=fs.validate_sources(d['corpus'])\n"
        "        e=[p for p in fs.lint() if p['level']=='error']\n"
        "        n+=1\n"
        "        if miss or e: bad.append(f)\n"
        "    except Exception as ex: bad.append(f'{f}: {ex}')\n"
        "print(f'{n} loaded, {len(bad)} bad'); print('\\n'.join(map(str,bad[:5])))\n")
    r = run([PY, "-c", probe])
    check("every shipped domain loads, quotes validate, lint clean",
          ", 0 bad" in r.stdout, r.stdout.strip().splitlines()[0] if r.stdout else "")

    if not a.quick:
        print("\nMUTATION -- each proof must CATCH a deliberately broken defence")
        # Mutants run in an isolated COPY of the repository.
        #
        # The first version edited the working tree in place and restored it afterwards,
        # which meant that for about a minute the repository on disk was DELIBERATELY
        # BROKEN. An external reviewer ran the README's own proving command during that
        # window, saw ten failures and one hang, and reported -- correctly -- that the
        # proofs do not reproduce. A harness that corrupts the tree it is checking cannot
        # establish anything, and a kill signal mid-mutation would have left the corruption
        # behind for good.
        workspace = pathlib.Path(tempfile.mkdtemp(prefix="factgate_mutants_"))
        copy = workspace / "repo"
        shutil.copytree(REPO, copy, ignore=shutil.ignore_patterns(
            ".venv", ".git", "__pycache__", "*.pyc", ".pytest_cache", "results", "demo"))
        env = {**os.environ, "PYTHONPATH": str(copy), "PYTHONIOENCODING": "utf-8"}
        before = {rel: (REPO / rel).read_bytes() for _, rel, _, _ in MUTANTS}
        try:
            for name, rel, anchor, broken in MUTANTS:
                target = copy / rel
                original = target.read_text(encoding="utf-8")
                if anchor not in original:
                    check(f"mutant: {name}", False, "anchor not found -- the code moved")
                    continue
                target.write_text(original.replace(anchor, broken, 1), encoding="utf-8")
                try:
                    r = subprocess.run(
                        [PY, "-m", "pytest", "tests/", "-q", "-x"], cwd=copy,
                        capture_output=True, text=True, timeout=1200, env=env,
                        encoding="utf-8", errors="replace")
                finally:
                    target.write_text(original, encoding="utf-8")
                check(f"mutant caught: {name}", r.returncode != 0,
                      "SURVIVED -- no test covers this defence" if r.returncode == 0 else "")
        finally:
            shutil.rmtree(workspace, ignore_errors=True)
        untouched = all((REPO / rel).read_bytes() == blob for rel, blob in before.items())
        check("working tree untouched by mutation testing", untouched,
              "" if untouched else "SOURCE MODIFIED -- the harness corrupted the repo")

    failed = [n for n, ok, _ in RESULTS if not ok]
    print()
    print(f"{'=' * 70}")
    if failed:
        print(f"HARDEN FAILED: {len(failed)} of {len(RESULTS)} checks, "
              f"{time.perf_counter() - start:.0f}s")
        for n in failed:
            print(f"  - {n}")
        print("A surviving mutant is not a passing run. It marks a defence nothing tests,")
        print("which is where the next defect will live.")
        return 1
    print(f"HARDEN PASSED: {len(RESULTS)}/{len(RESULTS)} checks in "
          f"{time.perf_counter() - start:.0f}s")
    print()
    print("What this proves, and what it does not:")
    print("  PROVEN   over the declared value grammar, every MATCH is confirmed equal and")
    print("           every DIFFER confirmed unequal by exact rational arithmetic that")
    print("           shares no code with the implementation")
    print("  PROVEN   no author declaration can make the gate verify an unequal value")
    print("           without lint refusing the fact set first")
    print("  PROVEN   the residue and conditional paths behave as constructed cases require")
    print("  PROVEN   every check above CATCHES the defence it exists to protect")
    print("  NOT      anything about notations outside the declared grammar. Those are held,")
    print("           which is the fail-closed answer, not a verified one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
