"""Clean-clone acceptance: can a member of the public install and use this?

Every measurement in this project ran inside a venv that already had everything, from a
working tree, with the maintainer's PYTHONPATH. That proves nothing about a stranger who
clones the repo. This builds a FRESH virtual environment, installs the package the way the
README says to, and exercises the documented surface from a directory outside the repo.

The first version of this script installed from the WORKING TREE, which was still wrong:
the working tree contains gitignored private domains and results that no downloader ever
receives, so a passing run said nothing about whether the published tree is complete. It
now exports `git archive HEAD` -- byte-for-byte what a clone gets -- and tests that. If a
file is needed but untracked, this is the check that catches it.

    python scripts/acceptance.py            # full run, builds a throwaway venv
    python scripts/acceptance.py --keep     # leave the venv and export for inspection
    python scripts/acceptance.py --worktree # test the working tree instead of HEAD

Exit code is 0 only if every check passes. Anything else is a public-facing defect.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""),
          flush=True)


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 900):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--worktree", action="store_true",
                    help="test the working tree instead of the published HEAD")
    a = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="factgate_acceptance_"))
    venv = tmp / "venv"
    py = venv / ("Scripts" if sys.platform == "win32" else "bin") / "python.exe"
    if sys.platform != "win32":
        py = venv / "bin" / "python"

    print(f"clean-clone acceptance in {tmp}")
    try:
        # SRC is what the public actually receives. Exporting HEAD rather than copying the
        # working tree is the whole point: gitignored private domains and results are
        # absent here, exactly as they are for a downloader.
        src = REPO
        if not a.worktree:
            src = tmp / "src"
            src.mkdir()
            tarball = tmp / "head.tar"
            r = run(["git", "archive", "--format=tar", "-o", str(tarball), "HEAD"],
                    cwd=REPO)
            if r.returncode == 0:
                r = run(["tar", "-xf", str(tarball), "-C", str(src)])
            n = len(list(src.rglob("*"))) if src.exists() else 0
            check("git archive HEAD exports a tree", r.returncode == 0 and n > 20,
                  f"{n} paths" if r.returncode == 0 else r.stderr.strip()[:160])
            if r.returncode != 0:
                return 1

            # Private validation corpora are embedded verbatim in their domain files and
            # must never reach the published tree. Checked here, not just in .gitignore,
            # because a `git add -f` would defeat the ignore rule silently.
            leaked = [str(p.relative_to(src)) for p in src.rglob("*")
                      if p.is_file() and (p.name.startswith("real_")
                                          or "private" in p.parts)]
            check("no private corpus in the published tree", not leaked,
                  ", ".join(leaked[:4]) or "none")

        r = run([sys.executable, "-m", "venv", str(venv)])
        check("fresh venv builds", r.returncode == 0, r.stderr.strip()[-160:])
        if r.returncode != 0:
            return 1

        # The README's install line, verbatim in spirit: install the package itself.
        r = run([str(py), "-m", "pip", "install", "-q", "-e", str(src)])
        check("pip install -e . succeeds", r.returncode == 0,
              r.stderr.strip().splitlines()[-1][:160] if r.returncode else "")

        # Import from OUTSIDE the repo, so nothing is picked up from the working tree.
        probe = (
            "import json\n"
            "from factgate.domain.factset import FactSet\n"
            "from factgate.domain.gate import gate_claim\n"
            "from factgate.domain.suggest import suggest_qualifiers\n"
            "from factgate.domain.quantity import compare_values\n"
            "fs = FactSet.from_dict({'domain':'d','entities':{'x':['ex']},"
            "'relations':{'p':{'kind':'quantity'}},"
            "'facts':[{'s':'x','r':'p','o':'$199','source':'Costs $199.'}]})\n"
            "assert gate_claim(fs,'ex','p','$199').status == 'VERIFIED'\n"
            "assert gate_claim(fs,'ex','p','$398').status == 'BLOCK'\n"
            "assert gate_claim(fs,'ex','p','Not provided').status == 'HELD'\n"
            "assert gate_claim(fs,'nope','p','$199').status == 'HELD'\n"
            "assert fs.validate_sources('Costs $199.')[1] == []\n"
            "assert fs.lint() == []\n"
            "assert len(fs.fingerprint) == 16\n"
            "print('OK')\n"
        )
        r = run([str(py), "-c", probe], cwd=tmp)
        check("documented API works from outside the repo", "OK" in r.stdout,
              (r.stderr.strip().splitlines() or [""])[-1][:200])

        # No hidden dependency on the optional KB engine for the domain gate.
        r = run([str(py), "-c",
                 "import sys, factgate.domain.gate, factgate.domain.link, "
                 "factgate.domain.suggest\n"
                 "bad=[m for m in sys.modules if m.startswith('rck')]\n"
                 "print('RCK' if bad else 'CLEAN')"], cwd=tmp)
        check("domain gate needs no knowledge-base engine", "CLEAN" in r.stdout,
              r.stdout.strip()[:120])

        # pytest before any check that uses it -- the first version of this script
        # installed it afterwards and reported a false failure.
        run([str(py), "-m", "pip", "install", "-q", "pytest"])

        # The README's own quickstart, executed.
        r = run([str(py), "-m", "pytest", str(src / "tests" / "test_readme_example.py"),
                 "-q"], cwd=tmp)
        check("README quickstart test passes in the fresh env", r.returncode == 0,
              (r.stdout.strip().splitlines() or [""])[-1][:160])

        # Shipped demo domains load and are lint-clean for a stranger.
        loaded = []
        for f in sorted((src / "data" / "domains").glob("*.json")):
            if f.name.startswith("real_"):
                continue                      # private, not vendored
            probe2 = (f"import json\nfrom factgate.domain.factset import FactSet\n"
                      f"d=json.load(open(r'{f}',encoding='utf-8'))\n"
                      f"fs=FactSet.from_dict(d)\n"
                      f"ok,miss=fs.validate_sources(d['corpus'])\n"
                      f"e=[x for x in fs.lint() if x['level']=='error']\n"
                      f"print('OK' if not miss and not e else f'BAD miss={{len(miss)}} err={{len(e)}}')\n")
            r = run([str(py), "-c", probe2], cwd=tmp)
            loaded.append((f.name, "OK" in r.stdout))
        check("every shipped demo domain loads clean",
              all(ok for _, ok in loaded),
              ", ".join(n for n, ok in loaded if not ok) or f"{len(loaded)} domains")

        # Full suite in the fresh environment -- the one that matters.
        r = run([str(py), "-m", "pytest", str(src / "tests"), "-q"], cwd=src,
                timeout=1200)
        tail = (r.stdout.strip().splitlines() or [""])[-1]
        check("full test suite passes in the fresh env", r.returncode == 0, tail[:160])

        print()
        failed = [n for n, ok, _ in RESULTS if not ok]
        if failed:
            print(f"ACCEPTANCE FAILED: {len(failed)} of {len(RESULTS)} checks")
            for n in failed:
                print(f"  - {n}")
            return 1
        print(f"ACCEPTANCE PASSED: {len(RESULTS)}/{len(RESULTS)} checks")
        return 0
    finally:
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"\nvenv kept at {tmp}")


if __name__ == "__main__":
    sys.exit(main())
