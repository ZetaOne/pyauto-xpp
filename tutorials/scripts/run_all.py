"""
Run every tutorial script and print one line each: ok/FAILED, time, checks.

    python run_all.py              # all
    python run_all.py MAPS 2D      # only names starting with these
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ["2D-FLOWS", "EVENTS", "MAPS", "3D-FLOWS", "DELAYS", "NOISE", "SPACE"]


def key(p):
    group = p.stem.split("_")[0]
    return (ORDER.index(group) if group in ORDER else 99, p.stem)


def main(want):
    scripts = sorted((p for p in HERE.glob("*_*.py") if p.stem.split("_")[0] in ORDER), key=key)
    if want:
        scripts = [p for p in scripts if any(p.stem.startswith(w) for w in want)]
    total_ok = total = failed = 0
    for p in scripts:
        t0 = time.time()
        r = subprocess.run([sys.executable, p.name], cwd=HERE, capture_output=True, text=True,
                           env={**os.environ, "MPLBACKEND": "Agg"})   # no windows here
        dt = time.time() - t0
        m = re.search(r"(\d+)/(\d+) checks passed", r.stdout)
        if r.returncode == 0 and m:
            ok, n = int(m.group(1)), int(m.group(2))
            total_ok, total = total_ok + ok, total + n
            status = "ok" if ok == n else "CHECKS FAILED"
            failed += ok != n
            print(f"{p.stem:45s} {dt:6.1f} s  {ok}/{n} checks  {status}")
        else:
            failed += 1
            err = (r.stderr.strip().splitlines() or ["?"])[-1][:120]
            print(f"{p.stem:45s} {dt:6.1f} s  ERROR  {err}")
    print(f"\n{len(scripts)} scripts, {failed} with problems; {total_ok}/{total} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
