"""
Run every worked-example notebook headless and print one line each: time, checks.

    python run_all.py              # all notebooks
    python run_all.py 01 06a       # only some (by the start of the file name)
    python run_all.py --save       # also write the outputs back into the notebooks

The notebooks run on the Jupyter kernel "python3", in this folder.  Each one
ends with a cell that prints "N/M checks passed" for the whole notebook.
"""
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

HERE = Path(__file__).resolve().parent
KERNEL = "python3"


def last_count(nb):
    """the last "N/M checks passed" printed by the notebook's last code cell"""
    code = [c for c in nb.cells if c.cell_type == "code"]
    text = "".join(o.get("text", "") for o in code[-1].get("outputs", []) if o.output_type == "stream")
    found = re.findall(r"(\d+)/(\d+) checks passed", text)
    return tuple(map(int, found[-1])) if found else None


def main(args):
    save = "--save" in args
    want = [a for a in args if a != "--save"]
    books = sorted(HERE.glob("[0-9]*.ipynb"))
    if want:
        books = [p for p in books if any(p.stem.startswith(w) for w in want)]
    total_ok = total = failed = 0
    t_all = time.time()
    for p in books:
        nb = nbformat.read(p, as_version=4)
        client = NotebookClient(nb, kernel_name=KERNEL, timeout=None,
                                resources={"metadata": {"path": str(HERE)}})
        t0 = time.time()
        err = None
        try:
            client.execute()
        except CellExecutionError as e:
            err = str(e).strip().splitlines()[-1][:120]
        dt = time.time() - t0
        if save:
            nbformat.write(nb, p)
        count = None if err else last_count(nb)
        if count:
            ok, n = count
            total_ok, total = total_ok + ok, total + n
            failed += ok != n
            status = "ok" if ok == n else "CHECKS FAILED"
            print(f"{p.stem:50s} {dt:7.1f} s  {ok}/{n} checks  {status}", flush=True)
        else:
            failed += 1
            print(f"{p.stem:50s} {dt:7.1f} s  ERROR  {err or 'no check count found'}", flush=True)
    print(f"\n{len(books)} notebooks, {failed} with problems; {total_ok}/{total} checks passed "
          f"({time.time() - t_all:.0f} s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
