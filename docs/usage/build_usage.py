"""
Build USAGE.mdx from the section scripts in sections/.

    python build_usage.py            # every section
    python build_usage.py 04 09      # only these (the others are kept from the last build)

Each section is a plain Python script in "percent" cells:

    # %% [markdown]
    # Text of the guide (Markdown).
    # %%
    code that is run

The cells of one section run in order in one namespace, in their own
working folder (work/<section>/), so the files a cell writes stay there.
Every code cell goes into the guide with what it printed, and each figure
it saved under FIG (a variable the cell sees) is shown under it. A cell
that raises stops the build: every example in the guide has run.
"""
import contextlib
import io
import json
import os
import re
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
sys.path.insert(0, str(PKG))
SECTIONS = HERE / "sections"
FIGS = HERE / "figures"
WORK = HERE / "work"
CACHE = HERE / "build_cache.json"
MAX_OUT = 60          # printed lines kept per cell


def cells(path):
    """[(kind, text)] with kind 'md' or 'code'"""
    out, kind, buf = [], None, []
    for ln in path.read_text().splitlines():
        if ln.startswith("# %%"):
            if kind:
                out.append((kind, "\n".join(buf).strip("\n")))
            kind, buf = ("md" if "[markdown]" in ln else "code"), []
            continue
        if kind == "md":
            buf.append(ln[2:] if ln.startswith("# ") else ln.lstrip("#"))
        elif kind == "code":
            buf.append(ln)
    if kind:
        out.append((kind, "\n".join(buf).strip("\n")))
    return [(k, t) for k, t in out if t.strip()]


def mdx_text(s):
    """Markdown prose for MDX: braces and '<' outside `code` are escaped."""
    parts = re.split(r"(`[^`]*`)", s)
    for i in range(0, len(parts), 2):
        parts[i] = (parts[i].replace("{", "&#123;").replace("}", "&#125;")
                    .replace("<", "&lt;"))
    return "".join(parts)


def scrub(text):
    """Local paths out of the printed output (the guide is shared)."""
    text = text.replace(str(WORK), "work").replace(str(PKG), "pyauto-xpp")
    text = re.sub(r"/tmp/[^\s'\"]*pyxpp_[^/\s'\"]*", "<workdir>", text)
    return text.replace(str(Path.home()), "~")


def run_section(path):
    sid = path.stem
    work = WORK / sid
    shutil.rmtree(work, ignore_errors=True)     # files of a previous build would make XPP ask "Overwrite?"
    work.mkdir(parents=True, exist_ok=True)
    fig_dir = FIGS / sid
    fig_dir.mkdir(parents=True, exist_ok=True)
    for f in fig_dir.glob("*"):
        f.unlink()
    ns = {"__name__": "__main__", "FIG": fig_dir, "PKG": PKG}
    blocks = []
    cwd = os.getcwd()
    os.chdir(work)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        for kind, text in cells(path):
            if kind == "md":
                blocks.append({"md": text})
                continue
            before = {f: f.stat().st_mtime for f in fig_dir.glob("*")}
            buf = io.StringIO()
            t0 = time.time()
            with warnings.catch_warnings(record=True) as wl, contextlib.redirect_stdout(buf):
                warnings.simplefilter("always")
                try:
                    exec(compile(text, f"{sid}", "exec"), ns)
                except Exception:
                    print(buf.getvalue(), file=sys.__stdout__)
                    traceback.print_exc()
                    raise SystemExit(f"\ncell failed in {path.name}:\n{text}")
            plt.close("all")
            out = scrub(buf.getvalue()).rstrip("\n").splitlines()
            for w in wl:
                if "pyxpp" in str(w.category.__module__) or w.category.__name__ == "AnswersNotUsed":
                    out.append(f"{w.category.__name__}: {str(w.message).splitlines()[0]}")
            if len(out) > MAX_OUT:
                out = out[:MAX_OUT - 5] + ["..."] + out[-4:]
            new = sorted(f.name for f in fig_dir.glob("*")
                         if f.suffix in (".png", ".svg") and before.get(f) != f.stat().st_mtime)
            blocks.append({"code": text, "out": "\n".join(out), "figs": new,
                           "sec": round(time.time() - t0, 1)})
    finally:
        os.chdir(cwd)
    return blocks


def title_of(blocks):
    for b in blocks:
        if "md" in b:
            m = re.match(r"#+\s*(.+)", b["md"])
            if m:
                return m.group(1).strip()
    return "?"


def anchor(t):
    return re.sub(r"[^a-z0-9 -]", "", t.lower()).strip().replace(" ", "-")


def render(results):
    order = sorted(results)
    lines = [
        "---",
        "title: pyauto-xpp usage",
        "description: Every part of pyauto-xpp with an example that has been run",
        "---",
        "",
        "# pyauto-xpp: usage and examples",
        "",
        "pyauto-xpp runs XPPAUT 8.4 and its own AUTO from Python, for batch work and",
        "Python workflows. Every example below was run with the package when this page was",
        "built (`docs/usage/build_usage.py`); the output under each example is what it",
        "printed, and each figure is what it saved. The import name is `pyxpp`.",
        "",
        "## Contents",
        "",
    ]
    for k, sid in enumerate(order, 1):
        t = title_of(results[sid])
        lines.append(f"{k}. [{mdx_text(t)}](#{anchor(t)})")
    lines.append("")
    for sid in order:
        for b in results[sid]:
            if "md" in b:
                lines += [mdx_text(b["md"]), ""]
                continue
            lines += ["```python", b["code"], "```", ""]
            if b["out"].strip():
                lines += ["```text", b["out"], "```", ""]
            for f in b["figs"]:
                lines += [f"![{f}](figures/{sid}/{f})", ""]
    return "\n".join(lines).rstrip() + "\n"


def main(argv):
    want = set(argv)
    results = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    paths = sorted(SECTIONS.glob("[0-9]*.py"))
    results = {k: v for k, v in results.items() if k in {p.stem for p in paths}}
    for p in paths:
        if want and not any(p.stem.startswith(w) for w in want):
            continue
        t0 = time.time()
        print(f"{p.stem} ...", end=" ", flush=True)
        results[p.stem] = run_section(p)
        CACHE.write_text(json.dumps(results, indent=1))     # keep each finished section
        print(f"{time.time() - t0:.0f} s", flush=True)
    (HERE / "USAGE.mdx").write_text(render(results))
    print("wrote", HERE / "USAGE.mdx")


if __name__ == "__main__":
    main(sys.argv[1:])
