"""Make JupyterLab open .py files as notebooks (jupytext) in this environment.

Writes share/jupyter/lab/settings/overrides.json under the running Python's
prefix, so it affects only this environment.  Run by `pixi run lab` and by
Binder's postBuild; safe to run again.
"""
import json
import sys
from pathlib import Path

f = Path(sys.prefix) / "share" / "jupyter" / "lab" / "settings" / "overrides.json"
f.parent.mkdir(parents=True, exist_ok=True)
d = json.loads(f.read_text()) if f.exists() else {}
d.setdefault("@jupyterlab/docmanager-extension:plugin", {}).setdefault("defaultViewers", {})["python"] = "Jupytext Notebook"
f.write_text(json.dumps(d, indent=2) + "\n")
