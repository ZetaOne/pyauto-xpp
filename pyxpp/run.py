"""
Run conditions as data.

A `Run` holds everything about *how* a model is integrated, apart from the
equations: numerics, the method and its settings, parameter values, initial
conditions, and any other `@` option.  The model file keeps the equations
(and its own defaults); a `Run` is applied for one run only and the model is
left as it was.

    from pyxpp import Run
    base = Run(total=300, dt=0.05, meth="rk4")
    stiff = base.replace(meth="cvode", tol=1e-9, atol=1e-9)
    pulse = base.with_pars(iapp=0.1).with_ics(v=-0.3, w=0)
    tr = m.run(stiff)                       # the model's settings are unchanged afterwards
    trs = m.run_many([base, stiff, pulse])  # tr.run is the Run that made tr
    stiff.save("stiff.json"); Run.load("stiff.json")

Fields left as None keep the model's current value.  Adaptive methods
(cvode, dopri5, dopri83, stiff, gear, rosen23, qualrk) store every step:
XPP sets nout = 1 for them.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace as _replace
from pathlib import Path

# Run field -> name of the XPP @ option
_NUMERICS = {"total": "total", "dt": "dt", "t0": "t0", "trans": "trans", "nout": "nout",
             "bound": "bound", "tol": "tol", "atol": "atol", "dtmin": "dtmin", "dtmax": "dtmax",
             "delay": "delay"}


@dataclass(frozen=True)
class Run:
    """How to integrate: numerics, method, parameters, initial conditions."""
    total: float | None = None
    dt: float | None = None
    meth: str | None = None
    t0: float | None = None
    trans: float | None = None
    nout: int | None = None
    bound: float | None = None
    tol: float | None = None          # relative tolerance (adaptive methods)
    atol: float | None = None         # absolute tolerance (adaptive methods)
    dtmin: float | None = None
    dtmax: float | None = None
    delay: float | None = None        # maximum delay (delay equations)
    pars: dict = field(default_factory=dict)
    ics: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)   # any other @ option, e.g. {"seed": 3}
    start: str = "ics"                # "ics": from the initial conditions; "last": from the end of the previous run
    name: str = ""

    def __post_init__(self):
        if self.start not in ("ics", "last"):
            raise ValueError("start must be 'ics' or 'last'")
        for f in ("pars", "ics", "options"):
            object.__setattr__(self, f, dict(getattr(self, f) or {}))

    # -- making variants ---------------------------------------------------
    def replace(self, **changes):
        """A copy with some fields changed (pars/ics/options are replaced
        whole; use with_pars / with_ics / with_options to add to them)."""
        return _replace(self, **changes)

    def with_pars(self, **pars):
        return _replace(self, pars={**self.pars, **pars})

    def with_ics(self, **ics):
        return _replace(self, ics={**self.ics, **ics})

    def with_options(self, **opts):
        return _replace(self, options={**self.options, **opts})

    @classmethod
    def from_model(cls, m, name=""):
        """The model's current settings, parameters and initial conditions as a Run."""
        n = m.numerics
        return cls(**{k: n[k if k != "bound" else "bound"] for k in _NUMERICS},
                   meth=n["meth"], pars=m.pars.as_dict(), ics=m.ics.as_dict(), name=name)

    # -- files ----------------------------------------------------------------
    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v not in (None, {}, "")}

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        bad = set(d) - known
        if bad:
            raise KeyError(f"unknown Run fields {sorted(bad)}")
        return cls(**d)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1) + "\n")
        return path

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def __repr__(self):
        body = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items() if k != "start" or v != "ics")
        return f"Run({body})"

    # -- applying -------------------------------------------------------------
    def settings(self):
        """the @ options of this Run, method first (it may reset nout)"""
        out = {}
        if self.meth is not None:
            out["meth"] = self.meth
        for k, opt in _NUMERICS.items():
            v = getattr(self, k)
            if v is not None:
                out[opt] = v
        out.update(self.options)
        return out
