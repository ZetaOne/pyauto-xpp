"""
Model: one loaded .ode file running in its own private copy of XPPAUT.
"""
import ctypes
import os
import re
import warnings
from pathlib import Path

import numpy as np

from ._lib import ESC, Lib, XPPError, format_answer, new_workdir
from . import buttons as _buttons

METHODS = ["discrete", "euler", "modeuler", "rungekutta", "adams", "gear", "volterra",
           "backeul", "qualrk", "stiff", "cvode", "5dp", "83dp", "2rb", "ymp"]
# XPP's own key letters for METH=..., same order as METHODS
_METHOD_KEYS = "demragvbqsc582y"
_METHOD_ALIASES = {
    "rk4": "r", "runge-kutta": "r", "rungekutta": "r", "runge_kutta": "r",
    "discrete": "d", "euler": "e", "modeuler": "m", "mod_euler": "m", "adams": "a",
    "gear": "g", "volterra": "v", "backeul": "b", "qualrk": "q", "qualst": "q",
    "stiff": "s", "cvode": "c", "dopri5": "5", "dp5": "5", "5dp": "5", "dopri83": "8",
    "dp83": "8", "83dp": "8", "rosen23": "2", "rb23": "2", "2rb": "2", "symplectic": "y",
    "ymp": "y",
    # XPP's documented spellings and common variants
    "5dorpri": "5", "dorpri5": "5", "dormprince": "5", "dormand-prince": "5", "dormandprince": "5",
    "83dorpri": "8", "dorpri8": "8", "dorpri83": "8", "rosenbrock": "2", "symplec": "y",
    "backeuler": "b", "backward-euler": "b", "rk": "r",
}


class AnswersNotUsed(UserWarning):
    """Some answers passed to a button were never asked for."""


class Trajectory:
    """Contents of XPP's data browser: column 0 is t, then every variable and
    auxiliary quantity in .ode order.  Index by name or use .array."""

    def __init__(self, array, names, run=None):
        self.array = array
        self.names = list(names)
        self.run = run          # the pyxpp.Run that made it, if any
        self._idx = {n.lower(): i for i, n in enumerate(self.names)}

    def __getitem__(self, name):
        if isinstance(name, str):
            try:
                return self.array[:, self._idx[name.lower()]]
            except KeyError:
                raise KeyError(f"{name!r}; columns are {self.names}") from None
        return self.array[name]

    def __len__(self):
        return self.array.shape[0]

    def __repr__(self):
        return f"<Trajectory {self.array.shape[0]} rows x {self.names}>"

    @property
    def t(self):
        return self.array[:, 0]

    def to_pandas(self):
        import pandas as pd
        return pd.DataFrame(self.array, columns=self.names)


class Equilibrium:
    def __init__(self, x, eigenvalues, discrete=False):
        self.x = x
        self.eigenvalues = eigenvalues
        if discrete:
            self.stable = bool(np.all(np.abs(eigenvalues) < 1))
        else:
            self.stable = bool(np.all(eigenvalues.real < 0))

    def __repr__(self):
        xs = ", ".join(f"{k}={v:.6g}" for k, v in self.x.items())
        return f"<Equilibrium {'stable' if self.stable else 'unstable'} {xs}>"


class _Named:
    """dict-like view onto parameters or initial conditions"""

    def __init__(self, model, names, getter, setter, kind):
        self._m, self._names = model, names
        self._get, self._set, self._kind = getter, setter, kind

    def _key(self, name):
        for n in self._names:
            if n.lower() == name.lower():
                return n
        raise KeyError(f"no {self._kind} {name!r}; have {self._names}")

    def __getitem__(self, name):
        return self._get(self._key(name))

    def __setitem__(self, name, value):
        self._set(self._key(name), float(value))

    def __iter__(self):
        return iter(self._names)

    def __len__(self):
        return len(self._names)

    def __contains__(self, name):
        return any(n.lower() == str(name).lower() for n in self._names)

    def keys(self):
        return list(self._names)

    def items(self):
        return [(n, self[n]) for n in self._names]

    def update(self, d=None, **kw):
        for k, v in {**(d or {}), **kw}.items():
            self[k] = v

    def as_dict(self):
        return dict(self.items())

    def __repr__(self):
        return f"{self._kind}s(" + ", ".join(f"{k}={v:g}" for k, v in self.items()) + ")"


class Model:
    """
    A running XPPAUT session without the X11 interface.

        m = pyxpp.load("lecar.ode")
        m.pars["iapp"] = 0.1
        tr = m.run(total=200)            # Initialconds > Go
        eq = m.equilibrium()             # Sing pts, non-interactive
        m.buttons.numerics.dt(0.01)      # any GUI button
        m.auto.run(...)                  # bifurcation diagrams

    Parameters
    ----------
    path : .ode file.  `text=` may be given instead (written to the workdir).
    workdir : where XPP writes its files (default: a fresh temporary folder,
        removed when the model is freed or Python exits).  XPP is run with
        this as the current directory during calls that write files.
    quiet : send XPP/AUTO terminal output to `m.log` instead of stdout.
    args : extra XPP command-line arguments, e.g. ["-setfile", "x.set"].
    """

    def __init__(self, path=None, *, text=None, workdir=None, quiet=True, args=()):
        if (path is None) == (text is None):
            raise ValueError("give exactly one of path or text")
        self.workdir = Path(workdir).resolve() if workdir else new_workdir()
        self.workdir.mkdir(parents=True, exist_ok=True)
        if not workdir:
            # a temporary working folder is removed when the model is freed or Python exits;
            # give workdir= to keep XPP's files
            import weakref
            self._libs = []     # released before the folder is removed (Windows: open files)
            self._cleanup = weakref.finalize(self, _remove_workdir, str(self.workdir), self._libs)
        if text is not None:
            path = self.workdir / "model.ode"
            path.write_text(text if text.endswith("\n") else text + "\n")
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.quiet = quiet
        self._lib = Lib(self.workdir / ".pyxpp")
        if getattr(self, "_libs", None) is not None:
            self._libs.append(self._lib)
        self._c = self._lib.c
        autodir = self.workdir / ".pyxpp" / "auto"
        autodir.mkdir(parents=True, exist_ok=True)
        self._c.pyxpp_set_auto_dir(str(autodir).encode())

        argv = [b"xppaut", str(self.path).encode(), *[str(a).encode() for a in args]]
        arr = (ctypes.c_char_p * len(argv))(*argv)
        with self._call():
            rc = self._c.pyxpp_init(len(argv), arr, str(self._lib.logpath).encode())
        if rc != 0:
            raise XPPError(f"loading {self.path.name} failed: "
                           f"{self._c.pyxpp_error().decode()}\n--- XPP log ---\n{self.log[-3000:]}")

        node, nmark, neq = self._c.pyxpp_node(), self._c.pyxpp_nmarkov(), self._c.pyxpp_neq()
        allnames = [self._c.pyxpp_var_name(i).decode() for i in range(neq)]
        self.vars = allnames[:node + nmark]
        self.aux = allnames[node + nmark:neq]
        self.columns = ["t"] + allnames
        self.par_names = [self._c.pyxpp_par_name(i).decode() for i in range(self._c.pyxpp_npar())]

        self._last_ic = (ctypes.c_double * 1).in_dll(self._c, "last_ic")
        addr = ctypes.addressof(self._last_ic)
        self._last_ic = (ctypes.c_double * max(1, len(self.vars))).from_address(addr)

        self.pars = _Named(self, self.par_names, self._get_val, self._set_val, "parameter")
        self.ics = _Named(self, self.vars, self._get_ic, self._set_ic, "initial condition")
        # delay equations: which variables have a history of their own (x(0)=formula in
        # the file, or history()); the others follow their initial value (see history())
        src = "\n".join(ln.split("#")[0] for ln in self.path.read_text(errors="replace").splitlines())
        self.has_delays = bool(re.search(r"\bdelay\s*\(", src, re.I))
        self._history_given = {i for i in range(len(self.vars))
                               if self._c.pyxpp_delay_string(i).decode().strip() not in ("0.0", "")}
        self.buttons = _buttons.build_main(self)
        from .auto import Auto
        self.auto = Auto(self)
        self._last_transcript = ""

    # ------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------
    def _call(self):
        """context: chdir to workdir and capture C stdout"""
        model = self

        class _Ctx:
            def __enter__(self):
                self.cwd = os.getcwd()
                os.chdir(model.workdir)
                self.cap = model._lib.captured(model.quiet)
                self.cap.__enter__()

            def __exit__(self, *exc):
                self.cap.__exit__(*exc)
                os.chdir(self.cwd)
                return False
        return _Ctx()

    def _run(self, fn, *args, answers=()):
        self._c.pyxpp_transcript_clear()
        self._lib.tape(answers)
        with self._call():
            rc = fn(*args)
        self._last_transcript = self._c.pyxpp_transcript().decode(errors="replace")
        left = self._lib.leftover()
        self._c.pyxpp_tape_clear()
        if rc != 0:
            msg = self._c.pyxpp_error().decode(errors="replace")
            raise XPPError(msg + (f"\n--- transcript ---\n{self._last_transcript}"
                                  if self._last_transcript else ""))
        if left:
            warnings.warn(f"{left} answer(s) were not asked for.\n"
                          f"--- transcript ---\n{self._last_transcript}",
                          AnswersNotUsed, stacklevel=4)
        return self._last_transcript

    def _press_main(self, path, answers):
        self._c.pyxpp_main_menu()
        return self._run(self._c.pyxpp_key, ord(path[0]), answers=[*path[1:], *answers])

    def _press_numerics(self, path, answers):
        if not path:
            return ""
        return self._run(self._c.pyxpp_num_key, ord(path[0]), answers=[*path[1:], *answers])

    def _press_file(self, path, answers):
        if not path:
            return ""
        self._c.pyxpp_main_menu()
        self._c.pyxpp_key(ord("f"))
        return self._run(self._c.pyxpp_key, ord(path[0]), answers=[*path[1:], *answers])

    def _press_auto(self, path, answers):
        if not path:
            return self._run(self._c.pyxpp_auto_open)
        return self._run(self._c.pyxpp_auto_key, ord(path[0]), answers=[*path[1:], *answers])

    def press(self, key, *answers):
        """Press one key in XPP's main window and answer what it asks.
        `m.press("i", "g")` is Initialconds > Go."""
        if key in ("\x1b", "esc", "Esc", "ESC"):
            self._c.pyxpp_main_menu()
            return ""
        return self._press_main([key], answers)

    def command(self, com, *answers):
        """Run an internal XPP command id (the M_* constants in menudrive.h)."""
        return self._run(self._c.pyxpp_command, int(com), answers=answers)

    @property
    def transcript(self):
        """Questions XPP asked during the last call and the answers used."""
        return self._last_transcript

    @property
    def log(self):
        """Everything XPP and AUTO printed so far."""
        self._c.pyxpp_flush_log()
        try:
            return self._lib.logpath.read_text(errors="replace")
        except FileNotFoundError:
            return ""

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    def _get_val(self, name):
        v = ctypes.c_double()
        if self._c.pyxpp_get_val(name.encode(), ctypes.byref(v)) != 0:
            raise KeyError(name)
        return v.value

    def _set_val(self, name, value):
        if self._c.pyxpp_set_val(name.encode(), value) != 0:
            raise KeyError(name)

    def _get_ic(self, name):
        return self._last_ic[self.vars.index(name)]

    def _set_ic(self, name, value):
        self._last_ic[self.vars.index(name)] = value

    def get(self, name):
        """Current value of any parameter, variable or derived quantity."""
        if name in self.ics:
            return self.ics[name]
        return self._get_val(name)

    def set(self, **options):
        """Set parameters, initial conditions and @-options in one call.

            m.set(iapp=0.1, v=-0.3, total=200, dt=0.01, meth="cvode")

        Names are tried as parameter, then variable (initial condition),
        then as an XPP @ option (anything legal on an `@` line).
        """
        for k, v in options.items():
            if k in self.pars:
                self.pars[k] = v
            elif k in self.ics:
                self.ics[k] = v
            else:
                self.set_option(k, v)
        return self

    def set_option(self, name, value):
        """Exactly like `@ name=value` in the .ode file."""
        if name.lower() in ("meth", "method"):
            value = method_key(value)
        r = self._c.pyxpp_set_option(str(name).encode(), format_answer(value).encode())
        if r != 0:
            raise XPPError(self._c.pyxpp_error().decode())

    @property
    def numerics(self):
        """Main integration settings (read-only snapshot; change with set())."""
        g = lambda t, n: t.in_dll(self._c, n).value  # noqa: E731
        d = ctypes.c_double
        i = ctypes.c_int
        return {
            "total": g(d, "TEND"), "dt": g(d, "DELTA_T"), "t0": g(d, "T0"),
            "trans": g(d, "TRANS"), "nout": g(i, "NJMP"), "bound": g(d, "BOUND"),
            "meth": METHODS[g(i, "METHOD")] if 0 <= g(i, "METHOD") < len(METHODS) else g(i, "METHOD"),
            "tol": g(d, "TOLER"), "atol": g(d, "ATOLER"), "dtmin": g(d, "HMIN"),
            "dtmax": g(d, "HMAX"), "maxstor": g(i, "MAXSTOR"), "delay": g(d, "DELAY"),
            # XPP's variable names are confusing: @ newt_tol sets EVEC_ERR (Newton's
            # tolerance) and @ jac_eps sets NEWT_ERR (the Jacobian's difference step)
            "newt_tol": g(d, "EVEC_ERR"), "jac_eps": g(d, "NEWT_ERR"),
            "newt_iter": g(i, "EVEC_ITER"), "poimap": g(i, "POIMAP"),
        }

    # ------------------------------------------------------------------
    # integration
    # ------------------------------------------------------------------
    @property
    def data(self):
        """The data browser (last integration) as a Trajectory."""
        rows, cols = self._c.pyxpp_data_rows(), self._c.pyxpp_data_cols()
        out = np.empty((rows, cols), dtype=float)
        if rows:
            self._c.pyxpp_data_copy(out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return Trajectory(out, self.columns[:cols])

    def history(self, **histories):
        """The past of a delay equation (t < 0): a number or a formula in t.

            m.history(x=1.2)                       # x(t) = 1.2 for t <= 0
            m.history(x="0.5+0.2*sin(3*t)")        # XPP's  x(0)=0.5+0.2*sin(3*t)

        The initial value x(0) is set to the history at t = 0.  Variables with
        no history of their own (none in the file, none given here) keep a
        constant past equal to their initial value, so `init x=1.2` or
        `run(x=1.2)` means x has always been 1.2.  (XPP itself would use a past
        of 0 there.)  With no arguments: the current history strings."""
        if not histories:
            return {v: self._c.pyxpp_delay_string(i).decode() for i, v in enumerate(self.vars)}
        for name, h in histories.items():
            i = [v.lower() for v in self.vars].index(name.lower())
            text = format_answer(h) if not isinstance(h, str) else h
            if self._c.pyxpp_set_delay_string(i, text.encode()) != 0:
                raise XPPError(f"history of {name}: {text!r} is too long (79 characters at most)")
            self._history_given.add(i)
            self._set_ic(self.vars[i], self.calc(text, t=0.0))

    def calc(self, expr, **values):
        """Evaluate an expression with the model's parameters (XPP's calculator).
        Keyword values (e.g. t=0) are set for the evaluation only."""
        keep = {k: self._get_val(k) for k in values}
        try:
            for k, v in values.items():
                self._set_val(k, v)
            z = ctypes.c_double()
            if self._c.pyxpp_calc(str(expr).encode(), ctypes.byref(z)) != 0:
                raise XPPError(f"cannot evaluate {expr!r}")
            return z.value
        finally:
            for k, v in keep.items():
                self._set_val(k, v)

    def _sync_history(self):
        """constant past = initial value, for delay variables without a history"""
        if not self.has_delays:
            return
        for i, v in enumerate(self.vars):
            if i not in self._history_given:
                self._c.pyxpp_set_delay_string(i, repr(float(self._last_ic[i])).encode())

    def run(self, run=None, *, keep=True, **settings):
        """Integrate (Initialconds > Go) and return the trajectory.

            m.run(total=200, iapp=0.1)        # keyword settings go through set() and stay
            m.run(total=200, keep=False)      # ... or only for this run
            m.run(Run(total=200, meth="cvode", pars={"iapp": 0.1}))   # a Run: only for this run

        With a `Run` (see pyxpp.run) or keep=False the model's numerics,
        parameters and initial conditions are put back afterwards, and the
        result's `.run` is the Run used."""
        if run is None and keep:
            if settings:
                self.set(**settings)
            self._sync_history()
            self._press_main(["i", "g"], ())
            return self.data
        if run is None:
            run = self.as_run(**settings)
        elif settings:
            run = self.as_run(run, **settings)
        with self.using(run):
            self._sync_history()
            self._press_main(["i", "l" if run.start == "last" else "g"], ())
            out = self.data
        out.run = run
        return out

    def run_many(self, runs):
        """One run per Run (or dict of settings); the model is unchanged
        afterwards.  Returns the trajectories, each with its `.run`."""
        from .run import Run
        return [self.run(r if isinstance(r, Run) else self.as_run(**dict(r))) for r in runs]

    def as_run(self, base=None, **settings):
        """A Run from keyword settings as set() takes them (parameters,
        initial conditions, numerics, other @ options), on top of `base`."""
        from dataclasses import fields as _fields
        from .run import Run
        base = base or Run()
        names = {f.name for f in _fields(Run)} - {"pars", "ics", "options", "start", "name"}
        pars, ics, opts, direct = dict(base.pars), dict(base.ics), dict(base.options), {}
        for k, v in settings.items():
            if k in self.pars:
                pars[k] = v
            elif k in self.ics:
                ics[k] = v
            elif k.lower() in names:
                direct[k.lower()] = v
            elif k.lower() == "method":
                direct["meth"] = v
            elif k in ("start", "name"):
                direct[k] = v
            else:
                opts[k] = v
        return base.replace(pars=pars, ics=ics, options=opts, **direct)

    def snapshot(self):
        """The current numerics, parameters and initial conditions (restore() puts them back)."""
        return {"numerics": self.numerics, "pars": self.pars.as_dict(), "ics": self.ics.as_dict()}

    def restore(self, snap):
        n = snap["numerics"]
        self.set_option("meth", n["meth"])          # first: the method may reset nout
        for k in ("total", "dt", "t0", "trans", "nout", "bound", "tol", "atol", "dtmin", "dtmax", "delay"):
            self.set_option(k, n[k])
        self.pars.update(snap["pars"])
        self.ics.update(snap["ics"])

    def using(self, run=None, **settings):
        """Context manager: apply a Run (or keyword settings, as in set())
        inside the block and put the model back afterwards.

            with m.using(meth="cvode", tol=1e-9, iapp=0.1):
                a = m.run(); b = m.run_last()

        Numerics, parameters and initial conditions are put back; other
        @ options given in Run.options stay as set."""
        model = self

        class _Using:
            def __enter__(self_):
                self_.snap = model.snapshot()
                if run is not None:
                    for k, v in run.settings().items():
                        model.set_option(k, v)
                    model.pars.update(run.pars)
                    model.ics.update(run.ics)
                if settings:
                    model.set(**settings)
                return model

            def __exit__(self_, *exc):
                model.restore(self_.snap)
                return False
        return _Using()

    def run_last(self, **settings):
        """Integrate from the end of the previous run (Initialconds > Last)."""
        if settings:
            self.set(**settings)
        self._press_main(["i", "l"], ())
        return self.data

    def continue_run(self, total):
        """Continue the current run until time `total` (Continue button)."""
        self._press_main(["c"], (total,))
        return self.data

    def last_point(self):
        """Final state of the last run as {var: value}."""
        d = self.data
        if len(d) == 0:
            raise XPPError("no data; run first")
        return {v: float(d[v][-1]) for v in self.vars}

    def last_state(self):
        """Where the last integration stopped, in double precision:
        {"t": final time, var: value, ...} (variables in .ode order; the
        names are case-insensitive).

        XPP integrates in double precision but keeps the data browser
        (m.data, last_point()) in single precision, about 7 digits; this
        reads XPP's own double state.  It is the state where the integration
        stopped (also with nOut > 1, a transient or a Poincare map, where the
        last stored row may be another point).  Importing an AUTO orbit into
        the data browser (File > Import orbit) reuses that state, so read it
        right after the run."""
        n = len(self.vars)
        x = (ctypes.c_double * max(n, 1))()
        t = ctypes.c_double()
        if self._c.pyxpp_last_state(x, ctypes.byref(t)) != 0:
            raise XPPError("no integration yet: run first")
        from .auto import CIDict
        return CIDict([("t", t.value)] + [(v, float(x[i])) for i, v in enumerate(self.vars)])

    def load_data(self, t, U=None):
        """Put given data into XPP's data browser, as if an integration had
        produced it: `t` (n) and `U` (n x len(m.vars)), or a Trajectory.  The
        auxiliary quantities are computed by XPP; the values are stored in
        single precision, like XPP's own.  Returns m.data.

        This is what AUTO's runs from scratch ("periodic", "bdry value",
        "homoclinic") start from; m.auto.run(..., initial=...) uses it."""
        t, U = _orbit_arrays(self, t, U)
        maxstor = self.numerics["maxstor"]
        if not 2 <= len(t) <= maxstor:
            raise XPPError(f"{len(t)} rows do not fit in XPP's storage (2 to MAXSTOR = {maxstor}; "
                           "raise it with m.set(maxstor=...))")
        rows = np.ascontiguousarray(np.column_stack([t, U]), dtype=float)
        rc = self._c.pyxpp_load_storage(len(t), U.shape[1],
                                        rows.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if rc != 0:
            raise XPPError(self._c.pyxpp_error().decode(errors="replace"))
        return self.data

    def sweep(self, name, values, **settings):
        """Run once per value of parameter (or initial condition) `name`,
        restoring it afterwards.  Returns a list of Trajectories.  (The GUI's
        Initialconds > Range is also available as a button.)"""
        target = self.pars if name in self.pars else self.ics
        old = target[name]
        out = []
        try:
            for v in values:
                target[name] = v
                out.append(self.run(**settings))
        finally:
            target[name] = old
        return out

    # ------------------------------------------------------------------
    # phase plane
    # ------------------------------------------------------------------
    def view(self, x=None, y=None, xlo=None, xhi=None, ylo=None, yhi=None):
        """Set the 2D view (Viewaxes > 2D).  Nullclines and direction fields
        are computed in this window, as in the GUI."""
        d = {}
        for k, v in (("X-axis", x), ("Y-axis", y), ("Xmin", xlo), ("Xmax", xhi),
                     ("Ymin", ylo), ("Ymax", yhi)):
            if v is not None:
                d[k] = v
        self._press_main(["v", "2"], (d,))

    def nullclines(self, x=None, y=None, xlo=None, xhi=None, ylo=None, yhi=None, mesh=None):
        """Compute nullclines in the current (or given) 2D view.  Returns
        {xvar: array (N,2), yvar: array (N,2)}; NaN rows separate pieces.
        Key xvar is the nullcline of the x-axis variable (its rhs = 0)."""
        if any(v is not None for v in (x, y, xlo, xhi, ylo, yhi)):
            self.view(x, y, xlo, xhi, ylo, yhi)
        if mesh is not None:
            self._press_numerics(["n"], (mesh,))
        self._press_main(["n", "n"], ())
        fname = ".pyxpp/nullclines.dat"
        path = self.workdir / fname
        if path.exists():
            path.unlink()
        self._press_main(["n", "s"], (fname,))
        if not path.exists():
            raise XPPError("XPP did not compute nullclines (is the view 2D "
                           "with two different variables?)")
        segs = {1: [], 2: []}
        last = None
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 3 or line.startswith("#"):
                if last is not None:
                    segs[last].append((np.nan, np.nan))
                continue
            k = int(float(parts[2]))
            segs[k].append((float(parts[0]), float(parts[1])))
            last = k
        # XPP plot indices: 0 is t, 1.. are the columns after t
        xi = ctypes.c_int.in_dll(self._c, "null_ix").value
        yi = ctypes.c_int.in_dll(self._c, "null_iy").value
        names = self.columns
        return {names[xi]: np.array(segs[1] or np.empty((0, 2))),
                names[yi]: np.array(segs[2] or np.empty((0, 2)))}

    # ------------------------------------------------------------------
    # vector field, computed by XPP
    # ------------------------------------------------------------------
    def rhs(self, point=None, t=0.0):
        """XPP's right-hand side at a point: {var: dvar/dt}.  `point` is a
        dict (missing variables take the current initial conditions) or a
        sequence; default: the current initial conditions."""
        n = self._c.pyxpp_node()
        y = (ctypes.c_double * max(n, len(self.vars) + len(self.aux) + 1))()
        dy = (ctypes.c_double * len(y))()
        for i in range(n):
            y[i] = self._last_ic[i]
        if isinstance(point, dict):
            g = {str(k).lower(): v for k, v in point.items()}
            for i in range(n):
                y[i] = g.get(self.vars[i].lower(), y[i])
        elif point is not None:
            for i, v in enumerate(point):
                y[i] = v
        self._run(self._c.pyxpp_rhs, float(t), y, dy)
        return {self.vars[i]: dy[i] for i in range(n)}

    def vector_field(self, x, y, xlo, xhi, ylo, yhi, grid=16, point=None, t=0.0):
        """XPP's vector field on a (grid+1) x (grid+1) mesh of the (x, y)
        plane, the other variables at `point` (default: initial conditions).
        Returns X, Y, U, V as (grid+1, grid+1) arrays (U = dx/dt, V = dy/dt),
        computed with XPP's own right-hand side (no integration)."""
        n = self._c.pyxpp_node()
        names = [v.lower() for v in self.vars]
        ix, iy = names.index(x.lower()), names.index(y.lower())
        base = (ctypes.c_double * max(n, len(self.vars) + len(self.aux) + 1))()
        dy = (ctypes.c_double * len(base))()
        for i in range(n):
            base[i] = self._last_ic[i]
        if isinstance(point, dict):
            g = {str(k).lower(): v for k, v in point.items()}
            for i in range(n):
                base[i] = g.get(names[i], base[i])
        xs, ys = np.linspace(xlo, xhi, grid + 1), np.linspace(ylo, yhi, grid + 1)
        X, Y = np.meshgrid(xs, ys)
        U, V = np.empty_like(X), np.empty_like(Y)
        with self._call():
            for j in range(grid + 1):
                for i in range(grid + 1):
                    base[ix], base[iy] = xs[i], ys[j]
                    if self._c.pyxpp_rhs(float(t), base, dy) != 0:
                        raise XPPError(self._c.pyxpp_error().decode())
                    U[j, i], V[j, i] = dy[ix], dy[iy]
        return X, Y, U, V

    def direction_field(self, x=None, y=None, xlo=None, xhi=None, ylo=None, yhi=None,
                        grid=16, scaled=False):
        """XPP's own direction field (Dir.field/flow) in the current or given
        2D view, as XPP would draw it: an (N, 4) array of arrows
        [x, y, x_end, y_end].  scaled=False is XPP's Direct field (arrow
        length proportional to |f| / max|f| on the grid); scaled=True is
        Scaled Dir.Fld (every arrow the same length on the screen).  The
        other variables are at their initial conditions."""
        if any(v is not None for v in (x, y, xlo, xhi, ylo, yhi)):
            self.view(x, y, xlo, xhi, ylo, yhi)
        fname = ".pyxpp/dirfield.dat"
        path = self.workdir / fname
        if path.exists():
            path.unlink()
        self._run(self._c.pyxpp_dirfield, int(grid), 0 if scaled else 1, fname.encode())
        if not path.exists():
            raise XPPError("XPP drew no direction field (is the view 2D with two different variables?)")
        return np.loadtxt(path, ndmin=2)

    # ------------------------------------------------------------------
    # equilibria
    # ------------------------------------------------------------------
    def equilibrium(self, guess=None, nroots=None):
        """Newton's method from `guess` (dict or sequence; default: current
        initial conditions).  Returns an Equilibrium with eigenvalues.
        The non-interactive core of Sing pts > Go.

        For a delay equation the eigenvalues are the `nroots` (default 12, at
        least the number of variables) rightmost roots of the characteristic
        equation det(lambda I - A0 - sum_k A_k exp(-lambda tau_k)) = 0, sorted
        by real part (XPP 8.2's delay spectrum, polished by Newton)."""
        n = self._c.pyxpp_node()
        x = (ctypes.c_double * n)()
        if guess is None:
            for i in range(n):
                x[i] = self._last_ic[i]
        elif isinstance(guess, dict):
            g = {str(k).lower(): v for k, v in guess.items()}
            for i in range(n):
                x[i] = g.get(self.vars[i].lower(), self._last_ic[i])
        else:
            for i, v in enumerate(guess):
                x[i] = v
        if self.has_delays:
            # the rightmost roots of the characteristic equation (XPP 8.2's delay spectrum)
            k = max(nroots or 12, n)
            ev, nun = (ctypes.c_double * (2 * k))(), ctypes.c_int(0)
            with self._call():
                nr = self._c.pyxpp_delay_roots(x, k, ev, ctypes.byref(nun))
            if nr <= 0:
                raise XPPError("equilibrium not found (delay equation)")
            roots = np.array([complex(ev[2 * j], ev[2 * j + 1]) for j in range(nr)])
            return Equilibrium({self.vars[i]: x[i] for i in range(n)}, roots)
        er, em, ierr = (ctypes.c_double * n)(), (ctypes.c_double * n)(), ctypes.c_int(0)
        self._run(self._c.pyxpp_equilibrium, x, er, em, ctypes.byref(ierr))
        if ierr.value != 0:
            raise XPPError(f"equilibrium not found (ierr={ierr.value})")
        ev = np.array([complex(er[i], em[i]) for i in range(n)])
        disc = ctypes.c_int.in_dll(self._c, "METHOD").value == 0
        return Equilibrium({self.vars[i]: x[i] for i in range(n)}, ev, discrete=disc)

    # ------------------------------------------------------------------
    def close(self):
        self._lib.close()
        if getattr(self, "_cleanup", None) is not None:
            self._cleanup()

    def __repr__(self):
        return (f"<pyxpp.Model {self.path.name}: vars={self.vars} aux={self.aux} "
                f"pars={len(self.par_names)}>")


def _orbit_arrays(model, t, U=None):
    """(t, U) from (t, U), a Trajectory or an (n x (1 + nvars)) array with t
    first; U gets one column per model variable (m.vars), in that order."""
    names = model.vars
    if U is None and isinstance(t, Trajectory):
        tr = t
        missing = [v for v in names if v.lower() not in tr._idx]
        if missing:
            raise ValueError(f"the Trajectory has no column(s) {missing}")
        t, U = tr["t"], np.column_stack([tr[v] for v in names])
    elif U is None:
        if isinstance(t, (tuple, list)) and len(t) == 2:
            t, U = t
        else:
            a = np.asarray(t, dtype=float)
            if a.ndim != 2:
                raise ValueError("give (t, U), a Trajectory, or an array with t in column 0")
            t, U = a[:, 0], a[:, 1:]
    if isinstance(U, dict):
        U = np.column_stack([np.asarray(U[v], dtype=float) for v in names])
    t = np.asarray(t, dtype=float).ravel()
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U[:, None]
    if U.shape[0] != len(t) and U.shape[1] == len(t):
        U = U.T
    if U.shape != (len(t), len(names)):
        raise ValueError(f"U must be {len(t)} x {len(names)} (rows = times, columns = {names}); "
                         f"got {U.shape}")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(U)):
        raise ValueError("t and U must be finite")
    if len(t) > 1 and not np.all(np.diff(t) > 0):
        raise ValueError("t must be strictly increasing")
    return t, U


def _remove_workdir(path, libs):
    for lib in libs:
        lib.close()
    import shutil
    shutil.rmtree(path, True)


def canonical_method(value):
    """the name XPP reads correctly for `@ meth=` (its first letter is XPP's key)"""
    return METHODS[_METHOD_KEYS.index(method_key(value))]


def method_key(value):
    v = str(value).strip().lower()
    if len(v) == 1 and v in _METHOD_KEYS:
        return v
    if v in _METHOD_ALIASES:
        return _METHOD_ALIASES[v]
    raise ValueError(f"unknown method {value!r}; choose from {sorted(_METHOD_ALIASES)}")


def load(path=None, **kw):
    """Load an .ode file.  See Model."""
    return Model(path, **kw)
