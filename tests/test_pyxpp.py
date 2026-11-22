"""
pyxpp test suite.  Run:  python -m pytest tests -q

Reference values come from running the same .ode through the XPPAUT
command-line binary or from the GUI's AUTO window on the same settings.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

import pyxpp

HERE = Path(__file__).resolve().parent
LECAR = HERE.parent / "examples" / "lecar.ode"

VDP = """
x'=y
y'=-x+mu*(1-x^2)*y
par mu=1
init x=2,y=0
@ total=50,dt=0.01
done
"""


@pytest.fixture(autouse=True)
def _in_tmp_dir(tmp_path, monkeypatch):
    """every test runs in its own temporary folder: models built with
    ModelBuilder write their .ode files there, not into the package"""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def lecar():
    m = pyxpp.load(LECAR)
    yield m
    m.close()


def test_load_names(lecar):
    assert [v.lower() for v in lecar.vars] == ["v", "w"]
    assert [a.lower() for a in lecar.aux] == ["cur", "ica", "ik", "rate"]
    assert "iapp" in lecar.pars and lecar.pars["iapp"] == 0.0
    assert lecar.numerics["total"] == 30


def test_run_matches_cli_endpoint(lecar):
    tr = lecar.run()
    assert tr.array.shape == (601, 7)
    # equilibrium at iapp=0 (same as AUTO's EP below)
    assert tr["v"][-1] == pytest.approx(-0.4956167, abs=1e-6)


def test_set_params_and_options(lecar):
    lecar.set(iapp=0.1, total=100, dt=0.025, meth="rk4")
    assert lecar.numerics["total"] == 100
    assert lecar.numerics["dt"] == 0.025
    tr = lecar.run()
    assert len(tr) == 4001
    assert tr["v"].max() > 0.2          # oscillates at iapp=0.1


def test_equilibrium(lecar):
    eq = lecar.equilibrium({"v": -0.5, "w": 0})
    assert eq.stable
    assert eq.x["V"] == pytest.approx(-0.4956167, abs=1e-6)
    assert np.all(eq.eigenvalues.real < 0)


def test_buttons_and_transcript(lecar):
    lecar.buttons.numerics.dt(0.01)
    assert lecar.numerics["dt"] == 0.01
    assert "Delta t" in lecar.transcript
    lecar.buttons.numerics.method.cvode()
    assert lecar.numerics["meth"] == "cvode"
    lecar.buttons.parameters("iapp", 0.07, "")
    assert lecar.pars["iapp"] == 0.07


def test_unused_answers_warn(lecar):
    with pytest.warns(pyxpp.AnswersNotUsed):
        lecar.buttons.erase("not asked")


def test_bad_menu_answer(lecar):
    with pytest.raises(pyxpp.XPPError, match="no entry"):
        lecar.buttons.initialconds("zz")


def test_exit_does_not_kill_python(lecar):
    with pytest.raises(pyxpp.XPPError, match="exit"):
        lecar.buttons.file.quit(True)


def test_endless_loop_guard(lecar):
    with pytest.raises(pyxpp.XPPError, match="without an answer"):
        lecar.buttons.file.tutorial()


def test_bad_ode():
    with pytest.raises(pyxpp.XPPError):
        pyxpp.load(text="x'=y+\ndone\n")


def test_two_models_are_independent(lecar):
    m2 = pyxpp.load(text=VDP)
    lecar.pars["iapp"] = 0.05
    tr = m2.run()
    assert tr["x"].max() == pytest.approx(2.0, abs=0.05)
    assert lecar.pars["iapp"] == 0.05
    assert "mu" not in lecar.pars


def test_nullclines(lecar):
    nc = lecar.nullclines()
    assert set(nc) == {"V", "W"}
    assert all(len(a) > 50 for a in nc.values())


def test_auto_steady_state_and_hopf(lecar):
    lecar.run(total=200)
    lecar.run_last(total=10)
    a = lecar.auto
    br = a.run("iapp", start="steady", ds=0.01, dsmax=0.02, parmin=-0.2, parmax=0.5)
    types = [(p.type, round(p.x, 4)) for p in br.labeled()]
    assert ("LP", 0.0833) in types
    assert ("LP", -0.0207) in types
    assert ("HB", 0.2038) in types
    hb = br.special("HB")[0]
    po = a.run(grab=hb.label, choice="periodic", dsmax=0.05, nmax=400)
    assert all(p.periodic for p in po)
    # cycles fold at iapp ~ 0.242 (subcritical Hopf)
    assert any(p.type == "LP" and abs(p.x - 0.2420) < 1e-3 for p in po)
    assert len(a.diagram) == len(br) + len(po)


def test_auto_two_parameter_fold(lecar):
    lecar.run(total=200)
    lecar.run_last(total=10)
    a = lecar.auto
    a.run("iapp", start="steady", ds=0.01, dsmax=0.02, parmin=-0.2, parmax=0.5)
    two = a.run(grab=2, param2="gca", nmax=300, parmax=0.6)
    assert len(two) > 10 and all(p.two_par for p in two)


def test_every_button_is_callable_without_crashing():
    """Each button pressed with no answers in a fresh process (a crash in C
    would take the test runner down otherwise)."""
    import subprocess
    import sys
    from pyxpp.buttons import Menu

    m = pyxpp.load(LECAR)
    paths = []

    def walk(menu, prefix):
        for n in menu._children:
            b = getattr(menu, n)
            paths.append(prefix + [n])
            if isinstance(b, Menu):
                walk(b, prefix + [n])

    walk(m.buttons, ["buttons"])
    walk(m.auto.buttons, ["auto", "buttons"])
    child = (
        "import sys, warnings; warnings.simplefilter('ignore')\n"
        "import pyxpp\n"
        f"m = pyxpp.load({str(LECAR)!r})\n"
        "m.run(total=200); m.run_last(total=5)\n"
        "if sys.argv[1] == 'auto':\n"
        "    m.auto.run('iapp', start='steady')  # so AUTO buttons have a diagram\n"
        "b = m\n"
        "for p in sys.argv[1:]: b = getattr(b, p)\n"
        "try:\n"
        "    b()\n"
        "except pyxpp.XPPError:\n"
        "    pass\n"
        "print('DONE')\n")
    crashed = []
    for p in paths:
        r = subprocess.run([sys.executable, "-c", child, *p], capture_output=True,
                           text=True, timeout=120, cwd=str(HERE.parent))
        if "DONE" not in r.stdout:
            crashed.append((".".join(p), r.returncode, r.stderr[-300:]))
    assert not crashed, crashed


@pytest.mark.parametrize("script", sorted((HERE.parent / "examples" / "scripts").glob("*.py")),
                         ids=lambda p: p.name)
def test_example_scripts_run(script, tmp_path):
    """Each AUTO example script runs and writes its figure."""
    import subprocess
    import sys
    import os
    # find pyxpp even when it is not installed (tests run from a checkout)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(HERE.parent), os.environ.get("PYTHONPATH", "")])}
    r = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True,
                       text=True, timeout=600, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    assert list(tmp_path.glob("*.png"))


def test_numpy_scalar_answers(lecar):
    lecar.set(total=np.float64(12.5))
    assert lecar.numerics["total"] == 12.5
    lecar.buttons.numerics.dt(np.float32(0.25))
    assert lecar.numerics["dt"] == 0.25


def test_run_objects_leave_the_model_unchanged(lecar, tmp_path):
    from pyxpp import Run
    before = lecar.snapshot()
    base = Run(total=100, dt=0.05, meth="rk4")
    trs = lecar.run_many([base, base.with_pars(iapp=0.1).with_ics(v=-0.3, w=0),
                          base.replace(meth="cvode", tol=1e-9, atol=1e-9)])
    assert [len(t) for t in trs] == [2001, 2001, 2001]
    assert trs[1]["v"][1000:].max() > 0.2 and trs[0]["v"][1000:].max() < 0   # keeps spiking only with iapp = 0.1
    assert trs[1].run.pars == {"iapp": 0.1}
    assert lecar.snapshot() == before
    t = lecar.run(total=20, iapp=0.1, v=-0.2, keep=False)
    assert t.run.pars == {"iapp": 0.1} and t.run.ics == {"v": -0.2} and lecar.snapshot() == before
    a = lecar.run(base.with_pars(iapp=0.1))
    b = lecar.run(base.with_pars(iapp=0.1).replace(start="last"))
    assert np.allclose(b.array[0, 1:3], a.array[-1, 1:3])
    p = base.replace(name="x").save(tmp_path / "r.json")
    assert Run.load(p) == base.replace(name="x")
    with lecar.using(meth="cvode", iapp=0.05):
        assert lecar.numerics["meth"] == "cvode" and lecar.pars["iapp"] == 0.05
    assert lecar.snapshot() == before


def test_vector_field_is_xpps_rhs(lecar):
    import math
    lecar.pars["iapp"] = 0.08
    f = lecar.rhs({"v": -0.3, "w": 0.1})
    p = lecar.pars
    v, w = -0.3, 0.1
    minf = .5 * (1 + math.tanh((v - p["v1"]) / p["v2"]))
    assert f["V"] == pytest.approx(p["iapp"] + p["gl"] * (p["vl"] - v) + p["gk"] * w * (p["vk"] - v)
                                   - p["gca"] * minf * (v - 1), rel=1e-12)
    X, Y, U, V = lecar.vector_field("v", "w", -0.6, 0.5, -0.1, 0.6, grid=10)
    assert U.shape == (11, 11) and U[3, 4] == pytest.approx(lecar.rhs({"v": X[3, 4], "w": Y[3, 4]})["V"])
    for scaled in (False, True):
        a = lecar.direction_field("v", "w", -0.6, 0.5, -0.1, 0.6, grid=16, scaled=scaled)
        assert a.shape == (17 * 17, 4)
        r = a[100]
        d = lecar.rhs({"v": r[0], "w": r[1]})
        dx, dy = r[2:] - r[:2]
        assert dx * d["W"] - dy * d["V"] == pytest.approx(0, abs=1e-6)        # parallel


def test_lyapunov_uses_the_points_own_data():
    from pyxpp.builder import ModelBuilder
    m = (ModelBuilder("fhn").parameter(i=0.2, a=0.7, b=0.8, eps=0.08).ode("v", "v-v^3/3-w+i")
         .ode("w", "eps*(v+a-b*w)").initial(v=-1.2, w=-0.6).option(total=200)).load()
    m.run(); m.run_last()
    hbs = m.auto.run("i", start="steady", ds=0.01, dsmax=0.02, parmin=0, parmax=2, nmax=500).special("HB")
    before = [m.auto.lyapunov(p)[0] for p in hbs]
    m.auto.reset(); m.set(i=1.0); m.run(total=300); m.run_last()
    m.auto.run("i", start="steady", ds=-0.01, dsmax=0.02, parmin=0, parmax=2, nmax=50)   # another diagram
    after = [m.auto.lyapunov(p)[0] for p in hbs]
    assert after == pytest.approx(before) and before[0] == pytest.approx(0.9716, abs=1e-3)


def _mb(name):
    from pyxpp.builder import ModelBuilder
    return ModelBuilder(name)


def test_method_names_are_translated_for_xpp():
    from pyxpp.builder import ModelBuilder
    for alias, want in (("dopri5", "5dp"), ("dormprince", "5dp"), ("rk4", "rungekutta"), ("stiff", "stiff")):
        mb = ModelBuilder("m").ode("x", "-x").initial(x=1).option(meth=alias)
        assert f"meth={want}" in mb.render()
        assert mb.load().numerics["meth"] == want
    with pytest.raises(ValueError):
        ModelBuilder("m").option(meth="nosuchmethod")


def test_numpy_numbers_are_written_as_plain_numbers():
    import numpy as np
    mb = _mb("np").ode("x", "-a*x").parameter(a=np.float64(1.4), n=np.int64(3)).initial(x=np.float32(0.5))
    text = mb.render()
    assert "a=1.4" in text and "n=3" in text and "np." not in text
    assert mb.load().pars["a"] == pytest.approx(1.4)


def test_second_auto_run_from_scratch_returns_the_whole_new_branch():
    m = _mb("lin").ode("x", "a-x").parameter(a=0).initial(x=0).option(total=20).load()
    m.run(); m.run_last()
    first = m.auto.run("a", start="steady", ds=0.05, dsmax=0.1, parmin=-1, parmax=1, nmax=200)
    m.set(a=0, x=0); m.run(); m.run_last()
    second = m.auto.run("a", start="steady", ds=0.05, dsmax=0.1, parmin=-1, parmax=1, nmax=200)
    assert len(second) >= len(first) * 0.8 > 0


def test_newton_options_from_the_ode_file_are_kept():
    m = _mb("nt").ode("x", "-x").initial(x=1).option(newt_tol=1e-9, jac_eps=1e-7).load()
    assert m.numerics["newt_tol"] == pytest.approx(1e-9) and m.numerics["jac_eps"] == pytest.approx(1e-7)


def test_delay_buffer_follows_a_smaller_dt():
    # Mackey-Glass oscillates at tau = 17; with the old bug a smaller dt than at load killed the delay
    mb = (_mb("mgdt").ode("x", "0.2*delay(x,tau)/(1+delay(x,tau)^10)-0.1*x").parameter(tau=17)
          .initial(x=0.5).option(total=600, dt=0.1, delay=20))
    m = mb.load()
    amp = [np.ptp(m.run(keep=False, dt=dt, trans=300)["x"]) for dt in (0.1, 0.05)]
    assert min(amp) > 0.5 and amp[1] == pytest.approx(amp[0], rel=0.2)


def test_volterra_kernel_follows_a_parameter_change():
    # x' = -int{exp(-t/tau)#x}: compare a parameter change with a freshly loaded model
    def mb(tau):
        return _mb(f"vk{int(tau*10)}").ode("x", "-int{exp(-t/tau)#x}").parameter(tau=tau).initial(x=1).option(total=20, dt=0.05)
    changed = mb(1.0).load()
    a = changed.run(keep=False, tau=3.0)["x"]
    b = mb(3.0).load().run(keep=False)["x"]
    assert np.max(np.abs(a - b)) < 1e-6


def test_equilibrium_with_an_unstable_real_direction_is_found():
    # Rossler: the outer equilibrium has one unstable real eigenvalue (XPP's eigenvector step used to fail)
    m = (_mb("ros").ode("x", "-y-z").ode("y", "x+a*y").ode("z", "b+z*(x-c)")
         .parameter(a=0.2, b=0.2, c=5.7).initial(x=1, y=1, z=0)).load()
    m.set(newt_tol=1e-10, jac_eps=1e-8)
    xo = (5.7 + np.sqrt(5.7**2 - 4 * 0.04)) / 2
    e = m.equilibrium({"x": xo, "y": -xo / 0.2, "z": xo / 0.2})
    assert e.x["X"] == pytest.approx(xo, rel=1e-6)


def test_delay_history_follows_the_initial_value_unless_given():
    m = (_mb("cuspd").parameter(a=-1.9, b=3).ode("x", "a+b*delay(x,1)-delay(x,1)^3").initial(x=1.2274)
         .option(total=50, dt=0.01, delay=1, meth="rk4", bound=1000)).load()
    assert m.has_delays
    assert m.run(keep=False)["x"][-1] == pytest.approx(1.17739, abs=1e-3)   # the past is 1.2274, not 0
    m.history(x="1+0.1*cos(t)")
    assert m.ics["x"] == pytest.approx(1.1)
    m.run(keep=False, x=0.3)
    assert m.history()["X"] == "1+0.1*cos(t)"                                # an explicit history is kept


def test_delay_equilibrium_gives_the_characteristic_roots():
    m = _mb("twodel").parameter(a=0.8).ode("x", "-a*(delay(x,1)+delay(x,1.7))").initial(x=0).option(delay=2).load()
    roots = m.equilibrium({"x": 0}).eigenvalues
    assert roots[0] == pytest.approx(complex(0.11191770, 1.24352376), abs=1e-7) or \
        roots[0] == pytest.approx(complex(0.11191770, -1.24352376), abs=1e-7)
    assert any(abs(r - complex(-1.33013388, 4.45755707)) < 1e-7 for r in roots)


def _fhn_auto():
    m = (_mb("fhn2").parameter(i=0.2, a=0.7, b=0.8, eps=0.08).ode("v", "v-v^3/3-w+i")
         .ode("w", "eps*(v+a-b*w)").initial(v=-1.2, w=-0.6).option(total=200)).load()
    m.auto.parameters("i", "eps")
    m.run(); m.run_last()
    return m, m.auto.run("i", start="steady", ds=0.01, dsmax=0.02, parmin=0, parmax=2, nmax=500)


def test_point_multipliers_and_eigenvalues():
    m, d = _fhn_auto()
    p = d[40]
    assert np.allclose(np.sort(np.exp(p.eigenvalues).real), np.sort(p.multipliers.real), rtol=3e-3)
    po = m.auto.run(grab=d.special("HB")[0].label, choice="periodic", ntst=40, ds=0.01, dsmax=0.05, nmax=20)
    with pytest.raises(AttributeError):
        po[5].eigenvalues
    assert np.isclose(po[5].multipliers, 1, atol=1e-3).any()          # the trivial Floquet multiplier


def test_unanswered_auto_menu_raises():
    m, d = _fhn_auto()
    with pytest.raises(pyxpp.XPPError, match="no answer"):
        m.auto.run(grab=d.special("HB")[0].label)                     # XPP asks what to do at HB


def test_two_parameter_curves_use_tight_tolerances():
    m, d = _fhn_auto()
    before = (m.auto.settings["epsl"], m.auto.settings["epsu"])
    c = m.auto.run(grab=d.special("HB")[0].label, choice="two param", param2="eps", ds=0.005, dsmax=0.02,
                   nmax=300, parmin=0, parmax=2)
    assert (m.auto.settings["epsl"], m.auto.settings["epsu"]) == before    # put back afterwards
    a, b = 0.7, 0.8
    errs = []
    for p in c:
        e = p.par["eps"]
        if 0.01 < e < 1:
            cc = np.sqrt(1 - e * b)
            errs.append(min(abs(p.x - (s * (cc**3 / 3 - cc + cc / b) + a / b)) for s in (-1, 1)))
    assert np.median(errs) < 1e-8 and max(errs) < 1e-5


# ---- 2026-11-17: starting orbits, stored solutions, double final state, long meshes ----
HOPF_NF = """
x'=mu*x-w*y-x*(x^2+y^2)
y'=w*x+mu*y-y*(x^2+y^2)
par mu=1,w=2
init x=0.1,y=0
@ total=5,dt=0.01
done
"""


def _hopf_branch_from_given_orbit(t=None):
    """periodic orbits of the Hopf normal form (radius sqrt(mu), period 2 pi / w),
    started from a given orbit, not from the data browser"""
    m = pyxpp.load(text=HOPF_NF)
    m.run()                                     # the browser holds a spiral, not an orbit
    if t is None:
        t = np.linspace(0, np.pi, 301)
    U = np.column_stack([np.cos(2 * t), np.sin(2 * t)])
    m.auto.parameters("mu", "w")
    d = m.auto.run("mu", start="periodic", initial=(t, U), ntst=50, nmax=30, npr=10,
                   ds=0.02, dsmax=0.1, parmin=0.1, parmax=3)
    return m, d


def test_periodic_run_from_a_given_orbit():
    m, d = _hopf_branch_from_given_orbit()
    assert len(d) > 10 and all(p.periodic for p in d)
    # the first point is the starting data (period in single precision); then AUTO's own
    for p in d.points[1:]:
        assert p.period == pytest.approx(np.pi, abs=1e-8)
        assert p.umax["x"] == pytest.approx(np.sqrt(p.x), abs=1e-6)
    # unevenly spaced times are resampled; a Trajectory works as well
    t = np.pi * np.linspace(0, 1, 400) ** 1.5
    m2, d2 = _hopf_branch_from_given_orbit(t)
    assert d2[3].period == pytest.approx(np.pi, abs=1e-8)
    tr = m.load_data(t, np.column_stack([np.cos(2 * t), np.sin(2 * t)]))
    assert len(tr) == 400 and tr["y"][-1] == pytest.approx(np.sin(2 * t[-1]), abs=1e-6)
    with pytest.raises(ValueError):
        m.auto.run("mu", start="steady", initial=tr)


def test_point_solution_matches_a_simulation():
    m, d = _hopf_branch_from_given_orbit()
    p = [q for q in d.labeled() if q.type == ""][0]
    t, U = p.solution
    assert t[0] == 0 and t[-1] == pytest.approx(1) and U.shape == (50 * 4 + 1, 2)
    assert np.allclose(np.hypot(U[:, 0], U[:, 1]), np.sqrt(p.x), atol=1e-8)
    tr, Ur = m.auto.solution(p.label, real_time=True)
    assert tr[-1] == pytest.approx(p.period)
    with m.using(mu=p.x, x=Ur[0, 0], y=Ur[0, 1], total=p.period, dt=p.period / 20000, meth="rk4"):
        m.run()
        s = m.data
    for j, v in enumerate(("x", "y")):
        assert np.abs(np.interp(tr, s["t"], s[v]) - Ur[:, j]).max() < 1e-6
    with pytest.raises(pyxpp.XPPError):
        m.auto.solution([q for q in d if not q.label][0])


def test_last_state_is_double_precision():
    m = pyxpp.load(text="x'=3*t^2\ny'=-y\ninit x=0,y=1\n@ total=1.37,dt=0.001,meth=rk4\ndone\n")
    with pytest.raises(pyxpp.XPPError):
        m.last_state()
    m.run()
    s = m.last_state()
    assert s["t"] == pytest.approx(1.37, abs=1e-12)
    assert abs(s["x"] - s["t"] ** 3) < 1e-12               # RK4 is exact for a cubic
    assert abs(s["Y"] - np.exp(-s["t"])) < 1e-12           # RK4 error ~ dt^4
    assert abs(m.last_point()["X"] - s["t"] ** 3) > 1e-10   # the browser keeps floats


def test_long_meshes_restart_without_crashing():
    # NTST*NCOL >= ~5000 used to make AUTO's .s header unreadable; restarting from a
    # label then crashed the whole process
    m = pyxpp.load(text=HOPF_NF.replace("w=2", "w=1"))
    m.run()
    m.auto.parameters("mu", "w")
    t = np.linspace(0, 2 * np.pi, 501)
    d = m.auto.run("mu", start="periodic", initial=(t, np.column_stack([np.cos(t), np.sin(t)])),
                   ntst=1400, ncol=4, nmax=8, npr=4, ds=0.01, dsmax=0.05, parmin=0, parmax=2)
    lab = [p for p in d.labeled() if p.type == ""][0]
    assert lab.solution[0].shape == (1400 * 4 + 1,)
    e = m.auto.run(grab=lab.label, choice="extend", nmax=4)
    assert len(e) == 4 and e[-1].period == pytest.approx(2 * np.pi, abs=1e-8)
    # a label whose stored solution is gone: an XPPError, not a crash
    for f in (m.workdir / ".pyxpp" / "auto").glob("*.s"):
        f.unlink()
    with pytest.raises(pyxpp.XPPError, match="restart"):
        m.auto.run(grab=e.labeled()[-1].label, choice="extend", nmax=4)


LORENZ = """
x'=s*(y-x)
y'=x*(r-z)-y
z'=x*y-b*z
par s=10,r=28,b=2.6666667
init x=1,y=1,z=20
@ total=20,dt=0.01,meth=rk4,maxstor=20000
done
"""


def test_lyapunov_spectrum_of_lorenz():
    # XPPAUT 8.4's QR method: the classic values 0.906, 0, -14.57, whose sum is the divergence
    m = pyxpp.load(text=LORENZ)
    m.run()
    ly = m.lyapunov_spectrum(time=1000, transient=50)
    lam = ly["exponents"]
    assert lam.shape == (3,)
    assert lam[0] == pytest.approx(0.906, abs=0.03)
    assert abs(lam[1]) < 0.01
    assert lam[2] == pytest.approx(-14.57, abs=0.05)
    assert ly["sum"] == pytest.approx(-(10 + 1 + 8 / 3), abs=1e-3)
    assert ly["divergence"] == pytest.approx(ly["sum"], abs=1e-3)
    assert ly["kaplan_yorke"] == pytest.approx(2.06, abs=0.01)


def test_new_start_keeping_the_diagram_numbers_on():
    # XPPAUT 8.4: a second run from scratch that keeps the diagram continues its labels
    # and branch numbers, and the first run's points can still be continued from
    m = pyxpp.load(LECAR)
    m.run(total=200); m.run_last(total=200)
    d1 = m.auto.run("iapp", start="steady", ds=0.002, dsmax=0.05, nmax=200, parmin=-0.1, parmax=0.4)
    m.auto.clear_grab()
    m.pars["iapp"] = 0.3
    m.run(total=300); m.run_last(total=300)
    d2 = m.auto.run("iapp", start="steady", keep=True, ds=-0.002, dsmax=0.05, nmax=50,
                    parmin=-0.1, parmax=0.4)
    l1, l2 = [p.label for p in d1.labeled()], [p.label for p in d2.labeled()]
    assert min(l2) == max(l1) + 1
    assert {abs(p.branch) for p in d2} == {max(abs(p.branch) for p in d1) + 1}
    assert len(m.auto.diagram.labeled()) == len(l1) + len(l2)
    hb = [p for p in d1.labeled() if p.type == "HB"][0]
    po = m.auto.run(grab=hb.label, choice="periodic", nmax=20, ds=0.01)
    assert len(po) == 20 and po[-1].periodic
