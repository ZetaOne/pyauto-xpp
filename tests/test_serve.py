"""pyxpp serve: the local server behind the editor on the project page."""
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from pyxpp import serve as S


@pytest.fixture
def server(tmp_path):
    srv = S.Server(0, tmp_path, set(S.ORIGINS) | {"http://127.0.0.1:9"})
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.stop_all()
    srv.shutdown()
    srv.server_close()


def call(url, body=None, method=None, headers=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method or ("GET" if body is None else "POST"),
                                 headers=dict({"Content-Type": "application/json"}, **(headers or {})))
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def run(base, code, session="t"):
    job = call(base + "/run", {"code": code, "session": session})["job"]
    t = time.time()
    while time.time() - t < 60:
        d = call(f"{base}/job/{job}")
        if d["done"]:
            return d
        time.sleep(0.1)
    raise TimeoutError


def test_ping_and_status(server):
    srv, base = server
    assert call(base + "/ping")["app"] == "pyxpp"
    st = call(base + "/status")
    assert st["workdir"] == str(srv.workdir) and "python" in st


def test_run_prints_figures_and_keeps_variables(server):
    _, base = server
    d = run(base, "import pyxpp, numpy as np\nimport matplotlib.pyplot as plt\n"
                  "m = pyxpp.load(text=\"x'=-x\\ninit x=1\\n@ total=2\\ndone\\n\")\n"
                  "tr = m.run()\nprint('x2', tr['X'][-1])\nplt.plot(tr['t'], tr['X'])\nkeep = 7\n")
    assert d["ok"]
    text = "".join(c.get("text", "") for c in d["chunks"])
    assert abs(float(text.split("x2")[1].split()[0]) - 0.1353352832) < 1e-6
    assert sum(c["t"] == "fig" for c in d["chunks"]) == 1
    d = run(base, "print(keep * 6)")
    assert d["ok"] and "42" in d["chunks"][0]["text"]


def test_errors_and_a_crash_do_not_stop_the_server(server):
    _, base = server
    d = run(base, "1/0")
    assert not d["ok"] and "ZeroDivisionError" in d["chunks"][-1]["text"]
    d = run(base, "import os\nos._exit(3)")
    assert not d["ok"] and "exit code 3" in d["chunks"][-1]["text"]
    assert run(base, "print('again')")["ok"]


def test_files_stay_in_the_work_folder(server):
    _, base = server
    call(base + "/file", {"path": "m/a.ode", "text": "x'=-x\ndone\n"}, method="PUT")
    assert call(base + "/file?path=m/a.ode")["text"].startswith("x'")
    assert [f["path"] for f in call(base + "/files")["files"]] == ["m/a.ode"]
    for bad in ("../x.txt", "/etc/passwd"):
        with pytest.raises(urllib.error.HTTPError) as e:
            call(base + "/file?path=" + bad)
        assert e.value.code == 403
    with pytest.raises(urllib.error.HTTPError):
        call(base + "/file", {"path": "run.sh", "text": "echo"}, method="PUT")


def test_other_sites_and_rebinding_are_refused(server):
    _, base = server
    for h in ({"Origin": "https://example.com"}, {"Host": "evil.example"}):
        with pytest.raises(urllib.error.HTTPError) as e:
            call(base + "/ping", headers=h)
        assert e.value.code == 403
    assert call(base + "/ping", headers={"Origin": "https://zetaone.github.io"})["app"] == "pyxpp"


def test_update_only_in_the_installers_environment(server):
    _, base = server
    with pytest.raises(urllib.error.HTTPError) as e:
        call(base + "/update", {})
    assert e.value.code == 400


def test_stops_a_while_after_the_last_editor_tab(tmp_path):
    srv = S.Server(0, tmp_path, set(S.ORIGINS))
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()
    threading.Thread(target=S._idle_watch, args=(srv, 3), daemon=True).start()
    try:
        time.sleep(4)
        assert call(base + "/ping")["app"] == "pyxpp"        # no tab ever came: it keeps running
        for _ in range(4):                                  # a tab's heartbeat keeps it up
            call(base + "/ping?tab=1")
            time.sleep(1.5)
        assert t.is_alive()
        t.join(10)                                          # the tab is gone: it stops by itself
        assert not t.is_alive()
    finally:
        srv.server_close()


def test_package_updates_only_in_the_installers_environment(server):
    _, base = server
    assert call(base + "/outdated") == {"can_update": False, "packages": []}
    with pytest.raises(urllib.error.HTTPError) as e:
        call(base + "/upgrade", {"packages": ["numpy"]})
    assert e.value.code == 400
