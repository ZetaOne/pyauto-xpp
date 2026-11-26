"""
pyxpp serve -- run pyauto-xpp on this computer for the editor on the project page
(https://zetaone.github.io/pyauto-xpp/editor/).

    python -m pyxpp serve [--port 8790] [--workdir ~/pyxpp-work] [--open]

The page sends code; a worker process (pyxpp/_worker.py) runs it with the
pyauto-xpp of this environment and sends back what it printed and its figures.

Safety:
  * listens on 127.0.0.1 only;
  * the Host header must be 127.0.0.1 or localhost (no DNS rebinding);
  * a browser may call it only from the project page's origin
    (https://zetaone.github.io), or an origin given with --allow-origin;
  * files are read and written only inside the work folder;
  * "update" can only upgrade pyauto-xpp from the project's own wheel list, and
    only in the environment the installer made (~/.pyxpp/env).
Only the standard library is used.
"""
from __future__ import annotations

import json
import mimetypes
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

SITE = os.environ.get("PYXPP_SITE", "https://zetaone.github.io/pyauto-xpp/").rstrip("/") + "/"
EDITOR = SITE + "editor/"
INDEX = os.environ.get("PYXPP_INDEX", SITE + "wheels/")
ORIGINS = {"https://zetaone.github.io", "{0.scheme}://{0.netloc}".format(urlparse(SITE))}
PORT = 8790
HOME = Path(os.environ.get("PYXPP_HOME", Path.home() / ".pyxpp"))
IDLE_EXIT = 60          # seconds without an open editor tab before the server stops (0: never)
WORKDIR = Path.home() / "pyxpp-work"
MARK = b"\x00PYXPP-"
TEXT_EXT = {".py", ".ipynb", ".ode", ".xps", ".txt", ".md", ".csv", ".dat", ".json", ".set", ".tab", ".ani", ".toml"}
SHOW_EXT = TEXT_EXT | {".png", ".svg", ".gif", ".pdf", ".npz", ".npy"}
MAX_TEXT = 2_000_000
MAX_NOTEBOOK = 40_000_000      # notebooks carry their figures


def version():
    try:
        from importlib.metadata import version as v
        return v("pyauto-xpp")
    except Exception:
        from . import __version__
        return __version__


def _vkey(s):
    return tuple(int(x) for x in re.findall(r"\d+", s)[:3])


def latest_version(timeout=10):
    """the newest pyauto-xpp on the project's wheel list"""
    with urllib.request.urlopen(INDEX, timeout=timeout) as r:
        html = r.read().decode(errors="replace")
    vs = set(re.findall(r"pyauto_xpp-(\d+\.\d+\.\d+)-", html))
    return max(vs, key=_vkey) if vs else None


def installer_env():
    """True when this Python is the environment the installer made (~/.pyxpp/env)"""
    try:
        return Path(sys.prefix).resolve() == (HOME / "env").resolve()
    except OSError:
        return False


def find_uv():
    p = HOME / "uv-path"
    if p.is_file():
        u = p.read_text(encoding="utf-8", errors="replace").strip().lstrip("\ufeff")
        if u and Path(u).exists():
            return u
    for c in (shutil.which("uv"), str(Path.home() / ".local" / "bin" / ("uv.exe" if os.name == "nt" else "uv")),
              str(Path.home() / ".cargo" / "bin" / "uv")):
        if c and Path(c).exists():
            return c
    return None


def recent_dirs():
    try:
        r = json.loads((HOME / "recent.json").read_text())
        return [d for d in r if Path(d).is_dir()][:8]
    except Exception:
        return []


def remember_dir(p):
    """the folder used last (pyxpp serve starts there next time) and the recent ones"""
    try:
        HOME.mkdir(parents=True, exist_ok=True)
        (HOME / "workdir").write_text(str(p))
        r = [str(p)] + [d for d in recent_dirs() if d != str(p)]
        (HOME / "recent.json").write_text(json.dumps(r[:8]))
    except OSError:
        pass


def last_workdir():
    try:
        p = Path((HOME / "workdir").read_text().strip())
        return p if p.is_dir() else None
    except OSError:
        return None


# ----------------------------------------------------------------------------
# sessions: one worker process each
# ----------------------------------------------------------------------------
class Job:
    def __init__(self, jid):
        self.id = jid
        self.chunks = []            # {"t": "out", "text"} or {"t": "fig", "png"}
        self.done = False
        self.ok = None
        self.line = None            # the line running now (of the editor's file)
        self.err_line = None        # the line an error came from
        self.started = time.time()
        self.ended = None

    def add(self, chunk):
        # merge consecutive text so the page gets fewer, larger pieces
        if chunk["t"] == "out" and self.chunks and self.chunks[-1]["t"] == "out" and len(self.chunks[-1]["text"]) < 8000:
            self.chunks[-1]["text"] += chunk["text"]
        else:
            self.chunks.append(chunk)

    def finish(self, ok):
        if not self.done:
            self.ok, self.done, self.ended = bool(ok), True, time.time()


class Session:
    def __init__(self, name, workdir):
        self.name, self.workdir = name, workdir
        self.proc = None
        self.job = None
        self.lock = threading.Lock()

    def _start(self):
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.proc = subprocess.Popen([sys.executable, "-u", "-m", "pyxpp._worker"], cwd=self.workdir, env=env,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     creationflags=flags)
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()

    def _read(self, proc):
        held = None                  # a blank line written just before a marker is not output
        for raw in iter(proc.stdout.readline, b""):
            job = self.job
            if raw.startswith(MARK):
                held = None
                parts = raw[len(MARK):].decode(errors="replace").strip().split(" ")
                if parts[0] == "FIG" and len(parts) >= 3 and job and parts[1] == job.id:
                    job.add({"t": "fig", "png": parts[2]})
                elif parts[0] in ("PLOTLY", "HTML") and len(parts) >= 3 and job and parts[1] == job.id:
                    job.add({"t": parts[0].lower(), "b64": parts[2]})
                elif parts[0] == "DONE" and len(parts) >= 3 and job and parts[1] == job.id:
                    job.finish(parts[2] == "1")
                elif parts[0] in ("LINE", "ERR") and len(parts) >= 3 and job and parts[1] == job.id:
                    try:
                        setattr(job, "line" if parts[0] == "LINE" else "err_line", int(parts[2]))
                    except ValueError:
                        pass
                continue
            if held is not None and job:
                job.add({"t": "out", "text": held})
            text = raw.decode("utf-8", errors="replace")
            held = text if text in ("\n", "\r\n") else None
            if held is None and job:
                job.add({"t": "out", "text": text})
        code = proc.wait()
        job = self.job
        if job and not job.done and self.proc is proc:
            job.add({"t": "out", "text": f"\n(the Python process stopped, exit code {code}; it starts again on the next run)\n"})
            job.finish(False)

    def run(self, code, file="<editor>", line0=0, mode="exec"):
        with self.lock:
            if self.job and not self.job.done:
                raise RuntimeError("busy: a run is still going in this session (stop it first)")
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            self.job = Job(uuid.uuid4().hex[:12])
            msg = json.dumps({"id": self.job.id, "code": code, "file": file, "line0": int(line0 or 0),
                              "mode": "cell" if mode == "cell" else "exec"}) + "\n"
            try:
                self.proc.stdin.write(msg.encode())
                self.proc.stdin.flush()
            except OSError:
                self._start()
                self.proc.stdin.write(msg.encode())
                self.proc.stdin.flush()
            return self.job

    def stop(self):
        with self.lock:
            p, self.proc = self.proc, None
            if p and p.poll() is None:
                p.kill()
            if self.job and not self.job.done:
                self.job.add({"t": "out", "text": "\n(stopped; the session's variables are gone)\n"})
                self.job.finish(False)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = os.name != "nt"

    def __init__(self, port, workdir, origins):
        super().__init__(("127.0.0.1", port), Handler)
        self.port, self.workdir, self.origins = port, workdir, origins
        self.sessions = {}
        self.jobs = {}
        self.slock = threading.Lock()
        self.started = time.time()
        self.last_tab = None            # last heartbeat of an open editor tab
        self.restart = False

    def tab_seen(self):
        self.last_tab = time.time()

    def busy(self):
        return any(s.job and not s.job.done for s in self.sessions.values())

    def session(self, name):
        name = re.sub(r"[^A-Za-z0-9_-]", "", name or "main")[:40] or "main"
        with self.slock:
            if name not in self.sessions:
                self.sessions[name] = Session(name, self.workdir)
            return self.sessions[name]

    def stop_all(self):
        for s in list(self.sessions.values()):
            s.stop()


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "pyxpp-serve"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):         # quiet; the page shows what matters
        if os.environ.get("PYXPP_SERVE_LOG"):
            super().log_message(fmt, *args)

    # -- checks ----------------------------------------------------------
    def _origin_ok(self):
        o = self.headers.get("Origin")
        return o is None or o in self.server.origins

    def _host_ok(self):
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        return host in ("127.0.0.1", "localhost", "::1")

    def _cors(self):
        o = self.headers.get("Origin")
        if o and o in self.server.origins:
            self.send_header("Access-Control-Allow-Origin", o)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Private-Network", "true")

    def _send(self, code, body=b"", ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, msg):
        self._send(code, {"error": msg})

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 60_000_000:
            raise ValueError("request too large")
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw or b"{}")

    def _guard(self):
        if not self._host_ok():
            self._err(403, "bad Host")
            return False
        if not self._origin_ok():
            self._err(403, "this page may not use pyxpp serve (origin not allowed)")
            return False
        return True

    def _path(self, rel):
        wd = self.server.workdir.resolve()
        p = (wd / (rel or "")).resolve()
        if p != wd and wd not in p.parents:
            raise PermissionError("outside the work folder")
        return p

    # -- verbs -----------------------------------------------------------
    def do_OPTIONS(self):
        if not self._guard():
            return
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if not self._guard():
            return
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if "tab" in q:                                   # an open editor tab says it is still there
            self.server.tab_seen()
        try:
            if u.path in ("/", "/index.html"):          # the editor itself, from this computer
                page = Path(__file__).with_name("editor") / "index.html"
                html = page.read_text(encoding="utf-8").replace("const SERVED_BY_PYXPP = false;",
                                                                 "const SERVED_BY_PYXPP = true;", 1)
                return self._send(200, html, "text/html; charset=utf-8")
            if u.path == "/ping":
                return self._send(200, {"app": "pyxpp", "version": version(), "api": 1})
            if u.path == "/status":
                return self._send(200, {
                    "app": "pyxpp", "api": 2, "home": str(Path.home()), "recent": recent_dirs(),
                    "version": version(), "python": platform.python_version(),
                    "system": platform.system(), "machine": platform.machine(), "prefix": sys.prefix,
                    "workdir": str(self.server.workdir), "can_update": installer_env() and find_uv() is not None,
                    "sessions": {k: bool(s.job and not s.job.done) for k, s in self.server.sessions.items()},
                    "uptime": round(time.time() - self.server.started)})
            if u.path == "/outdated":
                return self._send(200, outdated(max_age=0 if q.get("fresh") else 3600))
            if u.path == "/latest":
                return self._send(200, {"installed": version(), "latest": latest_version()})
            if u.path.startswith("/job/"):
                job = self.server.jobs.get(u.path[5:])
                if not job:
                    return self._err(404, "no such run")
                i = int(q.get("from", 0))
                return self._send(200, {"chunks": job.chunks[i:], "next": len(job.chunks), "done": job.done,
                                        "ok": job.ok, "seconds": round((job.ended or time.time()) - job.started, 2),
                                        "line": job.line, "err_line": job.err_line})
            if u.path == "/files":
                wd = self.server.workdir
                out, dirs = [], []
                for p in sorted(wd.rglob("*")):
                    rel = p.relative_to(wd)
                    if p.is_dir() and not any(x.startswith(".") or x == "__pycache__" for x in rel.parts):
                        if len(dirs) < 1000:
                            dirs.append(rel.as_posix())
                        continue
                    if p.is_file() and p.suffix.lower() in SHOW_EXT and not any(x.startswith(".") for x in rel.parts):
                        out.append({"path": rel.as_posix(), "size": p.stat().st_size, "mtime": p.stat().st_mtime,
                                    "text": p.suffix.lower() in TEXT_EXT})
                    if len(out) >= 2000:
                        break
                return self._send(200, {"workdir": str(wd), "files": out, "dirs": dirs})
            if u.path == "/dirs":                         # folder browser for "Open folder"
                home = Path.home().resolve()
                d = (home / (q.get("path") or "")).resolve()
                if d != home and home not in d.parents:
                    return self._err(403, "only folders in your home folder")
                sub = []
                try:
                    for c in sorted(d.iterdir(), key=lambda c: c.name.lower()):
                        if c.is_dir() and not c.name.startswith(".") and c.name != "__pycache__":
                            sub.append(c.name)
                        if len(sub) >= 500:
                            break
                except OSError as e:
                    return self._err(403, str(e))
                rel = d.relative_to(home).as_posix()
                return self._send(200, {"home": str(home), "path": "" if rel == "." else rel, "abs": str(d), "dirs": sub})
            if u.path == "/file":
                p = self._path(q.get("path"))
                if not p.is_file():
                    return self._err(404, "no such file")
                if p.stat().st_size > (MAX_NOTEBOOK if p.suffix.lower() == ".ipynb" else MAX_TEXT):
                    return self._err(413, "file too large to edit")
                return self._send(200, {"path": q.get("path"), "text": p.read_text(errors="replace")})
            if u.path == "/raw":
                p = self._path(q.get("path"))
                if not p.is_file():
                    return self._err(404, "no such file")
                return self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream")
            return self._err(404, "unknown")
        except PermissionError as e:
            return self._err(403, str(e))
        except Exception as e:
            return self._err(500, f"{type(e).__name__}: {e}")

    def do_PUT(self):
        if not self._guard():
            return
        u = urlparse(self.path)
        try:
            if u.path == "/file":
                b = self._body()
                p = self._path(b.get("path"))
                if p.suffix.lower() not in TEXT_EXT:
                    return self._err(400, "only text files (" + " ".join(sorted(TEXT_EXT)) + ")")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(b.get("text", ""))
                return self._send(200, {"saved": b.get("path")})
            return self._err(404, "unknown")
        except PermissionError as e:
            return self._err(403, str(e))
        except Exception as e:
            return self._err(500, f"{type(e).__name__}: {e}")

    def do_DELETE(self):
        if not self._guard():
            return
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/file":
                p = self._path(q.get("path"))
                if p.is_file():
                    p.unlink()
                return self._send(200, {"deleted": q.get("path")})
            return self._err(404, "unknown")
        except PermissionError as e:
            return self._err(403, str(e))

    def do_POST(self):
        if not self._guard():
            return
        u = urlparse(self.path)
        try:
            b = self._body()
            if u.path == "/run":
                s = self.server.session(b.get("session"))
                try:
                    job = s.run(b.get("code", ""), b.get("file") or "<editor>", b.get("line0") or 0, b.get("mode") or "exec")
                except RuntimeError as e:
                    return self._err(409, str(e))
                self.server.jobs[job.id] = job
                if len(self.server.jobs) > 200:                 # forget old finished runs
                    for k in [k for k, j in self.server.jobs.items() if j.done][:100]:
                        self.server.jobs.pop(k, None)
                return self._send(200, {"job": job.id, "session": s.name})
            if u.path == "/stop":
                self.server.session(b.get("session")).stop()
                return self._send(200, {"stopped": True})
            if u.path == "/workdir":
                p = Path(os.path.expanduser(b.get("path") or "")).resolve()
                home = Path.home().resolve()
                if p != home and home not in p.parents:
                    return self._err(403, "choose a folder in your home folder")
                if any(x.startswith(".") for x in p.relative_to(home).parts):
                    return self._err(403, "not a hidden folder")
                if not p.is_dir():
                    return self._err(404, "no such folder")
                self.server.stop_all()                   # the next run starts in the new folder
                self.server.workdir = p
                remember_dir(p)
                return self._send(200, {"workdir": str(p), "recent": recent_dirs()})
            if u.path == "/rename":
                a, z = self._path(b.get("from")), self._path(b.get("to"))
                if not a.exists():
                    return self._err(404, "no such file or folder")
                if z.exists():
                    return self._err(409, "that name is taken")
                z.parent.mkdir(parents=True, exist_ok=True)
                a.rename(z)
                return self._send(200, {"renamed": b.get("to")})
            if u.path == "/mkdir":
                self._path(b.get("path")).mkdir(parents=True, exist_ok=True)
                return self._send(200, {"made": b.get("path")})
            if u.path == "/update":
                return self._send(*update(self.server))
            if u.path == "/upgrade":
                return self._send(*upgrade(self.server, b.get("packages") or []))
            if u.path == "/shutdown":
                self._send(200, {"bye": True})
                threading.Thread(target=_shutdown, args=(self.server,), daemon=True).start()
                return
            return self._err(404, "unknown")
        except Exception as e:
            return self._err(500, f"{type(e).__name__}: {e}")


def _shutdown(server, restart=False):
    time.sleep(0.3)                    # let the reply reach the page first
    server.restart = restart
    server.shutdown()                  # serve() closes the port, then restarts if asked


_OUTDATED = {"t": 0.0, "packages": None}


def outdated(max_age=3600):
    """packages of this environment with a newer version (PyPI, and the project's wheel list),
    except pyauto-xpp itself, which has its own update button"""
    if not installer_env():
        return {"can_update": False, "packages": []}
    uv = find_uv()
    if not uv:
        return {"can_update": False, "packages": []}
    if _OUTDATED["packages"] is not None and time.time() - _OUTDATED["t"] < max_age:
        return {"can_update": True, "packages": _OUTDATED["packages"]}
    r = subprocess.run([uv, "pip", "list", "--outdated", "--format", "json", "--python", sys.executable,
                        "--find-links", INDEX], capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[-500:])
    pk = [{"name": d["name"], "version": d["version"], "latest": d["latest_version"]}
          for d in json.loads(r.stdout or "[]") if d.get("name", "").lower() != "pyauto-xpp"]
    _OUTDATED.update(t=time.time(), packages=pk)
    return {"can_update": True, "packages": pk}


def upgrade(server, names):
    """upgrade the named packages, only ones of this environment that `outdated` lists"""
    if not installer_env():
        return 400, {"error": "updates from the page work only in the environment the installer made"}
    allowed = {d["name"].lower(): d for d in outdated(max_age=0)["packages"]}
    names = [n for n in names if isinstance(n, str) and n.lower() in allowed]
    if not names:
        return 200, {"upgraded": [], "log": "nothing to update"}
    cmd = [find_uv(), "pip", "install", "--python", sys.executable, "--find-links", INDEX]
    for n in names:
        cmd += ["--upgrade-package", n]
    r = subprocess.run(cmd + names, capture_output=True, text=True, timeout=900)
    log = (r.stdout + r.stderr).strip()
    _OUTDATED.update(t=0.0, packages=None)
    if r.returncode != 0:
        return 500, {"error": "the update failed", "log": log}
    server.stop_all()                    # the next run starts a Python with the new versions
    return 200, {"upgraded": [{"name": n, "from": allowed[n.lower()]["version"], "to": allowed[n.lower()]["latest"]}
                              for n in names], "log": log}


def update(server):
    """upgrade pyauto-xpp in this environment from the project's wheel list, then restart"""
    if not installer_env():
        return 400, {"error": "updates from the page work only in the environment the installer made "
                              f"({HOME / 'env'}); this server runs in {sys.prefix}"}
    uv = find_uv()
    if not uv:
        return 400, {"error": "uv not found; run the installer again"}
    before = version()
    latest = latest_version()
    if latest and _vkey(latest) <= _vkey(before):
        return 200, {"updated": False, "version": before, "log": f"pyauto-xpp {before} is the newest"}
    cmd = [uv, "pip", "install", "--python", sys.executable, "--upgrade", "pyauto-xpp", "--find-links", INDEX]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    log = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        return 500, {"error": "the update failed", "log": log}
    threading.Thread(target=_shutdown, args=(server, True), daemon=True).start()
    return 200, {"updated": True, "from": before, "to": latest, "log": log, "restarting": True}


# ----------------------------------------------------------------------------
# starting, detaching, the pyxpp:// link
# ----------------------------------------------------------------------------
def ping(port=PORT, timeout=1.0):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/ping", timeout=timeout) as r:
            d = json.loads(r.read())
            return d if d.get("app") == "pyxpp" else None
    except Exception:
        return None


def _spawn_detached(args):
    HOME.mkdir(parents=True, exist_ok=True)
    log = open(HOME / "serve.log", "ab")
    if os.name == "nt":
        exe = Path(sys.executable)
        w = exe.with_name("pythonw.exe")
        if w.exists():
            args = [str(w)] + args[1:]
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=log, creationflags=flags, close_fds=True)
    else:
        subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, close_fds=True)


def start_background(port=PORT, wait=15.0):
    """start `pyxpp serve` in the background unless it already runs; True when it answers"""
    if ping(port):
        return True
    _spawn_detached([sys.executable, "-m", "pyxpp", "serve", "--port", str(port)])
    t = time.time()
    while time.time() - t < wait:
        if ping(port):
            return True
        time.sleep(0.25)
    return False


def open_url(url):
    """handler of pyxpp:// links: pyxpp://start starts the server in the background"""
    u = urlparse(url)
    action = (u.netloc or u.path.strip("/")).lower() or "start"
    if action == "start":
        start_background()
    elif action == "stop":
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/shutdown", data=b"{}", method="POST")
            urllib.request.urlopen(req, timeout=3)
        except Exception:
            pass
        t = time.time()
        while time.time() - t < 8 and ping():
            time.sleep(0.2)


def _python_for_links():
    exe = Path(sys.executable)
    if os.name == "nt":
        w = exe.with_name("pythonw.exe")
        return str(w if w.exists() else exe)
    return str(exe)


def register():
    """make pyxpp:// links start this environment's pyxpp (for this user only)"""
    py = _python_for_links()
    system = platform.system()
    if system == "Linux":
        apps = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "applications"
        apps.mkdir(parents=True, exist_ok=True)
        (apps / "pyxpp-handler.desktop").write_text(
            "[Desktop Entry]\nType=Application\nName=pyxpp\nComment=Starts pyxpp serve for the pyauto-xpp editor\n"
            f'Exec="{py}" -m pyxpp open-url %u\nNoDisplay=true\nTerminal=false\n'
            "MimeType=x-scheme-handler/pyxpp;\n")
        for cmd in (["xdg-mime", "default", "pyxpp-handler.desktop", "x-scheme-handler/pyxpp"],
                    ["update-desktop-database", str(apps)]):
            try:
                subprocess.run(cmd, check=False, capture_output=True, timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                pass
        return f"pyxpp:// registered ({apps / 'pyxpp-handler.desktop'})"
    if system == "Windows":
        import winreg
        base = r"Software\Classes\pyxpp"
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, base) as k:
            winreg.SetValueEx(k, "", 0, winreg.REG_SZ, "URL:pyxpp")
            winreg.SetValueEx(k, "URL Protocol", 0, winreg.REG_SZ, "")
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, base + r"\shell\open\command") as k:
            winreg.SetValueEx(k, "", 0, winreg.REG_SZ, f'"{py}" -m pyxpp open-url "%1"')
        return "pyxpp:// registered (HKEY_CURRENT_USER\\" + base + ")"
    if system == "Darwin":
        # an AppleScript applet: macOS hands URLs to apps as Apple events, not arguments
        app = Path.home() / "Applications" / "pyxpp.app"
        app.parent.mkdir(parents=True, exist_ok=True)
        if app.exists():
            shutil.rmtree(app)
        script = ('on open location theURL\n'
                  f'  do shell script quoted form of "{py}" & " -m pyxpp open-url " & quoted form of theURL '
                  '& " > /dev/null 2>&1 &"\n'
                  'end open location\n'
                  'on run\n'
                  f'  do shell script quoted form of "{py}" & " -m pyxpp open-url pyxpp://start > /dev/null 2>&1 &"\n'
                  'end run\n')
        subprocess.run(["osacompile", "-o", str(app), "-e", script], check=True, capture_output=True)
        plist = app / "Contents" / "Info.plist"
        pb = "/usr/libexec/PlistBuddy"
        for c in (["Add :CFBundleIdentifier string io.github.zetaone.pyxpp"],
                  ["Set :CFBundleIdentifier io.github.zetaone.pyxpp"],
                  ["Add :LSUIElement bool true"],
                  ["Add :CFBundleURLTypes array"],
                  ["Add :CFBundleURLTypes:0 dict"],
                  ["Add :CFBundleURLTypes:0:CFBundleURLName string pyxpp"],
                  ["Add :CFBundleURLTypes:0:CFBundleURLSchemes array"],
                  ["Add :CFBundleURLTypes:0:CFBundleURLSchemes:0 string pyxpp"]):
            subprocess.run([pb, "-c", c[0], str(plist)], capture_output=True)
        lsreg = ("/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/"
                 "Support/lsregister")
        subprocess.run([lsreg, "-f", str(app)], capture_output=True)
        return f"pyxpp:// registered ({app})"
    return f"pyxpp:// links are not supported on {system}; start with: python -m pyxpp serve"


def unregister():
    system = platform.system()
    if system == "Linux":
        apps = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "applications"
        (apps / "pyxpp-handler.desktop").unlink(missing_ok=True)
        try:
            subprocess.run(["update-desktop-database", str(apps)], capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            pass
    elif system == "Windows":
        import winreg
        for sub in (r"\shell\open\command", r"\shell\open", r"\shell", ""):
            try:
                winreg.DeleteKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\pyxpp" + sub)
            except OSError:
                pass
    elif system == "Darwin":
        app = Path.home() / "Applications" / "pyxpp.app"
        if app.exists():
            lsreg = ("/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/"
                     "Support/lsregister")
            subprocess.run([lsreg, "-u", str(app)], capture_output=True)
            shutil.rmtree(app, ignore_errors=True)
    return "pyxpp:// link removed"


def _idle_watch(srv, idle):
    """stop the server `idle` seconds after the last editor tab went away (never during a run)"""
    while True:
        time.sleep(2)
        if srv.last_tab is None or srv.busy():
            continue
        if time.time() - srv.last_tab > idle:
            print(f"no editor tab for {idle} s: stopping (start again with the editor's Start pyxpp)")
            sys.stdout.flush()
            srv.shutdown()
            return


def serve(port=PORT, workdir=None, origins=(), open_browser=False, idle_exit=IDLE_EXIT):
    workdir = Path(workdir).expanduser() if workdir else (last_workdir() or WORKDIR)
    workdir.mkdir(parents=True, exist_ok=True)
    if ping(port):
        print(f"pyxpp serve already runs on port {port}")
        if open_browser:
            import webbrowser
            webbrowser.open(EDITOR)
        return 0
    allowed = set(ORIGINS) | {o.rstrip("/") for o in origins} | {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
    srv = Server(port, workdir, allowed)
    print(f"pyxpp serve {version()} on http://127.0.0.1:{port}  (Python {platform.python_version()}, {sys.prefix})")
    print(f"work folder: {workdir}")
    print(f"editor: {EDITOR}   (stop: Ctrl+C, or the editor's Stop server)")
    if idle_exit:
        print(f"stops by itself {idle_exit} s after the last editor tab closes (--idle-exit 0: never)")
        threading.Thread(target=_idle_watch, args=(srv, idle_exit), daemon=True).start()
    sys.stdout.flush()
    if open_browser:
        import webbrowser
        threading.Timer(0.5, webbrowser.open, args=(EDITOR,)).start()
    try:
        srv.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop_all()
        srv.server_close()
    if getattr(srv, "restart", False):                # after an update: the new version takes over
        args = [sys.executable, "-m", "pyxpp", "serve", "--port", str(port), "--workdir", str(srv.workdir),
                "--idle-exit", str(idle_exit)]
        for o in sorted(allowed - ORIGINS - {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}):
            args += ["--allow-origin", o]
        _spawn_detached(args)
        print("restarting with the updated pyauto-xpp")
    return 0
