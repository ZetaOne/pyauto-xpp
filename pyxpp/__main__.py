"""
pyxpp on the command line.

    pyxpp serve [--port 8790] [--workdir DIR] [--open] [--allow-origin URL] [--idle-exit 60]
                          run pyauto-xpp for the editor on the project page
    pyxpp start           start `pyxpp serve` in the background (what pyxpp://start does)
    pyxpp stop            stop it
    pyxpp status          is it running, which version
    pyxpp register        make pyxpp:// links on the project page start it (this user only)
    pyxpp unregister      remove that link
    pyxpp uninstall       stop, remove the link and the installer's environment (~/.pyxpp)
    pyxpp version
(also: python -m pyxpp ...)
"""
import argparse
import shutil
import sys


def main(argv=None):
    from . import serve as S
    ap = argparse.ArgumentParser(prog="pyxpp", description="pyauto-xpp on this computer, for the editor at " + S.EDITOR)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("serve", help="run the local server for the editor")
    p.add_argument("--port", type=int, default=S.PORT)
    p.add_argument("--workdir", default=None, help="folder for your files (default: the last one, else ~/pyxpp-work)")
    p.add_argument("--open", action="store_true", help="open the editor in the browser")
    p.add_argument("--allow-origin", action="append", default=[], help="another web origin allowed to use it")
    p.add_argument("--idle-exit", type=int, default=S.IDLE_EXIT,
                   help="stop this many seconds after the last editor tab closes (0: never; default %(default)s)")
    sub.add_parser("start", help="start the server in the background")
    sub.add_parser("stop", help="stop the server")
    sub.add_parser("status", help="is the server running")
    sub.add_parser("register", help="register pyxpp:// links")
    sub.add_parser("unregister", help="remove pyxpp:// links")
    u = sub.add_parser("uninstall", help="remove the installer's environment and the link")
    u.add_argument("--yes", action="store_true")
    o = sub.add_parser("open-url", help=argparse.SUPPRESS)
    o.add_argument("url")
    sub.add_parser("version")
    a = ap.parse_args(argv)

    if a.cmd == "serve":
        return S.serve(a.port, a.workdir, a.allow_origin, a.open, a.idle_exit)
    if a.cmd == "start":
        ok = S.start_background()
        print("pyxpp serve is running" if ok else f"pyxpp serve did not start; see {S.HOME / 'serve.log'}")
        return 0 if ok else 1
    if a.cmd == "stop":
        S.open_url("pyxpp://stop")
        print("stopped")
        return 0
    if a.cmd == "status":
        d = S.ping()
        print(f"running: pyauto-xpp {d['version']} on port {S.PORT}" if d else "not running")
        return 0 if d else 1
    if a.cmd == "register":
        print(S.register())
        return 0
    if a.cmd == "unregister":
        print(S.unregister())
        return 0
    if a.cmd == "uninstall":
        if not a.yes:
            ans = input(f"Remove the pyxpp:// link and {S.HOME} (pyauto-xpp's own environment)? [y/N] ")
            if ans.strip().lower() not in ("y", "yes"):
                return 1
        S.open_url("pyxpp://stop")
        print(S.unregister())
        if S.HOME.exists():
            if sys.platform == "win32":
                import subprocess
                subprocess.Popen(f'cmd /c "timeout /t 3 >nul & rmdir /s /q "{S.HOME}""', shell=True,
                                 creationflags=subprocess.DETACHED_PROCESS)
            else:
                shutil.rmtree(S.HOME, ignore_errors=True)
            print(f"removed {S.HOME}; your files in {S.WORKDIR} are kept")
        return 0
    if a.cmd == "open-url":
        S.open_url(a.url)
        return 0
    if a.cmd == "version":
        print(S.version())
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
