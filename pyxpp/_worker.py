"""
The process that runs the editor's code for `pyxpp serve` (one per session).

Protocol, one JSON object per line:
  stdin   {"id": "...", "code": "..."}            run this code
  stdout  ordinary text: what the code (and XPP's C code) printed
          \x00PYXPP-FIG <id> <base64 png>         a matplotlib figure
          \x00PYXPP-DONE <id> <ok 0|1>            the code finished
The code of one session runs in one namespace, so variables persist from one
run to the next, as in a notebook.  The server kills the process to stop a run.
"""
import base64
import io
import json
import os
import sys
import traceback

MARK = "\x00PYXPP-"


def main():
    os.environ.setdefault("MPLBACKEND", "Agg")
    # line-buffered text on fd 1 so prints arrive while the code runs
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr = sys.stdout
    ns = {"__name__": "__main__"}
    current = {"id": None}

    def send_figures():
        if "matplotlib.pyplot" not in sys.modules:
            return
        plt = sys.modules["matplotlib.pyplot"]
        for num in plt.get_fignums():
            fig = plt.figure(num)
            buf = io.BytesIO()
            try:
                fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            except Exception as e:                       # a broken figure must not stop the run
                print(f"(figure {num} could not be drawn: {e})")
                continue
            data = base64.b64encode(buf.getvalue()).decode()
            sys.stdout.write(f"\n{MARK}FIG {current['id']} {data}\n")
        plt.close("all")
        sys.stdout.flush()

    def show(*args, **kwargs):                           # plt.show(): send the figures now
        send_figures()

    def flush_c():
        try:                                             # XPP's own printf output
            import ctypes, ctypes.util
            lib = ctypes.util.find_library("c")
            if lib:
                ctypes.CDLL(lib).fflush(None)
        except Exception:
            pass

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        current["id"] = req.get("id", "?")
        ok = 1
        try:
            if "matplotlib.pyplot" not in sys.modules:
                try:
                    import matplotlib.pyplot as _plt
                    _plt.show = show
                except ImportError:
                    pass
            else:
                sys.modules["matplotlib.pyplot"].show = show
            code = compile(req.get("code", ""), req.get("file", "<editor>"), "exec")
            exec(code, ns)
        except SystemExit:
            pass
        except BaseException:
            ok = 0
            etype, value, tb = sys.exc_info()
            # drop this file's own frame from the traceback
            traceback.print_exception(etype, value, tb.tb_next if tb else None)
        try:
            send_figures()
        except Exception:
            pass
        sys.stdout.flush()
        flush_c()
        sys.stdout.write(f"\n{MARK}DONE {current['id']} {ok}\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
