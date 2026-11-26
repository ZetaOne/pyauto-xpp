"""
The process that runs the editor's code for `pyxpp serve` (one per session).

Protocol, one JSON object per line:
  stdin   {"id": "...", "code": "..."}            run this code
  stdout  ordinary text: what the code (and XPP's C code) printed
          \x00PYXPP-FIG <id> <base64 png>         a matplotlib figure
          \x00PYXPP-PLOTLY <id> <base64 json>     a plotly figure (fig.show(), or a figure/widget shown)
          \x00PYXPP-HTML <id> <base64 html>       the text of an ipywidgets widget (HTML, Label)
          \x00PYXPP-LINE <id> <n>                 the line of the editor's file running now
          \x00PYXPP-ERR <id> <n>                  the line of that file where an error came from
          \x00PYXPP-DONE <id> <ok 0|1>            the code finished
A request may carry "line0": the code is then a piece of the file starting after that
many lines (a cell or a selection), so that line numbers are those of the file.
The code of one session runs in one namespace, so variables persist from one
run to the next, as in a notebook.  The server kills the process to stop a run.
"""
import ast
import base64
import io
import json
import os
import re
import sys
import threading
import time
import traceback

MARK = "\x00PYXPP-"


def b64(text):
    return base64.b64encode(text.encode("utf-8")).decode()


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

    # ---- plotly figures and ipywidgets (the interactive tutorials): shown, without live controls
    told = {"widgets": None}

    def emit_plotly(fig):
        try:
            js = fig.to_json()
        except Exception as e:
            print(f"(a plotly figure could not be sent: {e})")
            return
        sys.stdout.write(f"\n{MARK}PLOTLY {current['id']} {b64(js)}\n")
        sys.stdout.flush()

    def show_value(v, *more, **kw):
        """what Jupyter's display() does, as far as the editor can show it"""
        for x in (v,) + more:
            mod = type(x).__module__ or ""
            if hasattr(x, "to_plotly_json") and hasattr(x, "to_json") and mod.startswith("plotly"):
                emit_plotly(x)
            elif mod.startswith(("ipywidgets", "xpptut", "anywidget")) or hasattr(x, "children") and hasattr(x, "keys"):
                walk_widget(x)
                if told["widgets"] != current["id"]:
                    told["widgets"] = current["id"]
                    print("(shown without its sliders and buttons: the editor runs code, it does not keep widgets "
                          "alive. Change values in code, e.g. ex.set(i=0.5), and show it again.)")
            elif type(x).__name__ == "Figure" and mod.startswith("matplotlib"):
                send_figures()
            elif x is not None:
                print(repr(x))

    def walk_widget(w, depth=0):
        if depth > 40:
            return
        if hasattr(w, "to_plotly_json") and hasattr(w, "to_json"):
            emit_plotly(w)
            return
        cls = type(w).__name__
        val = getattr(w, "value", None)
        if cls in ("HTML", "HTMLMath", "Label") and isinstance(val, str) and val.strip():
            sys.stdout.write(f"\n{MARK}HTML {current['id']} {b64(val)}\n")
            sys.stdout.flush()
        for c in (getattr(w, "children", None) or ()):
            walk_widget(c, depth + 1)

    # plotly's fig.show() and IPython's display(), installed once if those packages are there
    import builtins
    builtins.display = show_value
    try:
        import importlib.util as _iu
        if _iu.find_spec("plotly"):
            import plotly.basedatatypes as _bd
            import plotly.io as _pio
            _bd.BaseFigure.show = lambda self, *a, **k: emit_plotly(self)
            _pio.show = lambda fig, *a, **k: emit_plotly(fig)
        if _iu.find_spec("IPython"):
            import IPython.display as _ipd
            _ipd.display = show_value
    except Exception:
        pass

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
        target = req.get("file", "<editor>")
        ok = 1
        # the running line: the tracer only notes it (cheap); a thread reports it 10 times a second,
        # also while XPP's C code runs (ctypes lets the thread run then)
        cur, sent, running = [0], [0], [True]

        def local(frame, event, arg):
            if event == "line":
                cur[0] = frame.f_lineno
            return local

        def glob(frame, event, arg):
            return local if frame.f_code.co_filename == target else None

        def reporter(jid=current["id"]):
            while running[0]:
                if cur[0] and cur[0] != sent[0]:
                    sent[0] = cur[0]
                    try:
                        sys.stdout.write(f"\n{MARK}LINE {jid} {cur[0]}\n")
                        sys.stdout.flush()
                    except Exception:
                        pass
                time.sleep(0.1)

        rep_thread = threading.Thread(target=reporter, daemon=True)
        rep_thread.start()
        try:
            if "matplotlib.pyplot" not in sys.modules:
                try:
                    import matplotlib.pyplot as _plt
                    _plt.show = show
                except ImportError:
                    pass
            else:
                sys.modules["matplotlib.pyplot"].show = show
            src = "\n" * int(req.get("line0", 0)) + req.get("code", "")
            last = None
            if req.get("mode") == "cell":
                # like a notebook cell or the console: IPython's %magic and !shell lines do nothing
                # (e.g. a Colab "%pip install" cell), and the value of a last expression is shown
                src = re.sub(r"(?m)^([ \t]*)([%!].*)$", lambda m: m.group(1) + "pass  # " + m.group(2), src)
                tree = ast.parse(src, target, "exec")
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last = compile(ast.Expression(tree.body.pop().value), target, "eval")
                code = compile(tree, target, "exec")
            else:
                code = compile(src, target, "exec")
            sys.settrace(glob)
            try:
                exec(code, ns)
                if last is not None:
                    value = eval(last, ns)
                    if value is not None:
                        ns["_"] = value
                        show_value(value)
            finally:
                sys.settrace(None)
        except SystemExit:
            pass
        except BaseException:
            ok = 0
            etype, value, tb = sys.exc_info()
            # drop this file's own frame from the traceback
            traceback.print_exception(etype, value, tb.tb_next if tb else None)
            err_line = getattr(value, "lineno", None) if isinstance(value, SyntaxError) else None
            t = tb
            while t is not None:                         # the last frame in the editor's file
                if t.tb_frame.f_code.co_filename == target:
                    err_line = t.tb_lineno
                t = t.tb_next
            if err_line:
                sys.stdout.write(f"\n{MARK}ERR {current['id']} {err_line}\n")
        running[0] = False
        rep_thread.join(0.5)
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
