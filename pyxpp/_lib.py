"""
Low-level access to the headless XPPAUT library.

Each Model gets its *own copy* of the shared library (copied to a temporary
file and dlopen'ed with RTLD_LOCAL).  XPP keeps all of its state in C
globals, so a private copy is what lets several models live in one Python
process without trampling each other.

On Windows the library is _libxpp.dll (MinGW-w64).  A loaded DLL cannot be
deleted, so each copy lives in a per-process folder under %TEMP% and is
unloaded (FreeLibrary) and removed when the model is closed or freed; copies
left behind by a crashed process are swept up the next time pyxpp starts.
"""
import contextlib
import numbers
import ctypes
import ctypes.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

_WINDOWS = sys.platform == "win32"
_HERE = Path(__file__).resolve().parent
_LIBNAME = ("_libxpp.dll" if _WINDOWS else
            "_libxpp.dylib" if sys.platform == "darwin" else "_libxpp.so")
_LIBPATH = _HERE / _LIBNAME

if _WINDOWS:
    # the DLL may use another C runtime than Python (msvcrt vs UCRT): stdout capture
    # and flushing are done inside the DLL (pyxpp_capture_begin/end)
    _libc = None
else:
    _libc = ctypes.CDLL(ctypes.util.find_library("c"))
    _libc.fflush.argtypes = [ctypes.c_void_p]

c_int, c_double, c_char_p = ctypes.c_int, ctypes.c_double, ctypes.c_char_p
P_int = ctypes.POINTER(c_int)
P_double = ctypes.POINTER(c_double)

_PROTOS = {
    "pyxpp_error": (c_char_p, []),
    "pyxpp_init": (c_int, [c_int, ctypes.POINTER(c_char_p), c_char_p]),
    "pyxpp_set_log": (c_int, [c_char_p]),
    "pyxpp_flush_log": (None, []),
    "pyxpp_tape_clear": (None, []),
    "pyxpp_tape_push": (c_int, [c_char_p]),
    "pyxpp_tape_len": (c_int, []),
    "pyxpp_transcript": (c_char_p, []),
    "pyxpp_transcript_clear": (None, []),
    "pyxpp_key": (c_int, [c_int]),
    "pyxpp_command": (c_int, [c_int]),
    "pyxpp_num_key": (c_int, [c_int]),
    "pyxpp_menu_state": (c_int, []),
    "pyxpp_main_menu": (c_int, []),
    "pyxpp_auto_key": (c_int, [c_int]),
    "pyxpp_auto_open": (c_int, []),
    "pyxpp_set_option": (c_int, [c_char_p, c_char_p]),
    "pyxpp_get_val": (c_int, [c_char_p, P_double]),
    "pyxpp_set_val": (c_int, [c_char_p, c_double]),
    "pyxpp_equilibrium": (c_int, [P_double, P_double, P_double, P_int]),
    "pyxpp_node": (c_int, []),
    "pyxpp_neq": (c_int, []),
    "pyxpp_nmarkov": (c_int, []),
    "pyxpp_npar": (c_int, []),
    "pyxpp_var_name": (c_char_p, [c_int]),
    "pyxpp_par_name": (c_char_p, [c_int]),
    "pyxpp_data_rows": (c_int, []),
    "pyxpp_data_cols": (c_int, []),
    "pyxpp_data_copy": (None, [P_double]),
    "pyxpp_diag_count": (c_int, []),
    "pyxpp_diag_get": (c_int, [c_int, P_int, P_double, P_double]),
    "pyxpp_diag_l1": (c_int, [c_int, P_double, P_double, P_int]),
    "pyxpp_hopf_l1": (c_int, [P_double, P_double, P_double, P_double, P_int]),
    "pyxpp_delay_string": (c_char_p, [c_int]),
    "pyxpp_set_delay_string": (c_int, [c_int, c_char_p]),
    "pyxpp_calc": (c_int, [c_char_p, P_double]),
    "pyxpp_delay_roots": (c_int, [P_double, c_int, P_double, P_int]),
    "pyxpp_rhs": (c_int, [c_double, P_double, P_double]),
    "pyxpp_dirfield": (c_int, [c_int, c_int, c_char_p]),
    "pyxpp_auto_par_index": (c_int, [c_int]),
    "pyxpp_n_auto_par": (c_int, []),
    "pyxpp_auto_settings": (None, [P_int, P_double]),
    "pyxpp_auto_par_name": (c_char_p, [c_int]),
    "pyxpp_set_auto_dir": (None, [c_char_p]),
    "pyxpp_real_to_pixel": (None, [c_double, c_double, P_int, P_int]),
    "pyxpp_load_storage": (c_int, [c_int, c_int, P_double]),
    "pyxpp_auto_solution": (c_int, [c_int, c_int, c_double, c_int, P_int, P_int, P_double, P_double]),
    "pyxpp_last_state": (c_int, [P_double, P_double]),
    "pyxpp_shutdown": (None, []),
}
if _WINDOWS:
    _PROTOS["pyxpp_capture_begin"] = (c_int, [c_char_p])
    _PROTOS["pyxpp_capture_end"] = (None, [])


class XPPError(RuntimeError):
    """An XPP command failed, needed input that was not supplied, or tried
    to do something that has no headless equivalent."""


def library_path() -> Path:
    if not _LIBPATH.exists():
        raise ImportError(
            f"{_LIBPATH} not found. Build it with:  python build_lib.py")
    return _LIBPATH


_DLLDIR = None
_DLLDIR_LOCK = None


def _dll_dir() -> Path:
    """Windows: per-process folder for the DLL copies (and a sweep of old ones)."""
    global _DLLDIR, _DLLDIR_LOCK
    if _DLLDIR is None:
        base = Path(tempfile.gettempdir()) / "pyxpp-dll"
        base.mkdir(exist_ok=True)
        for old in base.glob("p*"):
            # left by an earlier process.  A live process keeps its "lock" file open,
            # which Windows will not delete: then the folder is left alone.
            try:
                (old / "lock").unlink(missing_ok=True)
            except OSError:
                continue
            shutil.rmtree(old, ignore_errors=True)
        d = Path(tempfile.mkdtemp(prefix=f"p{os.getpid()}_", dir=base))
        _DLLDIR_LOCK = open(d / "lock", "w")
        _DLLDIR = d
    return _DLLDIR


def _release(copy, handle, c):
    """Windows: unload a model's DLL copy and delete the file."""
    if handle:
        try:
            c.pyxpp_shutdown()
        except Exception:
            pass
        import _ctypes
        try:
            _ctypes.FreeLibrary(handle)
        except OSError:
            return
    try:
        os.unlink(copy)
    except OSError:
        pass


class Lib:
    """A private instance of libxpp plus a scratch directory."""

    def __init__(self, workdir: Path):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        where = _dll_dir() if _WINDOWS else self.workdir
        self._copy = where / f"libxpp_{os.getpid()}_{id(self)}{_LIBPATH.suffix}"
        shutil.copyfile(library_path(), self._copy)
        mode = getattr(os, "RTLD_LOCAL", 0) | getattr(os, "RTLD_NOW", 2)
        self.c = ctypes.CDLL(str(self._copy), mode=mode)
        self._finalizer = None
        if os.name == "posix":
            # the loaded library stays usable; the 4.5 MB file is not needed any more
            try:
                os.unlink(self._copy)
            except OSError:
                pass
        for name, (res, args) in _PROTOS.items():
            f = getattr(self.c, name)
            f.restype = res
            f.argtypes = args
        self.logpath = self.workdir / "xpp.log"
        if _WINDOWS:
            import weakref
            self._finalizer = weakref.finalize(self, _release, str(self._copy),
                                               self.c._handle, self.c)

    # -- stdout capture -------------------------------------------------
    @contextlib.contextmanager
    def captured(self, enabled=True):
        """Send C-level stdout (XPP and AUTO print with printf) to the log
        file for the duration of a call."""
        if not enabled:
            yield
            return
        if _WINDOWS:
            sys.stdout.flush()
            sys.stderr.flush()
            self.c.pyxpp_capture_begin(str(self.logpath).encode())
            try:
                yield
            finally:
                self.c.pyxpp_capture_end()
            return
        sys.stdout.flush()
        sys.stderr.flush()
        _libc.fflush(None)
        saved, saved_err = os.dup(1), os.dup(2)
        fd = os.open(self.logpath, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.dup2(fd, 1)
            os.dup2(fd, 2)          # XPP also writes some messages (out of bounds, ...) to stderr
            yield
        finally:
            _libc.fflush(None)
            os.dup2(saved, 1)
            os.dup2(saved_err, 2)
            os.close(saved_err)
            os.close(saved)
            os.close(fd)

    def check(self, rc):
        if rc != 0:
            raise XPPError(self.c.pyxpp_error().decode(errors="replace"))
        return rc

    def tape(self, answers):
        self.c.pyxpp_tape_clear()
        for a in answers:
            if not isinstance(a, (str, bytes)):
                a = format_answer(a)
            if isinstance(a, str):
                a = a.encode()
            if self.c.pyxpp_tape_push(a) != 0:
                raise XPPError("too many answers")

    def leftover(self):
        return self.c.pyxpp_tape_len()

    def close(self):
        """Release the library copy.  On Windows this unloads the DLL, after
        which the model's C functions must not be called again."""
        if self._finalizer is not None:
            self._finalizer()
            return
        try:
            os.unlink(self._copy)
        except OSError:
            pass


def format_answer(a):
    """Turn a Python value into the text XPP would have been typed."""
    if isinstance(a, bool):
        return "y" if a else "n"
    # numbers.* covers numpy scalars; repr(np.float64) would be "np.float64(...)"
    if isinstance(a, numbers.Integral):
        return str(int(a))
    if isinstance(a, numbers.Real):
        return repr(float(a))
    if isinstance(a, dict):
        return ";".join(f"{k}={format_answer(v)}" for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return ",".join(format_answer(v) for v in a)
    return str(a)


ESC = "\x1b"


def new_workdir(prefix="pyxpp_"):
    return Path(tempfile.mkdtemp(prefix=prefix))
