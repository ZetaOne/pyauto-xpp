#!/usr/bin/env python3
"""
Build pyxpp/_libxpp.{so,dylib} from the vendored XPPAUT sources.

Steps
  1. copy csrc/xppaut/* to build/src
  2. apply source patches (PATCHES below): rename XPP's interactive X11
     dialog primitives to gui_<name> so headless.c's versions are used,
     stop do_main() before it enters the X event loop, rename the CLI main(),
     and give AUTO a per-model scratch directory instead of $HOME
  3. compile every .c with the vendored Xlib *headers* (csrc/include) and
     -include csrc/force.h, which turns every exit() into pyxpp_exit()
  4. link with csrc/nox11.c (no-op Xlib) and csrc/headless.c.  There is no
     -lX11: the result has no X11 runtime dependency at all.

Windows: build with MinGW-w64 gcc, natively (sys.platform == "win32", gcc on
PATH or CC=gcc) or cross-compiled from Linux (--cc x86_64-w64-mingw32-gcc, or
--target windows).  The result is pyxpp/_libxpp.dll; csrc/win holds the few
POSIX headers MinGW lacks and is only used for that build.

Usage:  python build_lib.py [--jobs N] [--debug] [--cc CC] [--target windows]
"""
import argparse
import concurrent.futures as cf
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"
UPSTREAM = CSRC / "xppaut"
BUILD = ROOT / "build"
SRC = BUILD / "src"
OBJ = BUILD / "obj"

# (file, regex on a whole line, replacement).  Each must match exactly once.
PATCHES = [
    # dialog primitives -> gui_* (headless.c supplies the real ones)
    ("ggets.c", r"^int new_string\(", "int gui_new_string("),
    ("ggets.c", r"^int new_float\(", "int gui_new_float("),
    ("ggets.c", r"^ int new_int\(", " int gui_new_int("),
    ("ggets.c", r"^int get_mouse_xy\(", "int gui_get_mouse_xy("),
    ("pop_list.c", r"^int pop_up_list\(", "int gui_pop_up_list("),
    ("pop_list.c", r"^int two_choice\(", "int gui_two_choice("),
    ("pop_list.c", r"^int do_string_box\(", "int gui_do_string_box("),
    ("pop_list.c", r"^void respond_box\(", "void gui_respond_box("),
    ("choice_box.c", r"^int do_choice_box\(", "int gui_do_choice_box("),
    ("dialog_box.c", r"^int get_dialog\(", "int gui_get_dialog("),
    ("rubber.c", r"^int rubber\(", "int gui_rubber("),
    ("auto_x11.c", r"^void traverse_diagram\(\)", "void gui_traverse_diagram()"),
    ("init_conds.c", r"^int file_selector\(", "int gui_file_selector("),
    ("edit_rhs.c", r"^int do_edit_box\(", "int gui_do_edit_box("),
    ("calc.c", r"^void q_calc\(\)", "void gui_q_calc()"),
    ("torus.c", r"^void choose_torus\(\)", "void gui_choose_torus()"),
    # upstream bug: empty column/variable/parameter list in "fIt data"
    # dereferences a NULL strtok() result
    ("do_fit.c", r"if\(item\[0\]==0L?\)return;", "if(item==NULL||item[0]==0)return;"),
    # CLI entry point is not part of the library
    ("my_rhs.c", r"^int main\(argc,argv\)", "int xppaut_cli_main(argc,argv)"),
    # XPP bug: twod_hist() caps the 2D histogram at MAXSTOR-1 cells but two_d_hist()
    # fills nbins1*nbins2 of them (default 100x100 = 10000 > MAXSTOR 5000): a heap
    # overflow.  Refuse that case with XPP's own message instead.
    ("histogram.c", r"^   hist_len=length;$",
     '   if(hist_inf.nbins*hist_inf.nbins2>length){err_msg("Too many bins: nbins1*nbins2 must be below MAXSTOR");'
     'return(-1);}\n   hist_len=length;'),
    # XPP bug: @ jac_eps and @ newt_tol in an .ode file set NEWT_ERR / EVEC_ERR but clear
    # only their own flags, so the defaults (1e-3) overwrite them afterwards
    ("load_eqn.c", r"^     \tNEWT_ERR=atof\(s2\);$", "     \tNEWT_ERR=atof(s2); notAlreadySet.NEWT_ERR=0;"),
    ("load_eqn.c", r"^     \tEVEC_ERR=atof\(s2\);$", "     \tEVEC_ERR=atof(s2); notAlreadySet.EVEC_ERR=0;"),
    # keep a log file chosen by Python instead of forcing stdout
    ("main.c", r"^  logfile=stdout;", "  if(logfile==NULL) logfile=stdout;"),
    # stop after initialisation instead of entering the X event loop
    ("main.c", r"^do_events\(min_wid,min_hgt\);", "/* pyxpp: event loop removed */"),
    # AUTO never closes fort.8 (the fclose in close_auto() is commented out) and then
    # renames it to <file>.s.  Linux allows renaming an open file and just leaks the
    # descriptor; Windows refuses (sharing violation), the .s file is never written
    # and the next run from a labelled point reads a NULL FILE*.  Close it first.
    ("auto_nox.c", r"^  /\*    if\(fp8_is_open\)\{$",
     "  if(fp8_is_open&&fp8!=NULL){fclose(fp8);fp8=NULL;}  /* pyxpp */\n  /*    if(fp8_is_open){"),
    # per-model AUTO scratch directory
    ("auto_nox.c", r'char\* HOME = getenv\("HOME"\);', 'char* HOME = pyxpp_auto_home();'),
]
# auto_nox.c needs a prototype for pyxpp_auto_home
PROTOTYPES = {"auto_nox.c": "char *pyxpp_auto_home(void);\n"}

DEFINES = [
    "-DNOERRNO", "-DNON_UNIX_STDIO", "-DAUTO", "-DCVODE_YES", "-DHAVEDLL",
    "-DMYSTR1=8.2", "-DMYSTR2=2",
]


def apply_patches():
    if SRC.exists():
        shutil.rmtree(SRC)
    shutil.copytree(UPSTREAM, SRC)
    for fname, pat, rep in PATCHES:
        p = SRC / fname
        text = p.read_text(errors="replace")
        rx = re.compile(pat, re.M)
        hits = rx.findall(text)
        # the HOME replacement legitimately matches twice in auto_nox.c
        if len(hits) == 0 or (len(hits) > 1 and "HOME" not in pat and "item" not in pat):
            sys.exit(f"patch failed: {fname}: {pat!r} matched {len(hits)} times")
        text = rx.sub(rep.replace("\\", "\\\\"), text)
        p.write_text(text)
    for fname, proto in PROTOTYPES.items():
        p = SRC / fname
        text = p.read_text(errors="replace")
        # after the last #include
        idx = text.rfind("#include")
        idx = text.find("\n", idx) + 1
        p.write_text(text[:idx] + proto + text[idx:])
    (SRC / "nox11.c").write_text((CSRC / "nox11.c").read_text())
    (SRC / "headless.c").write_text((CSRC / "headless.c").read_text())


def compile_one(cc, cflags, c):
    o = OBJ / (c.stem + ".o")
    cmd = [cc, *cflags, "-c", str(c), "-o", str(o)]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        return c.name, r.stderr
    return c.name, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 4))   # at most 16: no bursts on big machines
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--cc", default=os.environ.get("CC"))
    ap.add_argument("--target", choices=["native", "windows"], default="native")
    a = ap.parse_args()
    windows = (sys.platform == "win32" or a.target == "windows"
               or "mingw" in os.path.basename(a.cc or ""))
    if a.cc is None:
        if sys.platform == "win32":
            a.cc = "gcc"
        elif windows:
            a.cc = "x86_64-w64-mingw32-gcc"
        else:
            a.cc = "cc"

    apply_patches()
    OBJ.mkdir(parents=True, exist_ok=True)
    for o in OBJ.glob("*.o"):
        o.unlink()

    opt = ["-O0", "-g"] if a.debug else ["-O2", "-g"]
    # macOS cross builds (cibuildwheel sets ARCHFLAGS, e.g. "-arch x86_64")
    arch = os.environ.get("ARCHFLAGS", "").split() if sys.platform == "darwin" else []
    opt += arch
    cflags = [
        *opt, *([] if windows else ["-fPIC"]), "-fcommon", "-std=gnu99", "-w",
        # Windows (LLP64): an undeclared function returning a pointer is truncated to 32 bits
        *(["-Werror=implicit-function-declaration", "-Werror=int-conversion"] if windows else []),
        # old K&R code relies on implicit declarations
        "-Wno-implicit-function-declaration", "-Wno-int-conversion",
        "-Wno-incompatible-pointer-types", "-Wno-return-mismatch",
        "-fvisibility=default", "-U_FORTIFY_SOURCE", "-D_FORTIFY_SOURCE=0",
        f"-I{SRC}", f"-I{CSRC / 'include'}", "-include", str(CSRC / "force.h"),
        *DEFINES,
    ]
    if windows:
        # MinGW lacks <dlfcn.h>, <sys/wait.h>, <sys/resource.h>; HOME -> USERPROFILE
        # (csrc/win goes first: its X11/Xos.h replaces the vendored one's WIN32 branch)
        i = cflags.index(f"-I{CSRC / 'include'}")
        cflags[i:i] = [f"-I{CSRC / 'win'}"]
        cflags += ["-include", str(CSRC / "win" / "wincompat.h")]
    # some compilers do not know every -Wno flag; drop the unknown ones
    probe = subprocess.run([a.cc, "-Werror", "-Wno-return-mismatch", "-x", "c", "-c",
                            "-o", os.devnull, "-"], input="int x;", text=True,
                           capture_output=True)
    if probe.returncode != 0:
        cflags.remove("-Wno-return-mismatch")

    sources = sorted(SRC.glob("*.c"))
    errors = []
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for name, err in ex.map(lambda c: compile_one(a.cc, cflags, c), sources):
            if err:
                errors.append((name, err))
    if errors:
        for n, e in errors:
            print(f"--- {n}\n{e}", file=sys.stderr)
        sys.exit(f"{len(errors)} files failed to compile")

    libs = ["-lm", "-ldl"]
    if windows:
        out = ROOT / "pyxpp" / "_libxpp.dll"
        # no MinGW runtime DLLs needed next to the library (libgcc, winpthread)
        ldflags = ["-shared", "-Wl,--no-undefined", "-static-libgcc", "-static"]
        libs = ["-lm", "-lwinpthread"]     # clock_gettime()
    elif sys.platform == "darwin":
        out = ROOT / "pyxpp" / "_libxpp.dylib"
        ldflags = ["-dynamiclib", "-undefined", "error", *arch]
    else:
        out = ROOT / "pyxpp" / "_libxpp.so"
        ldflags = ["-shared", "-Wl,--no-undefined"]
    objs = sorted(str(o) for o in OBJ.glob("*.o"))
    tmp = out.with_name(out.name + ".tmp")
    cmd = [a.cc, *ldflags, "-o", str(tmp), *objs, *libs]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        sys.exit("link failed")
    os.replace(tmp, out)      # in one step: a process loading the library never sees half a file
    print(f"built {out} ({out.stat().st_size // 1024} kB, {len(objs)} objects)")


if __name__ == "__main__":
    main()
