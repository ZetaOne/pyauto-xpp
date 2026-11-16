/* wincompat.h -- force-included in every translation unit of the Windows
 * (MinGW-w64) build only, so csrc/xppaut stays upstream code.
 *
 *  - getenv("HOME") is unset on Windows; XPP builds paths like
 *    "$HOME/.xpprc" from it.  pyxpp_getenv() (headless.c) falls back to
 *    USERPROFILE.
 *  - XPP defines its own void strupr()/strlwr(); MinGW's <string.h> declares
 *    char *strupr()/strlwr(), so XPP's are renamed.
 *  - stdout/stderr: on Linux/macOS Python captures XPP's terminal output by
 *    dup2() on file descriptors 1 and 2.  On Windows the DLL's C runtime
 *    (msvcrt.dll with MinGW) is not Python's (UCRT): each has its own
 *    descriptor table, and dup2() on the DLL side would close the console
 *    handle Python itself writes to.  So the streams are redirected instead:
 *    stdout/stderr name pyxpp_out(), which is the real stream normally and
 *    the model's log file during a captured call (headless.c), and printf()
 *    and friends are routed through them.
 *  - fopen() goes through pyxpp_fopen() (headless.c): files XPP writes are
 *    binary (LF line ends, as on Linux), and so are the files it reads back
 *    from the AUTO scratch folder.  AUTO's findlb() steps backwards through
 *    fort.3 with fseek(), which text mode on Windows (CRLF, and msvcrt's
 *    ftell() on LF-only files) gets wrong.  User files (.ode, .set, tables)
 *    are still read in text mode, so CRLF files from Windows editors work.
 *  - XPP writes paths with '/', which the Windows C runtime accepts.
 *
 * Deliberately does not include <stdlib.h>: some XPP files declare malloc()
 * and free() K&R style before including it.
 */
#ifndef PYXPP_WINCOMPAT_H
#define PYXPP_WINCOMPAT_H
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

char *pyxpp_getenv(const char *name);
#define getenv pyxpp_getenv
#define strupr pyxpp_strupr
#define strlwr pyxpp_strlwr
FILE *pyxpp_fopen(const char *path, const char *mode);
#define fopen pyxpp_fopen

/* the C runtime's own streams, taken before stdout/stderr are redefined */
static __inline FILE *pyxpp_crt_stream(int which) { return which == 2 ? stderr : stdout; }
FILE *pyxpp_out(int which);
#undef stdout
#undef stderr
#define stdout pyxpp_out(1)
#define stderr pyxpp_out(2)
#undef printf
#undef vprintf
#undef putchar
#undef puts
#define printf(...) fprintf(stdout, __VA_ARGS__)
#define vprintf(f, a) vfprintf(stdout, (f), (a))
#define putchar(c) fputc((c), stdout)
#define puts(s) fprintf(stdout, "%s\n", (s))
#endif
