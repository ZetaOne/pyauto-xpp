/* sys/wait.h for Windows (pyxpp build): XPP only uses fork/wait to start an
 * editor or a browser, which a headless library never does. */
#ifndef PYXPP_WIN_SYS_WAIT_H
#define PYXPP_WIN_SYS_WAIT_H
#include <sys/types.h>
#include <process.h>
static int pyxpp_fork(void) { return -1; }
static int pyxpp_wait(int *st) { if (st) *st = 0; return -1; }
#define fork() pyxpp_fork()
#define wait(s) pyxpp_wait(s)
#endif
