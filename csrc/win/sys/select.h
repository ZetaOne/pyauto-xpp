/* sys/select.h for Windows (pyxpp build): XPP 8.4 uses select() only to wait for
 * events on the X display connection (pop_list.c, xpp_script.c), which a headless
 * library never does.  A stand-in that reports "nothing to read". */
#ifndef PYXPP_WIN_SYS_SELECT_H
#define PYXPP_WIN_SYS_SELECT_H
#include <sys/time.h>          /* struct timeval */
typedef struct { unsigned long bits[2]; } pyxpp_fd_set;
#define fd_set pyxpp_fd_set
#ifndef FD_ZERO
#define FD_ZERO(s) ((s)->bits[0] = (s)->bits[1] = 0)
#define FD_SET(fd, s) ((void)(fd), (void)(s))
#define FD_CLR(fd, s) ((void)(fd), (void)(s))
#define FD_ISSET(fd, s) ((void)(fd), (void)(s), 0)
#endif
static int pyxpp_select(int n, fd_set *r, fd_set *w, fd_set *e, struct timeval *t) {
  (void)n; (void)r; (void)w; (void)e; (void)t; return 0; }
#define select pyxpp_select
#endif
