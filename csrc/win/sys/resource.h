/* sys/resource.h for Windows (pyxpp build): only getrusage(RUSAGE_SELF) for timing */
#ifndef PYXPP_WIN_SYS_RESOURCE_H
#define PYXPP_WIN_SYS_RESOURCE_H
#include <sys/time.h>
#include <string.h>
#define RUSAGE_SELF 0
struct rusage { struct timeval ru_utime, ru_stime; };
static int getrusage(int who, struct rusage *r) {
  (void)who; memset(r, 0, sizeof *r); gettimeofday(&r->ru_utime, NULL); return 0;
}
#endif
