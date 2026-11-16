/* Xos.h for the pyxpp Windows build.  The real one (csrc/include/X11/Xos.h)
 * has a WIN32 branch that needs Xw32defs.h and redefines gettimeofday() with
 * one argument; XPP only wants the usual C headers from it. */
#ifndef _XOS_H_
#define _XOS_H_
#include <X11/Xosdefs.h>
#include <sys/types.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#ifndef index
#define index(s,c) (strchr((s),(c)))
#endif
#ifndef rindex
#define rindex(s,c) (strrchr((s),(c)))
#endif
#endif
