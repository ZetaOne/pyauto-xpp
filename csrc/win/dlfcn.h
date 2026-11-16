/* dlfcn.h for Windows (pyxpp build): XPP's "DLL" user functions via LoadLibrary */
#ifndef PYXPP_WIN_DLFCN_H
#define PYXPP_WIN_DLFCN_H
/* declared by hand: <windows.h> would clash with XPP and Xlib names (Window, Status, ...) */
__declspec(dllimport) void *__stdcall LoadLibraryA(const char *);
__declspec(dllimport) void *__stdcall GetProcAddress(void *, const char *);
__declspec(dllimport) int __stdcall FreeLibrary(void *);
#define RTLD_LAZY 1
#define RTLD_NOW 2
#define RTLD_LOCAL 0
static const char *pyxpp_dl_err = NULL;
static void *dlopen(const char *f, int mode) {
  void *h = LoadLibraryA(f);
  (void)mode;
  pyxpp_dl_err = h ? NULL : "LoadLibrary failed";
  return h;
}
static void *dlsym(void *h, const char *s) {
  void *p = GetProcAddress(h, s);
  pyxpp_dl_err = p ? NULL : "GetProcAddress failed";
  return p;
}
static int dlclose(void *h) { return FreeLibrary(h) ? 0 : -1; }
static char *dlerror(void) { const char *e = pyxpp_dl_err; pyxpp_dl_err = NULL; return (char *)e; }
#endif
