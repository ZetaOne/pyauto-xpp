/*
 * headless.c -- the glue that turns XPPAUT into a library.
 *
 * XPP's GUI is a keyboard/menu machine: a key press reaches commander(),
 * which calls a menu handler, which asks the user questions through a small
 * set of dialog primitives (new_float, pop_up_list, two_choice,
 * do_string_box, file_selector, ...).  The build script renames the original
 * X11 versions of those primitives to gui_<name> and links in the versions
 * below instead.  Each one takes its answer from an "input tape" -- a FIFO of
 * strings filled from Python -- or, if the tape is empty, accepts XPP's own
 * default, exactly as if the user had pressed Enter.
 *
 * Every question asked and every answer used is appended to a transcript so
 * the Python side can show what a button asked for.
 *
 * Errors: anything that would kill or hang the process (exit(), an
 * unhandled interactive loop) longjmps back to the entry point with a
 * message, which Python raises as XPPError.
 */
#include <setjmp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>
#include <X11/Xlib.h>

#include "auto_nox.h"
#include "mykeydef.h"

#define EXPORT __attribute__((visibility("default")))

/* ------------------------------------------------------------------ */
/* XPP internals we need                                                */
/* ------------------------------------------------------------------ */
extern void do_main(int argc, char **argv);
/* used to refresh delays, kernels and tables after a change (as XPP's dialogs do) */
extern double DELAY;
extern int NKernel, MyStart, INFLAG, AutoEvaluate;
extern void free_delay(void);
extern int alloc_delay(double big);
extern void alloc_kernels(int flag);
extern void re_evaluate_kernels(void);
extern void redo_all_fun_tables(void);
extern void commander(char ch);
extern void run_the_commands(int com);
extern int do_calc(char *temp, double *z);
extern void traverse_out(DIAGRAM *d, int *ix, int *iy, int dodraw);
extern void redraw_diagram(void);
extern void evaluate_derived(void);
extern void redraw_params(void);
extern void redraw_ics(void);
extern void set_ivar(int i, double value);
extern void get_ic(int it, double *x);
extern void set_option(char *s1, char *s2, int force, void *mask);
extern void do_meth(void);
extern int get_val(char *name, double *value);
extern int set_val(char *name, double value);
extern void do_sing_info(double *x, double eps, double err, double big, int maxit,
                         int n, double *er, double *em, int *ierr);
extern void do_auto_win(void);
extern void auto_plot_par(void), auto_num_par(void), auto_grab(void), auto_run(void);
extern void draw_bif_axes(void), auto_per_par(void), auto_params(void), auto_file(void);
extern void get_num_par(char ch);
extern void help(void);

extern int help_menu;
extern int NODE, NEQ, NUPAR, NMarkov, FIX_VAR, storind, Xup, XPPBatch;
extern char uvar_names[][12];
extern char upar_names[][11];
extern float **storage;
extern double last_ic[];
extern double NEWT_ERR, EVEC_ERR, BOUND;
extern int EVEC_ITER;
extern FILE *logfile;
extern int XPPVERBOSE;
extern int RunImmediately;

extern GRABPT grabpt;
extern BIFUR Auto;
extern DIAGRAM *bifd, *CUR_DIAGRAM;
extern int NBifs, NAutoPar, Auto_index_to_array[];
extern double constants[];

/* ------------------------------------------------------------------ */
/* error handling                                                       */
/* ------------------------------------------------------------------ */
static jmp_buf *guard = NULL;
static char errbuf[4096];
static int in_api = 0;

EXPORT const char *pyxpp_error(void) { return errbuf; }

void pyxpp_fail(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(errbuf, sizeof errbuf, fmt, ap);
  va_end(ap);
  if (guard) longjmp(*guard, 1);
  fprintf(stderr, "pyxpp fatal (no guard): %s\n", errbuf);
  abort();
}

/* every exit() inside the library is redirected here by force.h, so XPP
   can never terminate the Python interpreter. */
void pyxpp_exit(int code) { pyxpp_fail("XPP tried to exit (code %d)", code); }

/* run BODY under a setjmp guard; returns 0 on success, -1 on failure */
#define GUARDED(BODY)                                  \
  do {                                                 \
    jmp_buf jb, *old = guard;                          \
    errbuf[0] = 0;                                     \
    guard = &jb;                                       \
    in_api = 1; unanswered = 0;                        \
    if (setjmp(jb) == 0) {                             \
      BODY;                                            \
      guard = old; in_api = 0;                         \
      return 0;                                        \
    }                                                  \
    guard = old; in_api = 0;                           \
    return -1;                                         \
  } while (0)

/* ------------------------------------------------------------------ */
/* input tape and transcript                                            */
/* ------------------------------------------------------------------ */
#define TAPE_MAX 4096
static char *tape[TAPE_MAX];
static int tape_head = 0, tape_tail = 0;

EXPORT void pyxpp_tape_clear(void) {
  while (tape_head < tape_tail) free(tape[tape_head++]);
  tape_head = tape_tail = 0;
}
EXPORT int pyxpp_tape_push(const char *s) {
  if (tape_tail >= TAPE_MAX) return -1;
  tape[tape_tail++] = strdup(s);
  return 0;
}
EXPORT int pyxpp_tape_len(void) { return tape_tail - tape_head; }

/* pops the next answer; returns NULL if tape empty (caller uses default).
   Caller must free. */
static char *tape_pop(void) {
  if (tape_head >= tape_tail) return NULL;
  return tape[tape_head++];
}

/* Some XPP loops keep asking until they get a particular answer (the
   tutorial, "Parameter:" loops, ...).  With an empty tape that would spin
   forever, so too many unanswered questions in one call is an error. */
static int unanswered = 0;
#define MAX_UNANSWERED 500
static void no_answer(const char *what) {
  if (++unanswered > MAX_UNANSWERED)
    pyxpp_fail("XPP asked '%s' (and others) %d times without an answer; "
               "supply answers or end the loop explicitly", what, MAX_UNANSWERED);
}

static int is_escape(const char *s) {
  return s && (strcmp(s, "\x1b") == 0 || strcasecmp(s, "<esc>") == 0);
}

static char *tr = NULL;
static size_t trlen = 0, trcap = 0;
static void tlog(const char *fmt, ...) {
  char line[2048];
  va_list ap;
  size_t n;
  va_start(ap, fmt);
  vsnprintf(line, sizeof line, fmt, ap);
  va_end(ap);
  n = strlen(line);
  if (trlen + n + 2 > trcap) {
    trcap = 2 * (trcap + n + 2);
    tr = (char *)realloc(tr, trcap);
  }
  memcpy(tr + trlen, line, n);
  trlen += n;
  tr[trlen++] = '\n';
  tr[trlen] = 0;
}
EXPORT const char *pyxpp_transcript(void) { return tr ? tr : ""; }
EXPORT void pyxpp_transcript_clear(void) { trlen = 0; if (tr) tr[0] = 0; }

static void strip(char *s) {
  char *p = s, *e;
  while (*p && isspace((unsigned char)*p)) p++;
  memmove(s, p, strlen(p) + 1);
  e = s + strlen(s);
  while (e > s && isspace((unsigned char)e[-1])) *--e = 0;
}

/* ------------------------------------------------------------------ */
/* replacements for XPP's dialog primitives                             */
/* ------------------------------------------------------------------ */

/* one-line text prompt.  1 = accepted, 0 = cancelled (matches original) */
int new_string(char *name, char *value) {
  char *a = tape_pop();
  if (a == NULL) {
    no_answer(name);
    tlog("prompt  %s [%s] -> (default)", name, value);
    return 1;
  }
  if (is_escape(a)) {
    tlog("prompt  %s [%s] -> <esc>", name, value);
    free(a);
    return 0;
  }
  tlog("prompt  %s [%s] -> %s", name, value, a);
  strcpy(value, a); /* XPP buffers are >= 80 chars; Python enforces limits */
  free(a);
  return 1;
}

int new_float(char *name, double *value) {
  char tvalue[256];
  double z;
  snprintf(tvalue, sizeof tvalue, "%.16g", *value);
  if (new_string(name, tvalue) == 0 || strlen(tvalue) == 0) return -1;
  if (tvalue[0] == '%') {
    if (do_calc(&tvalue[1], &z) != -1) *value = z;
    return 0;
  }
  *value = atof(tvalue);
  return 0;
}

int new_int(char *name, int *value) {
  char svalue[256];
  snprintf(svalue, sizeof svalue, "%d", *value);
  if (new_string(name, svalue) == 0 || strlen(svalue) == 0) return -1;
  *value = atoi(svalue);
  return 0;
}

/* match an answer against a menu: a single key char, or the entry label
   (case-insensitive, ignoring parentheses), or a unique label prefix. */
static int match_menu(const char *ans, char **list, const char *key, int n) {
  int i, hit = -1, nhit = 0;
  char lab[256], a[256];
  if (strlen(ans) == 1) {
    for (i = 0; i < n; i++)
      if (tolower((unsigned char)key[i]) == tolower((unsigned char)ans[0])) return i;
    if (ans[0] == 27) return -2;
    return -1;
  }
  snprintf(a, sizeof a, "%s", ans);
  for (i = 0; a[i]; i++) a[i] = (char)tolower((unsigned char)a[i]);
  for (i = 0; i < n; i++) {
    const char *s = list[i];
    int k = 0;
    for (; *s && k < 255; s++)
      if (*s != '(' && *s != ')') lab[k++] = (char)tolower((unsigned char)*s);
    lab[k] = 0;
    strip(lab);
    if (strcmp(lab, a) == 0) return i;
    if (strncmp(lab, a, strlen(a)) == 0) { hit = i; nhit++; }
    /* entries written "k: name" (parameter sets, tables, ...): the name alone */
    if (lab[0] && lab[1] == ':' && strcmp(lab + 2 + (lab[2] == ' '), a) == 0) return i;
  }
  return nhit == 1 ? hit : -1;
}

int pop_up_list(Window *root, char *title, char **list, char *key, int n, int max,
                int def, int x, int y, char **hints, Window hwin, char *httxt) {
  char *a = tape_pop();
  int i;
  (void)root; (void)max; (void)x; (void)y; (void)hints; (void)hwin; (void)httxt;
  if (a == NULL || is_escape(a)) {
    if (a == NULL) no_answer(title);
    tlog("menu    %s -> %s", title, a ? "<esc>" : "(cancel: no answer on tape)");
    free(a);
    return 27;
  }
  i = match_menu(a, list, key, n);
  if (i < 0) {
    char msg[2048];
    size_t len;
    int j;
    snprintf(msg, sizeof msg, "menu '%s' has no entry '%s' (entries:", title, a);
    for (j = 0; j < n; j++) {
      len = strlen(msg);
      snprintf(msg + len, sizeof msg - len, " '%s'", list[j]);
    }
    len = strlen(msg);
    snprintf(msg + len, sizeof msg - len, ")");
    free(a);
    pyxpp_fail("%s", msg);
  }
  tlog("menu    %s -> %s", title, list[i]);
  free(a);
  (void)def;
  return (int)key[i];
}

int two_choice(char *c1, char *c2, char *q, char *key, int x, int y, Window w, char *title) {
  char *a = tape_pop();
  int r;
  (void)x; (void)y; (void)w; (void)title;
  if (a == NULL) {
    /* the GUI returns the Enter key code, which callers treat as neither
       choice (in practice: "no") */
    no_answer(q);
    tlog("choice  %s (%s/%s) -> <enter> (no answer on tape)", q, c1, c2);
    return 13;
  }
  if (is_escape(a)) { r = 27; }
  else if (tolower((unsigned char)a[0]) == tolower((unsigned char)key[0]) || strcasecmp(a, c1) == 0) r = key[0];
  else if (tolower((unsigned char)a[0]) == tolower((unsigned char)key[1]) || strcasecmp(a, c2) == 0) r = key[1];
  else { char m[300]; snprintf(m, sizeof m, "choice '%s' expects %s or %s, got '%s'", q, c1, c2, a); free(a); pyxpp_fail("%s", m); r = 0; }
  tlog("choice  %s (%s/%s) -> %s", q, c1, c2, r == key[0] ? c1 : (r == 27 ? "<esc>" : c2));
  free(a);
  return r;
}

/* Apply "name=value;name=value" to a list of fields.  Field names are
   matched case-insensitively after stripping a trailing '=' and spaces. */
/* index of the dialog field called lhs (nm gets its display name), or -1 */
static int field_index(int n, char **names, const char *lhs, char *nm, size_t nmsize) {
  int i;
  for (i = 0; i < n; i++) {
    const char *src = names[i];
    char shortnm[256], *paren;
    if (src[0] == '*' && isdigit((unsigned char)src[1])) src += 2;
    snprintf(nm, nmsize, "%s", src);
    if (strlen(nm) && nm[strlen(nm) - 1] == '=') nm[strlen(nm) - 1] = 0;
    strip(nm);
    snprintf(shortnm, sizeof shortnm, "%s", nm);
    paren = strrchr(shortnm, '(');
    if (paren && paren > shortnm && shortnm[strlen(shortnm) - 1] == ')') *paren = 0;
    strip(shortnm);
    if (strlen(shortnm) && shortnm[strlen(shortnm) - 1] == ':') shortnm[strlen(shortnm) - 1] = 0;
    strip(shortnm);
    if (strcasecmp(nm, lhs) == 0 || strcasecmp(shortnm, lhs) == 0) return i;
  }
  return -1;
}

static int apply_assignments(const char *spec, const char *title, int n, char **names,
                             char *values, int stride) {
  char *buf = strdup(spec), *save = NULL, *tok;
  int i, found;
  for (tok = strtok_r(buf, ";\n", &save); tok; tok = strtok_r(NULL, ";\n", &save)) {
    char *eq = strchr(tok, '='), lhs[256], nm[256];
    if (!eq) continue;
    /* a field name may itself hold '=' (8.4's "Time(0=orbit)"): split at the first
       '=' whose left side is a field name, else at the first '=' */
    {
      char *e;
      for (e = eq; e; e = strchr(e + 1, '=')) {
        char try_lhs[256];
        size_t len = (size_t)(e - tok);
        if (len >= sizeof try_lhs) break;
        memcpy(try_lhs, tok, len); try_lhs[len] = 0;
        strip(try_lhs);
        if (field_index(n, names, try_lhs, nm, sizeof nm) >= 0) { eq = e; break; }
      }
    }
    *eq = 0;
    snprintf(lhs, sizeof lhs, "%s", tok);
    strip(lhs);
    found = 0;
    for (i = 0; i < n; i++) {
      /* XPP marks drop-down fields with a "*<digit>" prefix: skip it */
      const char *src = names[i];
      if (src[0] == '*' && isdigit((unsigned char)src[1])) src += 2;
      snprintf(nm, sizeof nm, "%s", src);
      if (strlen(nm) && nm[strlen(nm) - 1] == '=') nm[strlen(nm) - 1] = 0;
      strip(nm);
      {
        /* also accept the name without a trailing "(Y/N)"-style hint or ':' */
        char shortnm[256];
        char *paren;
        snprintf(shortnm, sizeof shortnm, "%s", nm);
        paren = strrchr(shortnm, '(');
        if (paren && paren > shortnm && shortnm[strlen(shortnm) - 1] == ')') *paren = 0;
        strip(shortnm);
        if (strlen(shortnm) && shortnm[strlen(shortnm) - 1] == ':') shortnm[strlen(shortnm) - 1] = 0;
        strip(shortnm);
        if (strcasecmp(shortnm, lhs) == 0) snprintf(nm, sizeof nm, "%s", lhs);
      }
      if (strcasecmp(nm, lhs) == 0) {
        char v[512];
        snprintf(v, sizeof v, "%s", eq + 1);
        strip(v);
        if ((int)strlen(v) >= stride) { free(buf); pyxpp_fail("value for %s too long", lhs); }
        tlog("field   %s.%s = %s (was %s)", title, nm, v, values + i * stride);
        strcpy(values + i * stride, v);
        found = 1;
        break;
      }
    }
    if (!found) {
      char m[600], all[400] = "";
      for (i = 0; i < n && strlen(all) < 350; i++) {
        const char *src = names[i];
        if (src[0] == '*' && isdigit((unsigned char)src[1])) src += 2;
        strcat(all, "'"); strcat(all, src); strcat(all, "' ");
      }
      snprintf(m, sizeof m, "dialog '%s' has no field '%s' (fields: %s)", title, lhs, all);
      free(buf);
      pyxpp_fail("%s", m);
    }
  }
  free(buf);
  return 0;
}

#define MAX_LEN_SBOX 25
/* multi-field dialog.  Returns DONE_ALL(2) on accept, FORGET_ALL(0) on esc */
int do_string_box(int n, int row, int col, char *title, char **names,
                  char values[][MAX_LEN_SBOX], int maxchar) {
  char *a = tape_pop();
  (void)row; (void)col; (void)maxchar;
  if (a == NULL) { no_answer(title); tlog("dialog  %s -> (defaults)", title); return 2; }
  if (is_escape(a)) { tlog("dialog  %s -> <esc>", title); free(a); return 0; }
  apply_assignments(a, title, n, names, &values[0][0], MAX_LEN_SBOX);
  free(a);
  return 2;
}

/* RHS / function editor */
#define MAX_LEN_EBOX 86
int do_edit_box(int n, char *title, char **names, char **values) {
  char *a = tape_pop(), *flat;
  int i;
  if (a == NULL) { tlog("editor  %s -> (unchanged)", title); return 0; }
  if (is_escape(a)) { tlog("editor  %s -> <esc>", title); free(a); return 0; }
  flat = (char *)calloc((size_t)n, MAX_LEN_EBOX);
  for (i = 0; i < n; i++) snprintf(flat + i * MAX_LEN_EBOX, MAX_LEN_EBOX, "%s", values[i]);
  apply_assignments(a, title, n, names, flat, MAX_LEN_EBOX);
  for (i = 0; i < n; i++) strcpy(values[i], flat + i * MAX_LEN_EBOX);
  free(flat);
  free(a);
  return 2;
}

/* check-box list: answer is a comma list of 0/1 or of entry names to tick */
int do_choice_box(Window root, char *wname, int n, int mcc, char **names, int *check, int type) {
  char *a = tape_pop(), *save = NULL, *tok;
  int i;
  (void)root; (void)mcc; (void)type;
  if (a == NULL) { tlog("checks  %s -> (unchanged)", wname); return 1; }
  if (is_escape(a)) { free(a); tlog("checks  %s -> <esc>", wname); return 0; }
  if (strspn(a, "01, ") == strlen(a)) {
    i = 0;
    for (tok = strtok_r(a, ", ", &save); tok && i < n; tok = strtok_r(NULL, ", ", &save))
      check[i++] = atoi(tok);
  } else {
    for (i = 0; i < n; i++) check[i] = 0;
    for (tok = strtok_r(a, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
      strip(tok);
      for (i = 0; i < n; i++) {
        char nm[256]; snprintf(nm, sizeof nm, "%s", names[i]); strip(nm);
        if (strcasecmp(nm, tok) == 0) check[i] = 1;
      }
    }
  }
  tlog("checks  %s -> set", wname);
  free(a);
  return 1;
}

int get_dialog(char *wname, char *name, char *value, char *ok, char *cancel, int max) {
  char *a = tape_pop();
  (void)ok; (void)cancel;
  if (a == NULL) { no_answer(name); tlog("dialog  %s: %s [%s] -> (default)", wname, name, value); return 1; }
  if (is_escape(a)) { free(a); tlog("dialog  %s -> <esc>", wname); return 0; }
  if ((int)strlen(a) >= max) { free(a); pyxpp_fail("answer too long for %s", name); }
  tlog("dialog  %s: %s [%s] -> %s", wname, name, value, a);
  strcpy(value, a);
  free(a);
  return 1;
}

int file_selector(char *title, char *file, char *wild) {
  char *a = tape_pop();
  if (a == NULL || is_escape(a) || !*a) {
    tlog("file    %s (%s) -> (cancel)", title, wild);
    free(a);
    return 0;
  }
  tlog("file    %s (%s) -> %s", title, wild, a);
  strcpy(file, a);
  free(a);
  return 1;
}

/* mouse clicks: the answer is "ix,iy" in window pixels.  Python converts
   data coordinates to pixels with pyxpp_real_to_pixel. */
int get_mouse_xy(int *x, int *y, Window w) {
  char *a = tape_pop();
  (void)w;
  if (a == NULL || is_escape(a)) {
    tlog("click   -> (cancel)");
    free(a);
    return 0;
  }
  if (sscanf(a, "%d,%d", x, y) != 2) { free(a); pyxpp_fail("click answer must be 'ix,iy'"); }
  tlog("click   -> %d,%d", *x, *y);
  free(a);
  return 1;
}

int rubber(int *x1, int *y1, int *x2, int *y2, Window w, int f) {
  char *a = tape_pop();
  (void)w; (void)f;
  if (a == NULL || is_escape(a)) { tlog("drag    -> (cancel)"); free(a); return 0; }
  if (sscanf(a, "%d,%d,%d,%d", x1, y1, x2, y2) != 4) { free(a); pyxpp_fail("drag answer must be 'x1,y1,x2,y2'"); }
  tlog("drag    -> %d,%d,%d,%d", *x1, *y1, *x2, *y2);
  free(a);
  return 1;
}

void respond_box(char *button, char *message) {
  (void)button;
  tlog("message %s", message);
}

/* calculator: each tape entry is an expression; stops at <esc> or empty */
void q_calc(void) {
  char *a;
  double z;
  while ((a = tape_pop()) != NULL) {
    if (is_escape(a)) { free(a); break; }
    if (do_calc(a, &z) != -1) tlog("calc    %s = %.16g", a, z);
    else tlog("calc    %s = <error>", a);
    free(a);
  }
}

/* AUTO "Grab": answer selects a diagram point.
     "<n>"     the point with label n
     "#<k>"    the k-th point in the diagram list (0 = first, -1 = last)
     "<esc>"   cancel */
void traverse_diagram(void) {
  char *a = tape_pop();
  DIAGRAM *d = bifd, *e;
  int ix, iy, i, k;
  if (NBifs < 2 || d == NULL) { tlog("grab    -> (no diagram)"); free(a); return; }
  if (a == NULL || is_escape(a)) { tlog("grab    -> (cancel)"); free(a); return; }
  if (a[0] == '#') {
    k = atoi(a + 1);
    if (k < 0) { for (e = d; e->next; e = e->next) ; d = e; k = -k - 1; while (k-- > 0 && d->prev) d = d->prev; }
    else while (k-- > 0 && d->next) d = d->next;
  } else {
    int lab = atoi(a);
    for (e = bifd; e; e = e->next) if (e->lab == lab) break;
    if (e == NULL) { free(a); pyxpp_fail("no diagram point with label %d", lab); }
    d = e;
  }
  tlog("grab    -> %s (branch %d, pt %d, label %d)", a, d->ibr, d->ntot, d->lab);
  free(a);
  CUR_DIAGRAM = d;
  traverse_out(d, &ix, &iy, 1);
  /* identical to the Enter branch of the GUI traverse_diagram() */
  grabpt.ibr = d->ibr;
  grabpt.lab = d->lab;
  for (i = 0; i < 8; i++) grabpt.par[i] = d->par[i];
  grabpt.icp1 = d->icp1;
  grabpt.icp2 = d->icp2;
  grabpt.per = d->per;
  grabpt.torper = d->torper;
  for (i = 0; i < NODE; i++) {
    grabpt.uhi[i] = d->uhi[i];
    grabpt.ulo[i] = d->ulo[i];
    grabpt.u0[i] = d->u0[i];
    grabpt.ubar[i] = d->ubar[i];
    set_ivar(i + 1, grabpt.u0[i]);
  }
  get_ic(0, grabpt.u0);
  grabpt.flag = 1;
  grabpt.itp = d->itp;
  grabpt.ntot = d->ntot;
  grabpt.nfpar = d->nfpar;
  grabpt.index = d->index;
  for (i = 0; i < NAutoPar; i++) constants[Auto_index_to_array[i]] = grabpt.par[i];
  evaluate_derived();
  redraw_params();
  redraw_ics();
}

/* ------------------------------------------------------------------ */
/* per-instance AUTO scratch directory (replaces $HOME in auto_nox.c)   */
/* ------------------------------------------------------------------ */
static char autodir[1024] = "";
char *pyxpp_auto_home(void) {
  if (autodir[0]) return autodir;
  return getenv("HOME");
}
EXPORT void pyxpp_set_auto_dir(const char *d) { snprintf(autodir, sizeof autodir, "%s", d); }

#ifdef _WIN32
/* ------------------------------------------------------------------ */
/* Windows only                                                         */
/* ------------------------------------------------------------------ */
/* wincompat.h routes XPP's getenv() here: HOME is not set on Windows */
#undef getenv
/* wincompat.h renamed getenv before <stdlib.h> was read, so the real one is not
   declared: without this, its result is truncated to a 32-bit int (crash on Windows) */
char *getenv(const char *name);
char *pyxpp_getenv(const char *name) {
  char *v = getenv(name);
  if (v == NULL && name && strcmp(name, "HOME") == 0) {
    v = getenv("USERPROFILE");
    if (v == NULL) v = getenv("TEMP");
    if (v == NULL) v = ".";
  }
  return v;
}
#define getenv pyxpp_getenv

/* see win/wincompat.h: binary mode for written files and for AUTO's own files */
#undef fopen
FILE *pyxpp_fopen(const char *path, const char *mode) {
  char m[16];
  size_t n = strlen(autodir);
  int bin = !strchr(mode, 'b') && !strchr(mode, 't') &&
            (strchr(mode, 'w') || strchr(mode, 'a') || strchr(mode, '+') ||
             (n && path && strncmp(path, autodir, n) == 0));
  if (bin && strlen(mode) + 2 <= sizeof m) {
    snprintf(m, sizeof m, "%sb", mode);
    mode = m;
  }
  return fopen(path, mode);
}
#define fopen pyxpp_fopen

/* stdout/stderr capture (see win/wincompat.h).  The log FILE stays open
 * until pyxpp_shutdown(): XPP may have stored stdout in a variable during a
 * captured call. */
static FILE *cap_file = NULL;
static int cap_on = 0;
FILE *pyxpp_out(int which) {
  if (cap_on && cap_file) return cap_file;
  return pyxpp_crt_stream(which);
}
EXPORT int pyxpp_capture_begin(const char *path) {
  fflush(pyxpp_crt_stream(1)); fflush(pyxpp_crt_stream(2));
  if (cap_file == NULL) cap_file = fopen(path, "ab");
  if (cap_file == NULL) return -1;
  cap_on = 1;
  return 0;
}
EXPORT void pyxpp_capture_end(void) {
  if (cap_file) fflush(cap_file);
  if (logfile) fflush(logfile);
  cap_on = 0;
}
#endif

/* ------------------------------------------------------------------ */
/* API                                                                  */
/* ------------------------------------------------------------------ */
EXPORT int pyxpp_init(int argc, char **argv, const char *logpath) {
  if (logpath && *logpath) {
    FILE *f = fopen(logpath, "a");
    if (f) logfile = f;
  }
  GUARDED({
    do_main(argc, argv);
    /* what do_events() does before its loop */
    help();
    if (RunImmediately == 1) { run_the_commands(4); RunImmediately = 0; }
    if (logfile) fflush(logfile);
  });
}

EXPORT void pyxpp_flush_log(void) { if (logfile) fflush(logfile); }

/* the logfile is reset by do_main; keep it pointed at our file */
EXPORT int pyxpp_set_log(const char *logpath) {
  FILE *f = fopen(logpath, "a");
  if (!f) return -1;
  logfile = f;
  return 0;
}

/* release what keeps files open, before Python removes the work folder
 * (Windows cannot delete an open file).  The model is not usable afterwards. */
EXPORT void pyxpp_shutdown(void) {
#ifdef _WIN32
  pyxpp_capture_end();
  if (logfile == cap_file) logfile = NULL;
  if (cap_file) fclose(cap_file);
  cap_file = NULL;
#endif
  if (logfile && logfile != stdout && logfile != stderr) fclose(logfile);
  logfile = stdout;
}

/* one key press in the main window, honouring XPP's menu state */
EXPORT int pyxpp_key(int ch) { GUARDED(commander((char)ch)); }

/* run a command by its internal id (menudrive.h M_* constants) */
EXPORT int pyxpp_command(int com) { GUARDED(run_the_commands(com)); }

/* numerics menu key (same as pressing 'u' then ch) */
EXPORT int pyxpp_num_key(int ch) { GUARDED(get_num_par((char)ch)); }

EXPORT int pyxpp_menu_state(void) { return help_menu; }
EXPORT int pyxpp_main_menu(void) { GUARDED(help()); }

/* AUTO window buttons (keys a n g r d c u p f).  Opens the window first. */
EXPORT int pyxpp_auto_key(int ch) {
  GUARDED({
    if (Auto.exist == 0) do_auto_win();
    switch (tolower(ch)) {
      case 'a': auto_plot_par(); break;
      case 'n': auto_num_par(); break;
      case 'g': auto_grab(); break;
      case 'r': auto_run(); break;
      case 'd': redraw_diagram(); break;
      case 'c': draw_bif_axes(); break;
      case 'u': auto_per_par(); break;
      case 'p': auto_params(); break;
      case 'f': auto_file(); break;
      default: pyxpp_fail("unknown AUTO key '%c'", ch);
    }
  });
}
EXPORT int pyxpp_auto_open(void) { GUARDED(if (Auto.exist == 0) do_auto_win()); }

/* set any @-option exactly as in an .ode file */
EXPORT int pyxpp_set_option(const char *name, const char *value) {
  char n[256], v[256];
  snprintf(n, sizeof n, "%s", name);
  snprintf(v, sizeof v, "%s", value);
  GUARDED({
    set_option(n, v, 1, NULL);
    do_meth();
    /* as XPP's nUmerics > Dt / Delay dialogs do: the delay buffer and the Volterra
       kernels are sized from dt, so rebuild them when dt or the delay changes */
    if (strncasecmp(n, "DT", 2) == 0 || strncasecmp(n, "DELAY", 5) == 0) {
      if (DELAY > 0.0) { free_delay(); if (alloc_delay(DELAY)) INFLAG = 0; }
      else free_delay();
      if (NKernel > 0) { INFLAG = 0; MyStart = 1; alloc_kernels(1); }
    }
  });
}

/* parameters and variables by name */
EXPORT int pyxpp_get_val(const char *name, double *v) {
  char n[256];
  snprintf(n, sizeof n, "%s", name);
  return get_val(n, v) ? 0 : -1;
}
EXPORT int pyxpp_set_val(const char *name, double v) {
  char n[256];
  int ok;
  snprintf(n, sizeof n, "%s", name);
  ok = set_val(n, v);
  evaluate_derived();
  /* as XPP's parameter window does: kernels and tables may depend on parameters.
     XPP only re-evaluates Volterra kernels with AutoEval on (off by default), which
     silently keeps a kernel of the old parameters; pyxpp always does it. */
  { int keep = AutoEvaluate; AutoEvaluate = 1; re_evaluate_kernels(); AutoEvaluate = keep; }
  redo_all_fun_tables();
  evaluate_derived();
  return ok ? 0 : -1;
}

/* equilibrium via Newton from x (in/out); eigenvalues into er/em */
EXPORT int pyxpp_equilibrium(double *x, double *er, double *em, int *ierr) {
  int i;
  int filled = 1;
  GUARDED({
    evaluate_derived();
    for (i = 0; i < NODE; i++) er[i] = 0.0 / 0.0;
    do_sing_info(x, NEWT_ERR, EVEC_ERR, BOUND, EVEC_ITER, NODE, er, em, ierr);
    /* Newton converged and the eigenvalues were computed: a later failure in XPP's
       eigenvector step (for an unstable direction) does not make it "not found" */
    for (i = 0; i < NODE; i++) if (er[i] != er[i]) filled = 0;
    if (*ierr != 0 && filled) *ierr = 0;
  });
}

/* sizes and names */
EXPORT int pyxpp_node(void) { return NODE; }
EXPORT int pyxpp_neq(void) { return NEQ; }
EXPORT int pyxpp_nmarkov(void) { return NMarkov; }
EXPORT int pyxpp_npar(void) { return NUPAR; }
EXPORT const char *pyxpp_var_name(int i) { return uvar_names[i]; }
EXPORT const char *pyxpp_par_name(int i) { return upar_names[i]; }

/* browser data: rows = storind, cols = NEQ+1 (t first) */
/* The data browser: the stored solution, or whatever XPP put there instead
   (a histogram, a Fourier transform, the adjoint, H, Monte-Carlo mean, ...).
   get_browser_rows() is XPP 8.1's own count of the rows shown. */
int get_browser_rows(void);
float *get_data_col(int c);
EXPORT int pyxpp_data_rows(void) { return get_browser_rows(); }
EXPORT int pyxpp_data_cols(void) { return NEQ + 1; }
EXPORT void pyxpp_data_copy(double *out) {
  int r, c, nc = NEQ + 1, nr = get_browser_rows();
  for (c = 0; c < nc; c++) {
    float *col = get_data_col(c);
    for (r = 0; r < nr; r++) out[r * nc + c] = col ? col[r] : 0.0 / 0.0;
  }
}

/* bifurcation diagram */
EXPORT int pyxpp_diag_count(void) {
  int n = 0;
  DIAGRAM *d;
  for (d = bifd; d; d = d->next) n++;
  return n;
}
/* ints: ibr ntot itp lab icp1 icp2 flag2 index nfpar  (9)
   dbl : norm per torper par[0..19]                     (23)
   vec : uhi ulo u0 ubar evr evi, each NODE long        */
EXPORT int pyxpp_diag_get(int k, int *ints, double *dbl, double *vec) {
  DIAGRAM *d = bifd;
  int i;
  while (k-- > 0 && d) d = d->next;
  if (!d) return -1;
  ints[0] = d->ibr; ints[1] = d->ntot; ints[2] = d->itp; ints[3] = d->lab;
  ints[4] = d->icp1; ints[5] = d->icp2; ints[6] = d->flag2; ints[7] = d->index;
  ints[8] = d->nfpar;
  dbl[0] = d->norm; dbl[1] = d->per; dbl[2] = d->torper;
  for (i = 0; i < 20; i++) dbl[3 + i] = d->par[i];
  for (i = 0; i < NODE; i++) {
    vec[i] = d->uhi[i];
    vec[NODE + i] = d->ulo[i];
    vec[2 * NODE + i] = d->u0[i];
    vec[3 * NODE + i] = d->ubar[i];
    vec[4 * NODE + i] = d->evr ? d->evr[i] : 0.0;
    vec[5 * NODE + i] = d->evi ? d->evi[i] : 0.0;
  }
  return 0;
}
/* XPP 8.1 (hopf_l1.c): first Lyapunov coefficient l1 and frequency omega at
   diagram point k, from its parameters and its equilibrium (ubar).
   l1 < 0: supercritical Hopf, l1 > 0: subcritical.  *ok = 1 if computed. */
int hopf_l1_par(const double *par, const double *u, double *l1, double *omega);
EXPORT int pyxpp_diag_l1(int k, double *l1, double *omega, int *ok) {
  DIAGRAM *d = bifd;
  while (k-- > 0 && d) d = d->next;
  if (!d) return -1;
  GUARDED(*ok = hopf_l1_par(d->par, d->ubar, l1, omega));
}
/* ---- direction fields, computed by XPP ---------------------------------
   pyxpp_rhs: XPP's right-hand side f(t, y) for the first NODE variables
   (and the auxiliary quantities by extra()), at any point.
   pyxpp_dirfield: XPP's own direction-field routine (redraw_dfield) on the
   current 2D view, written to `path` instead of drawn: one line per arrow,
   "x y x_end y_end", with XPP's grid (DF_GRID) and scaling: type 1 = Direct
   field (length ~ |f| / max|f|), type 0 = Scaled (all the same length). */
#include "struct.h"
extern int DF_FLAG, DF_GRID, DFIELD_TYPE, DFSuppress, DF_IX, DF_IY;
extern GRAPH *MyGraph;
extern int (*rhs)();
void extra();
void redraw_dfield(void);
EXPORT int pyxpp_rhs(double t, double *y, double *ydot) {
  GUARDED({ rhs(t, y, ydot, NODE); extra(y, t, NODE, NEQ); });
}
static void dirfield_to_file(int grid, int type, const char *path) {
  int old_flag = DF_FLAG, old_grid = DF_GRID, old_type = DFIELD_TYPE, old_ix = DF_IX, old_iy = DF_IY;
  DF_FLAG = 1; DF_GRID = grid; DFIELD_TYPE = type;
  DF_IX = MyGraph->xv[0]; DF_IY = MyGraph->yv[0];
  DFSuppress = 1;
  redraw_dfield();              /* writes dirfields.dat in the working folder */
  DFSuppress = 0;
  DF_FLAG = old_flag; DF_GRID = old_grid; DFIELD_TYPE = old_type; DF_IX = old_ix; DF_IY = old_iy;
  if (strcmp(path, "dirfields.dat") != 0) rename("dirfields.dat", path);
}
EXPORT int pyxpp_dirfield(int grid, int type, const char *path) {
  GUARDED(dirfield_to_file(grid, type, path));
}
/* the same from a point's own AUTO parameter vector (20) and equilibrium */
EXPORT int pyxpp_hopf_l1(double *par, double *u, double *l1, double *omega, int *ok) {
  GUARDED(*ok = hopf_l1_par(par, u, l1, omega));
}
EXPORT int pyxpp_auto_par_index(int i) { return Auto_index_to_array[i]; }
EXPORT int pyxpp_n_auto_par(void) { return NAutoPar; }

/* AUTO settings snapshot.
   ints: ntst nmx npr ncol icp1 icp2 var plot ips nfpar grab_flag grab_lab exist
   dbl : ds dsmin dsmax rl0 rl1 a0 a1 epsl epsu epss xmin xmax ymin ymax */
extern int AutoPar[];
EXPORT void pyxpp_auto_settings(int *ints, double *dbl) {
  ints[0] = Auto.ntst; ints[1] = Auto.nmx; ints[2] = Auto.npr; ints[3] = Auto.ncol;
  ints[4] = Auto.icp1; ints[5] = Auto.icp2; ints[6] = Auto.var; ints[7] = Auto.plot;
  ints[8] = Auto.ips; ints[9] = Auto.nfpar; ints[10] = grabpt.flag; ints[11] = grabpt.lab;
  ints[12] = Auto.exist;
  dbl[0] = Auto.ds; dbl[1] = Auto.dsmin; dbl[2] = Auto.dsmax; dbl[3] = Auto.rl0;
  dbl[4] = Auto.rl1; dbl[5] = Auto.a0; dbl[6] = Auto.a1; dbl[7] = Auto.epsl;
  dbl[8] = Auto.epsu; dbl[9] = Auto.epss; dbl[10] = Auto.xmin; dbl[11] = Auto.xmax;
  dbl[12] = Auto.ymin; dbl[13] = Auto.ymax;
}
EXPORT const char *pyxpp_auto_par_name(int i) { return upar_names[AutoPar[i]]; }

/* data-coordinate -> draw-window pixel, for answering mouse prompts */
extern void scale_to_screen(double x, double y, int *i, int *j);
EXPORT void pyxpp_real_to_pixel(double x, double y, int *ix, int *iy) {
  scale_to_screen(x, y, ix, iy);
}

/* Phase space > Choose: which variables live on a circle.
   Answer: comma list of variable names; empty/<esc> keeps the old choice. */
extern int itor[], TORUS;
void choose_torus(void) {
  char *a = tape_pop(), *save = NULL, *tok;
  int i, any = 0;
  if (a == NULL || is_escape(a)) { tlog("torus   -> (unchanged)"); free(a); return; }
  for (i = 0; i < NEQ; i++) itor[i] = 0;
  for (tok = strtok_r(a, ", ", &save); tok; tok = strtok_r(NULL, ", ", &save)) {
    int found = 0;
    for (i = 0; i < NEQ; i++)
      if (strcasecmp(uvar_names[i], tok) == 0) { itor[i] = 1; found = 1; }
    if (!found) { char m[200]; snprintf(m, sizeof m, "no variable '%s'", tok); free(a); pyxpp_fail("%s", m); }
  }
  tlog("torus   -> %s", a);
  free(a);
  for (i = 0; i < NEQ; i++) if (itor[i] == 1) any = 1;
  TORUS = any;
}

/* ---- delay equations: history and characteristic roots (XPP 8.2) ---- */
extern char delay_string[][80];
extern double delay_list[];
extern int NDelay, del_stab_flag, NODE;
extern double NEWT_ERR, EVEC_ERR, BOUND;
extern int EVEC_ITER;
extern void rooter(double *x, double err, double eps, double big, double *work, int *ierr, int maxit, int n);
extern double delay_char_coef(double *x, int n, double eps, double **coef_out);
extern int delay_spectrum(double *coef, double *delay, int n, int m, int nev, double *ev,
                          int *nunstab, int *nrefined, int nmax);

/* the history formula of variable i (XPP's x(0)=formula, or Initialconds > Delay) */
EXPORT const char *pyxpp_delay_string(int i) { return (i >= 0 && i < NODE) ? delay_string[i] : ""; }
EXPORT int pyxpp_set_delay_string(int i, const char *s) {
  if (i < 0 || i >= NODE || strlen(s) >= 80) return -1;
  strcpy(delay_string[i], s);
  return 0;
}

/* evaluate an expression in the model's current state (XPP's calculator) */
EXPORT int pyxpp_calc(const char *expr, double *value) {
  char buf[256];
  double z = 0.0;
  int r;
  snprintf(buf, sizeof buf, "%s", expr);
  r = do_calc(buf, &z);
  *value = z;
  return r == -1 ? -1 : 0;
}

/* Newton from x (in/out), then the nev rightmost characteristic roots of the
   linearised delay equation, as (re, im) pairs sorted by real part.
   Returns the number of roots, or -1 (no convergence / failure). */
EXPORT int pyxpp_delay_roots(double *x, int nev, double *ev, int *nunstab) {
  double *work, *coef = NULL;
  int ierr = 0, nref = 0, nr = -1, n = NODE;
  work = (double *)malloc((n * (2 * n + 5) + 50) * sizeof(double));
  if (!work) return -1;
  del_stab_flag = 0;
  evaluate_derived();
  rooter(x, EVEC_ERR, NEWT_ERR, BOUND, work, &ierr, EVEC_ITER, n);
  free(work);
  if (ierr != 0) return -1;
  if (delay_char_coef(x, n, NEWT_ERR, &coef) < 0.0) return -1;
  del_stab_flag = -1;
  nr = delay_spectrum(coef, delay_list, n, NDelay, nev, ev, nunstab, &nref, 600);
  free(coef);
  del_stab_flag = 1;
  return nr;
}

/* ---- starting data, stored solutions and the final state (2026-11-17) ---- */
extern int MAXSTOR, INFLAG, DelayFlag;
extern double MyData[], MyTime;
extern char this_auto_file[];
extern void data_back(void);
extern void refresh_browser(int length);
extern void extra(double *y, double t, int nod, int neq);
int dde_po_get_orbit(int ibr, int lab, double per, int maxrows, int *nrow, double **ts, double **us);

/* Put given data into XPP's data storage, as if an integration had produced it:
   data is nrow x (1 + nvar), row-major, t first, then the first nvar variables
   (nvar = NODE + NMarkov).  The auxiliary quantities are computed by XPP.
   AUTO's runs from scratch ("Periodic", "Bdry value", "Homoclinic") start
   from these rows: the orbit is read at equally spaced row indices and the
   period is the last time (the first must be 0). */
static int load_storage(int nrow, int nvar, const double *data) {
  double x[MAXODE];
  int r, j;
  if (nrow < 2 || nrow > MAXSTOR) {
    snprintf(errbuf, sizeof errbuf, "%d rows do not fit in XPP's storage (2 to MAXSTOR = %d)", nrow, MAXSTOR);
    return -1;
  }
  if (nvar != NODE + NMarkov) {
    snprintf(errbuf, sizeof errbuf, "expected %d variables, got %d", NODE + NMarkov, nvar);
    return -1;
  }
  data_back();
  for (r = 0; r < nrow; r++) {
    const double *row = data + (size_t)r * (nvar + 1);
    for (j = 0; j < MAXODE && j < NEQ; j++) x[j] = 0.0;
    for (j = 0; j < nvar; j++) x[j] = row[1 + j];
    extra(x, row[0], NODE, NEQ);
    storage[0][r] = (float)row[0];
    for (j = 0; j < NEQ; j++) storage[j + 1][r] = (float)x[j];
  }
  storind = nrow;
  refresh_browser(nrow);
  return 0;
}
EXPORT int pyxpp_load_storage(int nrow, int nvar, const double *data) {
  if (nrow < 2 || nrow > MAXSTOR || nvar != NODE + NMarkov) {
    errbuf[0] = 0;
    return load_storage(nrow, nvar, data);      /* only writes the message */
  }
  GUARDED(load_storage(nrow, nvar, data));
}

/* The solution AUTO stored for label lab (the .s file, i.e. fort.8), on AUTO's
   mesh: t is AUTO's normalised time in [0,1].  Rows go to t[0..maxrow) and
   u[row*ndim + k] (all ndim components AUTO stored; the first NODE are the
   variables).  *nrow and *ndim are set even when maxrow is too small (call with
   maxrow = 0 to ask for the size).  Periodic orbits of delay equations are kept
   by XPP's delay orbit solver, not in the file (ibr, per identify them).
   Returns 0, or -1 if the label is not stored. */
static int read_solution_header(FILE *fp, int lab, int *ntpl, int *nar) {
  static char line[8192];
  long v[9];
  int i;
  while (fgets(line, sizeof line, fp)) {
    if (sscanf(line, "%ld %ld %ld %ld %ld %ld %ld %ld %ld",
               v, v + 1, v + 2, v + 3, v + 4, v + 5, v + 6, v + 7, v + 8) != 9) return 0;
    if (v[3] == lab) { *ntpl = (int)v[6]; *nar = (int)v[7]; return 1; }
    for (i = 0; i < v[8]; i++)
      if (!fgets(line, sizeof line, fp)) return 0;
  }
  return 0;
}
EXPORT int pyxpp_auto_solution(int ibr, int lab, double per, int maxrow,
                               int *nrow, int *ndim, double *t, double *u) {
  char path[1200];
  FILE *fp;
  int i, k, ntpl = 0, nar = 0;
  *nrow = 0; *ndim = 0;
  errbuf[0] = 0;
  if (ibr < 0 && DelayFlag) {
    double *ts = NULL, *us = NULL;
    int nr = 0;
    if (!dde_po_get_orbit(ibr, lab, per, 1 << 24, &nr, &ts, &us)) {
      snprintf(errbuf, sizeof errbuf, "the orbit of label %d is not stored (delay periodic orbits "
               "are kept for the current session only)", lab);
      return -1;
    }
    *nrow = nr; *ndim = NODE;
    for (i = 0; i < nr && i < maxrow; i++) {
      t[i] = ts[i];
      for (k = 0; k < NODE; k++) u[(size_t)i * NODE + k] = us[(size_t)i * NODE + k];
    }
    free(ts); free(us);
    return 0;
  }
  snprintf(path, sizeof path, "%s.s", this_auto_file);
  fp = fopen(path, "r");
  if (fp == NULL) {
    snprintf(errbuf, sizeof errbuf, "AUTO has stored no solutions yet (%s)", path);
    return -1;
  }
  if (!read_solution_header(fp, lab, &ntpl, &nar) || ntpl < 1 || nar < 2) {
    fclose(fp);
    snprintf(errbuf, sizeof errbuf, "label %d is not in AUTO's stored solutions", lab);
    return -1;
  }
  *nrow = ntpl; *ndim = nar - 1;
  if (maxrow >= ntpl) {
    for (i = 0; i < ntpl; i++) {
      if (fscanf(fp, "%lg", t + i) != 1) break;
      for (k = 0; k < nar - 1; k++)
        if (fscanf(fp, "%lg", u + (size_t)i * (nar - 1) + k) != 1) break;
      if (k < nar - 1) break;
    }
    if (i < ntpl) {
      fclose(fp);
      snprintf(errbuf, sizeof errbuf, "AUTO's stored solution of label %d is cut short", lab);
      return -1;
    }
  }
  fclose(fp);
  return 0;
}

/* The state where the last integration stopped, in double precision (XPP
   integrates in MyData; the data browser keeps floats): x[0..NODE+NMarkov)
   and the time.  Returns -1 if nothing has been integrated. */
EXPORT int pyxpp_last_state(double *x, double *t) {
  int i;
  if (!INFLAG) return -1;
  for (i = 0; i < NODE + NMarkov; i++) x[i] = MyData[i];
  *t = MyTime;
  return 0;
}

/* ------------------------------------------------------------------------
   Stable manifold of a saddle of a planar map, without the inverse
   (XPPAUT 8.4 of 25 Nov: mapmanifold.c, after England, Krauskopf & Osinga,
   SIAM J. Appl. Dyn. Syst. 3 (2004) 161-190).  XPP draws the manifold as an
   overlay on the graph; here the points are returned instead.
   base[NODE]: all variables (i1, i2 also the first guess of the saddle);
   i1, i2: the two free variables (0-based), the others stay at base.
   par[12]: initial step, Delta_min, Delta_max, alpha_min, alpha_max,
            (Delta alpha)_min, (Delta alpha)_max, arclength, xlo, xhi, ylo, yhi
   cap: room in each of x0,y0,x1,y1.  info[5]: saddle (2), stable multiplier,
   stable eigenvector (2).  n[2], why[2]: points and stop reason per branch
   (branch 0 leaves the saddle along +v, branch 1 along -v).
   Returns 0, or -1 with pyxpp_error() set.
   ------------------------------------------------------------------------ */
#include <math.h>
#include "mapmanifold.h"
extern int METHOD;
extern int (*rhs)();
typedef struct { double base[MAXODE]; int i1, i2; } PyxppMap2;
static int pyxpp_map2(const double *x, double *y, void *ctx) {
  PyxppMap2 *m = (PyxppMap2 *)ctx;
  double in[MAXODE], out[MAXODE];
  int i;
  for (i = 0; i < NODE; i++) in[i] = m->base[i];
  in[m->i1] = x[0];
  in[m->i2] = x[1];
  (*rhs)(0.0, in, out, NODE);
  y[0] = out[m->i1];
  y[1] = out[m->i2];
  return 1;
}
static void pyxpp_map_jac2(PyxppMap2 *m, const double *p, double J[4]) {
  int j;
  for (j = 0; j < 2; j++) {
    double q[2], a[2], b[2];
    double h = 1e-6 * (fabs(p[j]) > 1.0 ? fabs(p[j]) : 1.0);
    q[0] = p[0]; q[1] = p[1];
    q[j] = p[j] + h; pyxpp_map2(q, a, m);
    q[j] = p[j] - h; pyxpp_map2(q, b, m);
    J[0 + j] = (a[0] - b[0]) / (2.0 * h);
    J[2 + j] = (a[1] - b[1]) / (2.0 * h);
  }
}
/* as mapview.c's map_saddle_data: Newton with central differences, then the
   stable multiplier and unit eigenvector of the 2x2 Jacobian */
static const char *pyxpp_saddle(PyxppMap2 *m, double *p, double *lam, double *v) {
  double J[4], F[2], r0, r1, det, tr, disc, l1, l2, ls, vn;
  int it;
  for (it = 0; it < 30; it++) {
    pyxpp_map2(p, F, m);
    r0 = F[0] - p[0]; r1 = F[1] - p[1];
    if (fabs(r0) + fabs(r1) < 1e-14) break;
    pyxpp_map_jac2(m, p, J);
    J[0] -= 1.0; J[3] -= 1.0;
    det = J[0] * J[3] - J[1] * J[2];
    if (det == 0.0) return "singular Jacobian in Newton's method";
    p[0] -= (J[3] * r0 - J[1] * r1) / det;
    p[1] -= (-J[2] * r0 + J[0] * r1) / det;
  }
  pyxpp_map2(p, F, m);
  if (fabs(F[0] - p[0]) + fabs(F[1] - p[1]) > 1e-8) return "Newton's method did not reach a fixed point";
  pyxpp_map_jac2(m, p, J);
  tr = J[0] + J[3];
  det = J[0] * J[3] - J[1] * J[2];
  disc = tr * tr - 4.0 * det;
  if (disc <= 0.0) return "the fixed point is not a saddle (complex multipliers)";
  l1 = 0.5 * (tr + sqrt(disc));
  l2 = 0.5 * (tr - sqrt(disc));
  if (fabs(l1) < 1.0 && fabs(l2) > 1.0) ls = l1;
  else if (fabs(l2) < 1.0 && fabs(l1) > 1.0) ls = l2;
  else return "the fixed point is not a saddle";
  v[0] = J[1]; v[1] = ls - J[0];
  vn = sqrt(v[0] * v[0] + v[1] * v[1]);
  if (vn < 1e-10 * (fabs(J[0]) + fabs(J[1]) + fabs(J[2]) + fabs(J[3]))) {
    v[0] = ls - J[3]; v[1] = J[2];
    vn = sqrt(v[0] * v[0] + v[1] * v[1]);
  }
  if (vn == 0.0) return "no stable eigenvector";
  v[0] /= vn; v[1] /= vn;
  *lam = ls;
  return NULL;
}
static const char *pyxpp_manifold(double *base, int i1, int i2, double *par, int cap,
                                  double *x0, double *y0, double *x1, double *y1,
                                  int *n, int *why, double *info) {
  PyxppMap2 m;
  MfParams mp;
  MfBranch br[2];
  double p[2], v[2], lam = 0.0;
  const char *bad;
  int i, b;
  if (METHOD != 0) return "the stable manifold without the inverse is for maps (meth=discrete)";
  if (i1 < 0 || i2 < 0 || i1 >= NODE || i2 >= NODE || i1 == i2) return "two different variables are needed";
  for (i = 0; i < NODE; i++) m.base[i] = base[i];
  m.i1 = i1; m.i2 = i2;
  p[0] = base[i1]; p[1] = base[i2];
  bad = pyxpp_saddle(&m, p, &lam, v);
  if (bad) return bad;
  m.base[i1] = p[0]; m.base[i2] = p[1];
  memset(&mp, 0, sizeof mp);
  mp.delta0 = par[0]; mp.dmin = par[1]; mp.dmax = par[2];
  mp.amin = par[3]; mp.amax = par[4]; mp.devmin = par[5]; mp.dev = par[6];
  mp.maxlen = par[7]; mp.maxpts = cap;
  mp.xlo = par[8]; mp.xhi = par[9]; mp.ylo = par[10]; mp.yhi = par[11];
  if (mf_stable_manifold(pyxpp_map2, &m, p, v, lam, &mp, br) != 0)
    return "could not start the stable manifold computation";
  for (b = 0; b < 2; b++) {
    double *xs = b ? x1 : x0, *ys = b ? y1 : y0;
    n[b] = br[b].n < cap ? br[b].n : cap;
    for (i = 0; i < n[b]; i++) { xs[i] = br[b].x[i]; ys[i] = br[b].y[i]; }
    why[b] = br[b].why;
    mf_free(&br[b]);
  }
  info[0] = p[0]; info[1] = p[1]; info[2] = lam; info[3] = v[0]; info[4] = v[1];
  return NULL;
}
static int pyxpp_guarded_manifold(double *base, int i1, int i2, double *par, int cap,
                                  double *x0, double *y0, double *x1, double *y1,
                                  int *n, int *why, double *info, const char **bad) {
  GUARDED(*bad = pyxpp_manifold(base, i1, i2, par, cap, x0, y0, x1, y1, n, why, info));
}
EXPORT int pyxpp_map_stable_manifold(double *base, int i1, int i2, double *par, int cap,
                                     double *x0, double *y0, double *x1, double *y1,
                                     int *n, int *why, double *info) {
  const char *bad = NULL;
  int rc;
  rc = pyxpp_guarded_manifold(base, i1, i2, par, cap, x0, y0, x1, y1, n, why, info, &bad);
  if (rc == 0 && bad) { snprintf(errbuf, sizeof errbuf, "%s", bad); return -1; }
  return rc;
}
/* the 2D view: xmin xmax ymin ymax, and the plotted columns (0 = t, 1.. = variables) */
EXPORT void pyxpp_view_get(double *w, int *xy) {
  w[0] = MyGraph->xmin; w[1] = MyGraph->xmax; w[2] = MyGraph->ymin; w[3] = MyGraph->ymax;
  xy[0] = MyGraph->xv[0]; xy[1] = MyGraph->yv[0];
}
