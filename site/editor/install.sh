#!/bin/sh
# Install pyauto-xpp on this computer for the editor at https://zetaone.github.io/pyauto-xpp/editor/
#   curl -LsSf https://zetaone.github.io/pyauto-xpp/editor/install.sh | sh
# macOS and Linux.  Finds out what the computer is and installs only what is missing:
#   uv (https://docs.astral.sh/uv/), a Python for pyauto-xpp in its own folder (~/.pyxpp/env),
#   the pyauto-xpp wheel built for this machine (from the project's wheel list) and matplotlib.
# Then registers pyxpp:// links (the editor's "Start pyxpp" button) and starts `pyxpp serve`.
# No administrator rights.  Remove it all with:  ~/.pyxpp/env/bin/pyxpp uninstall
set -eu
SITE="${PYXPP_SITE:-https://zetaone.github.io/pyauto-xpp}"
SITE="${SITE%/}"
INDEX="${PYXPP_INDEX:-$SITE/wheels/}"
PYXPP_HOME="${PYXPP_HOME:-$HOME/.pyxpp}"
PYVER="${PYXPP_PYTHON:-3.12}"
NEED="0.6.0"   # the first pyauto-xpp with pyxpp serve
say() { printf 'pyxpp: %s\n' "$*"; }
os=$(uname -s); arch=$(uname -m)
case "$os" in
  Darwin) sys="macOS $(sw_vers -productVersion 2>/dev/null || true)";;
  Linux) sys="Linux";;
  *) say "this installer is for macOS and Linux (this is $os). On Windows: install-pyxpp.bat from the editor page."; exit 1;;
esac
say "this computer: $sys, $arch"
mkdir -p "$PYXPP_HOME"

# 1. uv
if command -v uv >/dev/null 2>&1; then UV=$(command -v uv)
elif [ -x "$HOME/.local/bin/uv" ]; then UV="$HOME/.local/bin/uv"
elif [ -x "$PYXPP_HOME/bin/uv" ]; then UV="$PYXPP_HOME/bin/uv"
else
  say "installing uv into $PYXPP_HOME/bin"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PYXPP_HOME/bin" UV_NO_MODIFY_PATH=1 sh >/dev/null
  UV="$PYXPP_HOME/bin/uv"
fi
say "$("$UV" --version)"
printf '%s\n' "$UV" > "$PYXPP_HOME/uv-path"

# 2. pyauto-xpp's own Python environment (uv fetches a Python if this computer has none)
PY="$PYXPP_HOME/env/bin/python"
if [ ! -x "$PY" ]; then
  say "making a Python $PYVER environment in $PYXPP_HOME/env"
  "$UV" venv --quiet --python "$PYVER" "$PYXPP_HOME/env"
fi

# 3. the pyauto-xpp wheel for this machine, and matplotlib for figures
say "installing pyauto-xpp from $INDEX"
"$UV" pip install --quiet --python "$PY" --upgrade "pyauto-xpp>=$NEED" matplotlib --find-links "$INDEX" || {
  say "could not install pyauto-xpp $NEED or newer from $INDEX (a new release may still be building: try again in a few minutes)"; exit 1; }
say "pyauto-xpp $("$PY" -m pyxpp version) for $("$PY" -c 'import platform; print(platform.system(), platform.machine(), "Python", platform.python_version())')"

# 4. pyxpp:// links, then start (an older server is stopped first)
"$PY" -m pyxpp register || say "pyxpp:// links not registered; start it with: $PYXPP_HOME/env/bin/pyxpp start"
"$PY" -m pyxpp stop >/dev/null 2>&1 || true
"$PY" -m pyxpp start
say "done. Editor: $SITE/editor/  (it connects by itself; next time use its Start pyxpp button)"
if [ -z "${PYXPP_NO_BROWSER:-}" ]; then
  if [ "$os" = Darwin ]; then open "$SITE/editor/" || true
  else (xdg-open "$SITE/editor/" >/dev/null 2>&1 &) || true; fi
fi
