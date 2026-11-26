# Install pyauto-xpp on this computer for the editor at https://zetaone.github.io/pyauto-xpp/editor/
#   powershell -ExecutionPolicy ByPass -c "irm https://zetaone.github.io/pyauto-xpp/editor/install.ps1 | iex"
# Windows.  Finds out what the computer is and installs only what is missing: uv, a Python for
# pyauto-xpp in its own folder (%USERPROFILE%\.pyxpp\env), the pyauto-xpp wheel for this machine
# and matplotlib.  Then registers pyxpp:// links and starts `pyxpp serve`.  No administrator rights.
# Remove it all with:  %USERPROFILE%\.pyxpp\env\Scripts\pyxpp uninstall
$ErrorActionPreference = "Stop"
$Site = if ($env:PYXPP_SITE) { $env:PYXPP_SITE.TrimEnd("/") } else { "https://zetaone.github.io/pyauto-xpp" }
$Index = if ($env:PYXPP_INDEX) { $env:PYXPP_INDEX } else { "$Site/wheels/" }
$PHome = if ($env:PYXPP_HOME) { $env:PYXPP_HOME } else { Join-Path $env:USERPROFILE ".pyxpp" }
$PyVer = if ($env:PYXPP_PYTHON) { $env:PYXPP_PYTHON } else { "3.12" }
function Say($m) { Write-Host "pyxpp: $m" }
$arch = $env:PROCESSOR_ARCHITECTURE
Say "this computer: Windows $([Environment]::OSVersion.Version), $arch"
New-Item -ItemType Directory -Force -Path $PHome | Out-Null

# 1. uv
$uv = $null
$cmd = Get-Command uv -ErrorAction SilentlyContinue
if ($cmd) { $uv = $cmd.Source }
foreach ($c in @((Join-Path $env:USERPROFILE ".local\bin\uv.exe"), (Join-Path $PHome "bin\uv.exe"))) {
  if (-not $uv -and (Test-Path $c)) { $uv = $c }
}
if (-not $uv) {
  Say "installing uv into $PHome\bin"
  $env:UV_INSTALL_DIR = Join-Path $PHome "bin"; $env:UV_NO_MODIFY_PATH = "1"
  powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex" | Out-Null
  $uv = Join-Path $PHome "bin\uv.exe"
}
Say (& $uv --version)
[IO.File]::WriteAllText((Join-Path $PHome "uv-path"), $uv)

# 2. pyauto-xpp's own Python environment (pyauto-xpp has x86_64 wheels for Windows;
#    on ARM64 Windows an x86_64 Python runs them under emulation)
$envdir = Join-Path $PHome "env"
$py = Join-Path $envdir "Scripts\python.exe"
# an environment whose Python no longer runs is made again (your files are not touched)
if (Test-Path $envdir) {
  $okpy = $false
  if (Test-Path $py) { & $py -c "import sys" 2>$null; $okpy = ($LASTEXITCODE -eq 0) }
  if (-not $okpy) { Say "the environment in $envdir is broken: making it again"; Remove-Item -Recurse -Force $envdir }
}
$before = ""
if (Test-Path $py) { $before = (& $py -m pyxpp version 2>$null) }
if (-not (Test-Path $py)) {
  $req = if ($arch -eq "ARM64") { "cpython-$PyVer-windows-x86_64-none" } else { $PyVer }
  Say "making a Python $PyVer environment in $envdir"
  & $uv venv --quiet --python $req $envdir
  if ($LASTEXITCODE -ne 0) { throw "uv venv failed" }
}

# 3. the pyauto-xpp wheel for this machine (the newest), and matplotlib only if missing
#    (the editor offers updates of the other packages, and you choose)
Say "installing pyauto-xpp from $Index"
& $uv pip install --quiet --python $py --upgrade-package pyauto-xpp "pyauto-xpp>=0.6.0" matplotlib scipy pandas tqdm plotly "ipywidgets>=8.1,<8.2" "anywidget>=0.11,<0.12" --find-links $Index
if ($LASTEXITCODE -ne 0) { throw "could not install pyauto-xpp 0.6.0 or newer from $Index (a new release may still be building: try again in a few minutes)" }
$after = (& $py -m pyxpp version)
Say "pyauto-xpp $after"

# 4. pyxpp:// links, then start (an older server is stopped first)
& $py -m pyxpp register
# the server: left alone when it already runs this version, restarted after an upgrade
& $py -m pyxpp status *> $null
if ($LASTEXITCODE -eq 0) {
  if ("$before" -eq "$after") { Say "pyxpp serve is already running pyauto-xpp ${after}: left as it is" }
  else { Say "restarting pyxpp serve with pyauto-xpp $after"; & $py -m pyxpp stop | Out-Null; & $py -m pyxpp start }
} else { & $py -m pyxpp start }
Say "done. Editor: $Site/editor/  (it connects by itself; next time use its Start pyxpp button)"
if (-not $env:PYXPP_NO_BROWSER) { Start-Process "$Site/editor/" }
