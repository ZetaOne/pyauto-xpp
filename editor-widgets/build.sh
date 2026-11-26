#!/bin/sh
# Build the editor's widget runtime into ../pyxpp/editor/widgets/ (needs Node.js 18+ and npm).
# node_modules lives in a temporary folder (the source may be on a drive without symlinks).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
OUT="$HERE/../pyxpp/editor/widgets"
TMP=$(mktemp -d)
cp "$HERE"/package.json "$HERE"/package-lock.json "$HERE"/entry.js "$HERE"/anyshim.js "$HERE"/build.mjs "$TMP"/
cd "$TMP"
npm ci --silent --no-audit --no-fund
OUT="$TMP/dist/widgets.js" node build.mjs
rm -rf "$OUT"; mkdir -p "$OUT"
cp "$TMP"/dist/* "$OUT"/
cp "$HERE"/THIRD-PARTY.md "$OUT"/
rm -rf "$TMP"
echo "built $OUT: $(du -sh "$OUT" | cut -f1)"
