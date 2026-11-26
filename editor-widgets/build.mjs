import * as esbuild from "esbuild";
import { readFile } from "node:fs/promises";
const anywidgetDefine = {
  name: "anywidget-define",
  setup(b) {
    b.onLoad({ filter: /anywidget[\\/]dist[\\/]index\.js$/ }, async (args) => {
      const src = await readFile(args.path, "utf8");
      const out = src.replace(/\bdefine\(\["@jupyter-widgets\/base"\]/, 'globalThis.__pyxpp_anywidget(["@jupyter-widgets/base"]');
      if (out === src) throw new Error("anywidget: define() call not found");
      return { contents: out, loader: "js" };
    });
  },
};
await esbuild.build({
  entryPoints: ["entry.js"], bundle: true, format: "iife", globalName: "PyxppWidgets", outfile: process.env.OUT || "dist/widgets.js", minify: true,
  define: { __webpack_public_path__: '"./"' }, plugins: [anywidgetDefine], logLevel: "error",
  loader: { ".css": "css", ".woff2": "file", ".woff": "file", ".ttf": "file", ".eot": "file", ".svg": "dataurl" },
});
console.log("built");
