// The pyauto-xpp editor's widget runtime: Jupyter's widget manager (ipywidgets 8) and anywidget,
// talking to the Python session through pyxpp serve instead of a Jupyter kernel.
import { HTMLManager } from "@jupyter-widgets/html-manager";
import * as base from "@jupyter-widgets/base";
import * as controls from "@jupyter-widgets/controls";

// anywidget's browser module registers itself with an AMD define(); the build turns that one
// call into __pyxpp_anywidget(...) (build.mjs), which hands its factory over here
import { anyFactory } from "./anyshim.js";
import "anywidget";

const b64ToView = (s) => { const bin = atob(s); const a = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i); return new DataView(a.buffer); };
const viewToB64 = (b) => { const u = b instanceof ArrayBuffer ? new Uint8Array(b) : new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
  let s = ""; for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000)); return btoa(s); };

// Third-party widgets: Jupyter widget packages publish an AMD bundle (dist/index.js) for pages
// without JupyterLab.  It is fetched from jsDelivr and run with a private define(), whose
// dependencies are this bundle's own copies of the widget base and controls.
const SHARED = { "@jupyter-widgets/base": base, "@jupyter-widgets/controls": controls };
const amdCache = new Map();
function cdnVersion(v) { const m = String(v || "").match(/\d+(\.\d+){0,2}/); return m ? m[0] : "latest"; }
export async function amdLoad(name, version) {
  const key = name + "@" + version;
  if (!amdCache.has(key)) amdCache.set(key, (async () => {
    const url = `https://cdn.jsdelivr.net/npm/${name}@${cdnVersion(version)}/dist/index.js`;
    const r = await fetch(url); if (!r.ok) throw new Error(`${name}@${version}: ${r.status} from ${url}`);
    const code = await r.text();
    let result;
    const define = (a, b, c) => {
      const deps = Array.isArray(a) ? a : Array.isArray(b) ? b : [], factory = [a, b, c].find((x) => typeof x === "function") || [a, b, c].find((x) => x && typeof x === "object" && !Array.isArray(x));
      const exports = {};
      const args = deps.map((d) => d === "exports" ? exports : d === "module" ? { exports } : d === "require" ? (x) => SHARED[x] : SHARED[d]);
      if (deps.some((d) => !["exports", "module", "require"].includes(d) && !(d in SHARED))) throw new Error(`${name} needs ${deps.join(", ")}`);
      const out = typeof factory === "function" ? factory(...args) : factory;
      result = out !== undefined ? out : exports;
    };
    define.amd = { jQuery: true };
    new Function("define", code + `\n//# sourceURL=${url}`)(define);
    if (!result) throw new Error(`${name}: no module defined in ${url}`);
    return result;
  })());
  return amdCache.get(key);
}

class Comm {
  constructor(transport, id, target) { this.t = transport; this.comm_id = id; this.target_name = target; this._msg = null; this._close = null; }
  on_msg(cb) { this._msg = cb; }
  on_close(cb) { this._close = cb; }
  _send(msg_type, content, metadata, buffers, callbacks) {
    const done = this.t.send({ msg_type, content, metadata: metadata || {}, buffers: (buffers || []).map(viewToB64) });
    // the widget model counts pending messages and waits for "idle": give it once Python has the message
    Promise.resolve(done).finally(() => { try { callbacks?.iopub?.status?.({ content: { execution_state: "idle" } }); } catch (e) {} });
    return this.comm_id;
  }
  send(data, callbacks, metadata, buffers) { return this._send("comm_msg", { comm_id: this.comm_id, data }, metadata, buffers, callbacks); }
  open(data, callbacks, metadata, buffers) { return this._send("comm_open", { comm_id: this.comm_id, target_name: this.target_name, data }, metadata, buffers, callbacks); }
  close(data, callbacks, metadata, buffers) { return this._send("comm_close", { comm_id: this.comm_id, data: data || {} }, metadata, buffers, callbacks); }
}

const toMsg = (m) => Object.assign({}, m, { buffers: (m.buffers || []).map(b64ToView) });

export class LiveManager extends HTMLManager {
  constructor(transport) {
    super({ loader: (name, version) => this._loadModule(name, version) });
    this.transport = transport; this.comms = new Map();
  }
  // the Python session ended: its widgets are gone
  async reset() {
    // the old widgets stay on the page (greyed out) but no longer talk to anything
    for (const c of this.comms.values()) c.t = { send: () => Promise.resolve() };
    this.comms.clear();
  }
  async _loadModule(name, version) {
    if (name === "anywidget") return (this._any = this._any || anyFactory()(base));   // once: model and view share its runtimes
    return amdLoad(name, version);
  }
  _get_comm_info() { return Promise.resolve({}); }
  _create_comm(target, model_id, data, metadata, buffers) {
    const c = new Comm(this.transport, model_id || (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2)), target);
    this.comms.set(c.comm_id, c);
    if (data !== undefined) c.open(data, undefined, metadata, buffers);
    return Promise.resolve(c);
  }
  // messages from Python
  async receive(m) {
    const msg = toMsg(m), id = msg.content.comm_id;
    if (msg.msg_type === "comm_open") {
      const c = new Comm(this.transport, id, msg.content.target_name); this.comms.set(id, c);
      return this.handle_comm_open(c, msg);
    }
    const c = this.comms.get(id); if (!c) return;
    if (msg.msg_type === "comm_msg") return c._msg && c._msg(msg);
    if (msg.msg_type === "comm_close") { this.comms.delete(id); return c._close && c._close(msg); }
  }
  async show(model_id, el) {
    for (let i = 0; i < 200 && !this.has_model(model_id); i++) await new Promise((r) => setTimeout(r, 50));
    const model = await this.get_model(model_id);
    // anywidget sets a model up in the background (it loads the widget's own JavaScript first):
    // until then a view cannot be made, so try again for a while
    let view = null, err = null;
    for (let i = 0; i < 120 && !view; i++) {
      try { view = await this.create_view(model); } catch (e) { err = e; await new Promise((r) => setTimeout(r, 250)); }
    }
    if (!view) throw err;
    return this.display_view(view, el);
  }
}
