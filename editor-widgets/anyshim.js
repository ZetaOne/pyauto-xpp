let factory = null;
globalThis.__pyxpp_anywidget = (deps, f) => { factory = typeof deps === "function" ? deps : f; };
export const anyFactory = () => factory;
