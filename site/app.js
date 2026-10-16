// Bundle configurator: pick a target, the execution providers and how each one is provided, then create a bundle
// request issue. The selection rules mirror make_plan in ci/bundle.py, which validates the request again.
"use strict";

// ---------------------------------------------------------------------------
// Selection logic (no DOM; also loaded by tests under Node)
// ---------------------------------------------------------------------------

const PROVIDER_LIBS = {
  cuda: "cuda", tensorrt: "tensorrt", "nv-tensorrt-rtx": "nv_tensorrt_rtx", openvino: "openvino", qnn: "qnn",
  migraphx: "migraphx", vitisai: "vitisai", dnnl: "dnnl", cann: "cann",
};
const CUDA_EPS = ["cuda", "tensorrt", "nv-tensorrt-rtx"];
const CUSTOM_CORE = "onnxruntime-custom";

const PLATFORMS = [
  ["Windows", (t) => t.includes("-windows-")],
  ["Linux", (t) => t.endsWith("-linux-gnu")],
  ["macOS", (t) => t.endsWith("-apple-darwin")],
  ["Android", (t) => t.includes("-android")],
  ["iOS", (t) => /-apple-ios$/.test(t) && !t.startsWith("x86_64")],
  ["iOS Simulator", (t) => /-apple-ios(-sim)?$/.test(t) && (t.endsWith("-sim") || t.startsWith("x86_64"))],
  ["Mac Catalyst", (t) => t.endsWith("-ios-macabi")],
  ["tvOS", (t) => /-apple-tvos$/.test(t) && !t.startsWith("x86_64")],
  ["tvOS Simulator", (t) => /-apple-tvos(-sim)?$/.test(t) && (t.endsWith("-sim") || t.startsWith("x86_64"))],
  ["visionOS", (t) => t.endsWith("-apple-visionos")],
  ["visionOS Simulator", (t) => t.endsWith("-apple-visionos-sim")],
  ["WebAssembly", (t) => t.startsWith("wasm32")],
];
const ARCHS = { x86_64: "x64", aarch64: "ARM64", armv7: "ARMv7", i686: "x86", s390x: "s390x", wasm32: "wasm32",
                arm64ec: "ARM64EC", arm64x: "ARM64X" };

function platformOf(triple) {
  const p = PLATFORMS.find(([, test]) => test(triple));
  return p ? p[0] : triple;
}

function archOf(triple) {
  const a = triple.split("-")[0];
  return ARCHS[a] || a;
}

function itemsFor(cat, triple) {
  return cat.items.filter((it) => it.triple === triple && it.asset);
}

// The libraries a core provides (onnxruntime and its shared provider libraries)
function coreProvides(sel, cat) {
  if (sel.core === CUSTOM_CORE) {
    return ["onnxruntime", ...sel.compile.filter((ep) => PROVIDER_LIBS[ep]).map((ep) => `onnxruntime_providers_${PROVIDER_LIBS[ep]}`)];
  }
  const core = findItem(cat, sel.core, sel.triple);
  return core ? core.provides : [];
}

function findItem(cat, pkg, triple) {
  return cat.items.find((it) => it.package === pkg && it.triple === triple);
}

function coreEps(sel, cat) {
  if (sel.core === CUSTOM_CORE) {
    const eps = { cpu: "builtin" };
    for (const ep of sel.compile) eps[ep] = PROVIDER_LIBS[ep] ? "provider" : "builtin";
    return eps;
  }
  const core = findItem(cat, sel.core, sel.triple);
  return core ? core.eps : {};
}

// Validate a selection; returns an error message or "".
function validate(sel, cat) {
  if (!sel.core) return "select a core";
  if (sel.core === CUSTOM_CORE && !sel.compile.length) return "select at least one in-tree EP for the custom core";
  const owner = {};
  const parts = [[sel.core, coreProvides(sel, cat)]];
  for (const p of sel.plugins) {
    const it = findItem(cat, p, sel.triple);
    if (!it || !it.asset) return `${p} is not available for ${sel.triple}`;
    parts.push([p, it.provides]);
  }
  for (const [name, libs] of parts) {
    for (const lib of libs) {
      if (owner[lib]) return `${name} and ${owner[lib]} both contain ${lib}`;
      owner[lib] = name;
    }
  }
  const majors = cudaMajors(sel, cat);
  if (majors.size > 1) return `mixes CUDA ${[...majors].sort().join(" and ")}`;
  return "";
}

function cudaMajors(sel, cat) {
  const majors = new Set();
  const names = sel.core === CUSTOM_CORE ? sel.plugins : [sel.core, ...sel.plugins];
  for (const n of names) {
    const it = findItem(cat, n, sel.triple);
    if (it && it.cuda_major) majors.add(it.cuda_major);
  }
  // a custom core follows the CUDA version of the plugins, if any
  if (sel.core === CUSTOM_CORE && sel.compile.some((ep) => CUDA_EPS.includes(ep)) && !majors.size) majors.add(sel.cudaMajor);
  return majors;
}

function cloneSel(sel) {
  return { ...sel, plugins: [...sel.plugins], compile: [...sel.compile], fetch: sel.fetch ? [...sel.fetch] : null };
}

// Drop the plugins that conflict with the core (after the core changed); returns them
function dropConflictingPlugins(sel, cat) {
  const kept = [], dropped = [];
  for (const p of sel.plugins) {
    const trial = cloneSel(sel);
    trial.plugins = [...kept, p];
    (validate(trial, cat) ? dropped : kept).push(p);
  }
  sel.plugins = kept;
  return dropped;
}

// Every way to get each EP on the selected triple: [{ep, kind: "core"|"plugin"|"compile", pkg, form}]
function options(cat, triple) {
  const out = {};
  const add = (ep, opt) => (out[ep] = out[ep] || []).push({ ep, ...opt });
  for (const it of itemsFor(cat, triple)) {
    for (const [ep, form] of Object.entries(it.eps)) add(ep, { kind: it.kind, pkg: it.package, form, item: it });
  }
  for (const ep of cat.compile[triple] || []) add(ep, { kind: "compile", pkg: CUSTOM_CORE, form: "compile" });
  const order = Object.keys(cat.eps);
  return order.filter((ep) => out[ep]).map((ep) => [ep, out[ep]]);
}

function isSelected(sel, opt) {
  if (opt.kind === "plugin") return sel.plugins.includes(opt.pkg);
  if (opt.kind === "compile") return sel.core === CUSTOM_CORE && sel.compile.includes(opt.ep);
  return sel.core === opt.pkg;
}

// The selection after clicking an option; {sel, note} or {error}
function apply(sel, opt, cat) {
  const next = cloneSel(sel);
  let note = "";
  if (opt.kind === "plugin") {
    if (next.plugins.includes(opt.pkg)) next.plugins = next.plugins.filter((p) => p !== opt.pkg);
    else next.plugins.push(opt.pkg);
  } else if (opt.kind === "compile") {
    if (next.core !== CUSTOM_CORE) {
      next.core = CUSTOM_CORE;
      next.compile = [];
    }
    if (next.compile.includes(opt.ep)) {
      // removing an EP also removes the EPs that need it
      const needs = (ep) => (cat.requires[ep] || []).includes(opt.ep);
      next.compile = next.compile.filter((ep) => ep !== opt.ep && !needs(ep));
      if (!next.compile.length) next.core = sel.lastCore || defaultCore(cat, sel.triple);
    } else {
      next.compile.push(opt.ep, ...(cat.requires[opt.ep] || []).filter((r) => !next.compile.includes(r)));
    }
  } else {
    if (next.core === opt.pkg) return { sel: next, note };
    next.core = opt.pkg;
    next.lastCore = opt.pkg;
    next.compile = [];
  }
  if (next.core === CUSTOM_CORE && sel.core !== CUSTOM_CORE) next.lastCore = sel.core;
  if (opt.kind !== "plugin") {
    const dropped = dropConflictingPlugins(next, cat);
    if (dropped.length) note = `Removed ${dropped.join(", ")}: they conflict with the new core.`;
  }
  const err = validate(next, cat);
  return err ? { error: err } : { sel: next, note };
}

function defaultCore(cat, triple) {
  const cores = itemsFor(cat, triple).filter((it) => it.kind === "core");
  const def = cores.find((it) => it.package === "onnxruntime");
  return (def || cores[0] || {}).package || "";
}

function newSelection(cat, triple) {
  const core = defaultCore(cat, triple);
  return { triple, core, lastCore: core, compile: [], plugins: [], cudaMajor: "13", dev: true, fetch: null };
}

// Everything the request will contain
function summarize(sel, cat) {
  const eps = { ...coreEps(sel, cat) };
  const sources = {};
  for (const ep of Object.keys(eps)) sources[ep] = [`${eps[ep]} (${sel.core})`];
  let size = 0;
  if (sel.core !== CUSTOM_CORE) size += (findItem(cat, sel.core, sel.triple).asset || {}).size || 0;
  for (const p of sel.plugins) {
    const it = findItem(cat, p, sel.triple);
    size += it.asset.size || 0;
    for (const [ep, form] of Object.entries(it.eps)) (sources[ep] = sources[ep] || []).push(`${form} (${p})`);
  }
  const bundled = new Set(), fetchable = {}, system = new Set();
  for (const ep of Object.keys(sources)) {
    const tp = cat.third_party[ep] || {};
    if (tp.bundle) (tp.components || []).forEach((c) => bundled.add(c));
    const f = (tp.fetch || []).filter((x) => sel.dev || !x.endsWith(":dev"));
    if (f.length) fetchable[ep] = { parts: f, components: tp.components || [] };
    (tp.system || []).forEach((s) => system.add(s));
  }
  const wanted = sel.fetch;
  const fetched = new Set();
  for (const [ep, f] of Object.entries(fetchable)) if (!wanted || wanted.includes(ep)) f.components.forEach((c) => fetched.add(c));
  const majors = cudaMajors(sel, cat);
  return { sources, size, bundled: [...bundled], fetchable, fetched: [...fetched], system: [...system],
           cudaMajor: majors.size ? [...majors][0] : null };
}

function config(sel, cat) {
  const s = summarize(sel, cat);
  const custom = sel.core === CUSTOM_CORE;
  return {
    schema: 1,
    source: { type: cat.source.type, id: String(cat.source.id) },
    triple: sel.triple,
    core: custom ? null : sel.core,
    compile: custom ? sel.compile : null,
    plugins: sel.plugins,
    cuda_major: s.cudaMajor,
    dev: sel.dev,
    fetch: sel.fetch ? sel.fetch.filter((ep) => s.fetchable[ep]) : null,
  };
}

function issueUrl(sel, cat) {
  const cfg = config(sel, cat);
  const what = cfg.core || `custom core (${cfg.compile.join(", ")})`;
  const title = `[bundle] ${sel.triple}: ${[what, ...cfg.plugins].join(" + ")}`.slice(0, 200);
  const params = new URLSearchParams({ template: "bundle.yml", title, config: JSON.stringify(cfg, null, 1) });
  return `https://github.com/${cat.repository}/issues/new?${params}`;
}

if (typeof module !== "undefined") {
  module.exports = { platformOf, archOf, options, apply, validate, newSelection, summarize, config, issueUrl,
                     isSelected, CUSTOM_CORE };
}

// ---------------------------------------------------------------------------
// User interface
// ---------------------------------------------------------------------------

if (typeof document !== "undefined") {
  const $ = (sel, root = document) => root.querySelector(sel);
  const h = (tag, attrs = {}, ...children) => {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === false || v === null || v === undefined) continue;
      if (k === "class") el.className = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) if (c !== null && c !== undefined && c !== false) el.append(c);
    return el;
  };
  const mib = (n) => (n >= 1 << 30 ? `${(n / (1 << 30)).toFixed(1)} GiB` : `${Math.max(1, Math.round(n / (1 << 20)))} MiB`);

  let cat = null;
  let sel = null;
  let note = "";

  const FORM_LABEL = { builtin: "built-in", provider: "provider library", plugin: "plugin EP", compile: "compile" };

  function store() {
    try {
      localStorage.setItem("ortb-selection", JSON.stringify(sel));
    } catch (e) { /* storage unavailable */ }
  }

  function restore() {
    try {
      const saved = JSON.parse(localStorage.getItem("ortb-selection") || "null");
      if (saved && cat.triples.includes(saved.triple) && !validate(saved, cat)) return saved;
    } catch (e) { /* ignore */ }
    return null;
  }

  function setTriple(triple) {
    sel = newSelection(cat, triple);
    note = "";
    render();
  }

  function renderTarget() {
    const platforms = [...new Set(cat.triples.map(platformOf))];
    const current = platformOf(sel.triple);
    const archs = cat.triples.filter((t) => platformOf(t) === current);
    return h("section", { class: "card" },
      h("h2", {}, "1. Target"),
      h("div", { class: "row" },
        h("label", {}, "Platform",
          h("select", { onchange: (e) => setTriple(cat.triples.find((t) => platformOf(t) === e.target.value)) },
            platforms.map((p) => h("option", { value: p, selected: p === current }, p)))),
        h("label", {}, "Architecture",
          h("select", { onchange: (e) => setTriple(e.target.value) },
            archs.map((t) => h("option", { value: t, selected: t === sel.triple }, archOf(t))))),
        h("code", { class: "triple" }, sel.triple)));
  }

  function chip(opt) {
    const selected = isSelected(sel, opt);
    let disabled = "";
    if (!selected) {
      const r = apply(sel, opt, cat);
      if (r.error) disabled = r.error;
    }
    const pkgLabel = opt.kind === "compile" ? "custom core" : opt.pkg.replace(/^onnxruntime-/, "");
    const size = opt.item && opt.item.asset ? mib(opt.item.asset.size) : "";
    const cls = ["chip", selected && "on", disabled && "off", opt.kind === "plugin" ? "plugin" : "core",
                 opt.item && opt.item.experimental && "exp"].filter(Boolean).join(" ");
    const title = disabled ? `Not available with the current selection: ${disabled}` :
      opt.kind === "plugin" ? `Plugin EP package ${opt.pkg} (${size}); can be combined with any core` :
      opt.kind === "compile" ? "Compile a custom core with this in-tree EP (takes up to a few hours)" :
      `Core package ${opt.pkg} (${size}); selecting it replaces the current core`;
    return h("button", {
      type: "button", class: cls, title, "aria-pressed": selected ? "true" : "false", disabled: !!disabled,
      onclick: () => {
        const r = apply(sel, opt, cat);
        if (r.error) return;
        sel = r.sel;
        note = r.note;
        render();
      },
    }, h("span", { class: "form" }, FORM_LABEL[opt.form] || opt.form), h("span", { class: "pkg" }, pkgLabel),
       opt.item && opt.item.cuda_major ? h("span", { class: "tag" }, `CUDA ${opt.item.cuda_major}`) : null,
       opt.item && opt.item.experimental ? h("span", { class: "tag warn" }, "experimental") : null);
  }

  function renderEps() {
    const rows = options(cat, sel.triple).map(([ep, opts]) => {
      const info = cat.eps[ep];
      const active = opts.some((o) => isSelected(sel, o));
      return h("tr", { class: active ? "active" : "" },
        h("th", { scope: "row" }, h("span", { class: "ep" }, info.name), h("span", { class: "vendor" }, info.vendor)),
        h("td", {}, h("div", { class: "chips" }, opts.map(chip))));
    });
    const custom = sel.core === CUSTOM_CORE;
    const cudaCompile = custom && sel.compile.some((ep) => CUDA_EPS.includes(ep));
    const pluginMajors = new Set(sel.plugins.map((p) => findItem(cat, p, sel.triple).cuda_major).filter(Boolean));
    return h("section", { class: "card" },
      h("h2", {}, "2. Execution providers"),
      h("p", { class: "hint" },
        "Every row lists the ways to get an EP: inside a core package (built-in, or a provider library next to the core), ",
        "as a plugin EP package, or compiled into a custom core. Exactly one core is used; plugin EPs combine with any core. ",
        "Options that conflict with the current selection are disabled; hover them for the reason."),
      h("p", { class: "current" }, "Core: ",
        h("strong", {}, custom ? `custom core with ${sel.compile.join(", ")} (compiled for this request)` : sel.core),
        cudaCompile && !pluginMajors.size ? h("label", { class: "inline" }, " CUDA ",
          h("select", { onchange: (e) => { sel.cudaMajor = e.target.value; render(); } },
            ["12", "13"].map((v) => h("option", { value: v, selected: v === sel.cudaMajor }, `${v}.x`)))) : null),
      note ? h("p", { class: "note", role: "status" }, note) : null,
      h("div", { class: "table-wrap" }, h("table", { class: "eps" }, h("tbody", {}, rows))));
  }

  function renderOptions(s) {
    const fetchEps = Object.keys(s.fetchable);
    return h("section", { class: "card" },
      h("h2", {}, "3. Options"),
      h("label", { class: "check" },
        h("input", { type: "checkbox", checked: sel.dev, onchange: (e) => { sel.dev = e.target.checked; render(); } }),
        " Headers and link files of the backends (the bundle includes those it may redistribute; fetch-restricted downloads the others)"),
      fetchEps.length ? h("fieldset", {},
        h("legend", {}, "Fetched on your machine by fetch-restricted"),
        fetchEps.map((ep) => h("label", { class: "check" },
          h("input", { type: "checkbox", checked: !sel.fetch || sel.fetch.includes(ep), onchange: (e) => {
            const all = fetchEps.filter((x) => !sel.fetch || sel.fetch.includes(x));
            sel.fetch = e.target.checked ? [...new Set([...all, ep])] : all.filter((x) => x !== ep);
            render();
          } }),
          ` ${cat.eps[ep].name}: ${s.fetchable[ep].parts.map((p) => p.split(":")[1] === "dev" ? "headers / import libraries" : "runtime").join(", ")}`))) : null);
  }

  function renderSummary(s) {
    const err = validate(sel, cat);
    const comp = (c) => cat.components[c];
    const licenses = [...new Set([...s.bundled, ...s.fetched])];
    const cfg = config(sel, cat);
    const ack = $("#ack") && $("#ack").checked;
    const rows = Object.entries(s.sources).map(([ep, srcs]) =>
      h("tr", {}, h("td", {}, cat.eps[ep] ? cat.eps[ep].name : ep), h("td", {}, srcs.join("; "))));
    return h("section", { class: "card" },
      h("h2", {}, "4. Request"),
      h("div", { class: "table-wrap" }, h("table", { class: "summary" },
        h("thead", {}, h("tr", {}, h("th", {}, "EP"), h("th", {}, "Provided as"))), h("tbody", {}, rows))),
      h("p", {}, `Packages to download: ${mib(s.size)} (before third-party runtimes)`,
        sel.core === CUSTOM_CORE ? "; the custom core is compiled first, which takes up to a few hours." : "."),
      s.bundled.length ? h("p", {}, "Bundled third-party runtimes: ", s.bundled.map((c) => comp(c).name).join(", "), ".") : null,
      s.fetched.length ? h("p", {}, "Downloaded on your machine by fetch-restricted: ",
        s.fetched.map((c) => `${comp(c).name} (${comp(c).fetched})`).join("; "), ".") : null,
      s.system.length ? h("p", { class: "warn" }, "Install yourself: ",
        s.system.map((c) => `${comp(c).name}: ${comp(c).system}`).join("; "), ".") : null,
      licenses.length ? h("div", { class: "licenses" },
        h("h3", {}, "Third-party licenses"),
        h("ul", {}, licenses.map((c) => h("li", {}, h("a", { href: comp(c).license, target: "_blank", rel: "noopener" },
          comp(c).name))))) : null,
      h("label", { class: "check ack" },
        h("input", { type: "checkbox", id: "ack", checked: ack, onchange: () => render() }),
        licenses.length ? " I have read these licenses and accept them for the bundled and fetched files."
          : " I understand that the bundle is built from this repository's packages under their licenses."),
      err ? h("p", { class: "error", role: "alert" }, err) : null,
      h("div", { class: "actions" },
        h("a", { class: `button primary${err || !ack ? " disabled" : ""}`, href: err || !ack ? undefined : issueUrl(sel, cat),
                 target: "_blank", rel: "noopener", "aria-disabled": err || !ack ? "true" : "false" },
          "Create bundle request"),
        h("button", { type: "button", class: "button", onclick: (e) => {
          navigator.clipboard.writeText(JSON.stringify(cfg, null, 1)).then(() => { e.target.textContent = "Copied"; });
        } }, "Copy configuration")),
      h("p", { class: "hint" }, "The request opens a prefilled issue. Builds run only for collaborators of ",
        h("a", { href: `https://github.com/${cat.repository}` }, cat.repository),
        "; the bundle is linked in the issue when it is ready (download needs a GitHub login, kept for 90 days)."),
      h("details", {}, h("summary", {}, "Configuration JSON"), h("pre", {}, JSON.stringify(cfg, null, 1))));
  }

  function render() {
    store();
    const s = summarize(sel, cat);
    const app = $("#app");
    const focus = document.activeElement && document.activeElement.id;
    app.replaceChildren(renderTarget(), renderEps(), renderOptions(s), renderSummary(s));
    if (focus && document.getElementById(focus)) document.getElementById(focus).focus();
  }

  async function main() {
    try {
      const res = await fetch("catalog.json", { cache: "no-cache" });
      if (!res.ok) throw new Error(`catalog.json: HTTP ${res.status}`);
      cat = await res.json();
    } catch (e) {
      $("#app").replaceChildren(h("p", { class: "error" }, `Cannot load the catalog: ${e.message}`));
      return;
    }
    const src = cat.source;
    $("#source").replaceChildren(src
      ? h("span", {}, "Packages: ", h("a", { href: src.url }, `${src.type === "release" ? "release" : "build run"} ${src.id}`),
          ` · ${new Date(src.created_at).toLocaleDateString()}`)
      : h("span", { class: "warn" }, "No packages are published yet."));
    if (!src) return;
    const first = cat.triples.find((t) => t === "x86_64-pc-windows-msvc") || cat.triples[0];
    sel = restore() || newSelection(cat, first);
    render();
  }

  main();
}
