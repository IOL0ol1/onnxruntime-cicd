// Bundle configurator: pick a target and, for every execution provider, whether to leave it out, add its plugin EP
// package, or compile it into the core; then create a bundle request issue. The rules mirror make_plan in
// ci/bundle.py, which validates the request again.
"use strict";

// ---------------------------------------------------------------------------
// Selection logic (no DOM)
// ---------------------------------------------------------------------------

const CUDA_EPS = ["cuda", "tensorrt", "nv-tensorrt-rtx"];

const PLATFORMS = [
  ["Windows", (t) => /-windows-/.test(t) && !/^arm64(ec|x)-/.test(t)],
  ["Windows ARM64EC / ARM64X", (t) => /^arm64(ec|x)-/.test(t)],
  ["Linux", (t) => t.endsWith("-linux-gnu") && !t.startsWith("s390x")],
  ["Linux on IBM Z", (t) => t.startsWith("s390x")],
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
const WASM_MODES = {
  cpu: "CPU only",
  jsep: "JSEP: WebGPU and WebNN through JavaScript",
  webgpu: "WebGPU (native) and WebNN",
  "webgpu-jspi": "WebGPU (native) and WebNN, with JSPI instead of Asyncify",
  static: "Static library libonnxruntime_webassembly.a (CPU) for linking into your own WebAssembly build",
};

function platformOf(triple) {
  const p = PLATFORMS.find(([, test]) => test(triple));
  return p ? p[0] : triple;
}

function archOf(triple) {
  const a = triple.split("-")[0];
  return ARCHS[a] || a;
}

const platformInfo = (cat, triple) => cat.platforms[triple];
const pluginOptions = (cat, triple, ep) => cat.plugins.filter((p) => p.triple === triple && p.ep === ep);

// The EPs a target offers, in display order: {ep, core: info | null, plugins: [...]}
function epRows(cat, triple) {
  const info = platformInfo(cat, triple);
  return Object.keys(cat.eps)
    .filter((ep) => ep !== "cpu")
    .map((ep) => ({ ep, core: info.core_eps[ep] || null, plugins: pluginOptions(cat, triple, ep) }))
    .filter((r) => r.core || r.plugins.length);
}

function newSelection(cat, triple) {
  const info = platformInfo(cat, triple);
  const choices = {};
  for (const ep of info.default_core) choices[ep] = "core";
  return { triple, choices, link: {}, cudaMajor: "13", wasmMode: info.wasm_modes ? "cpu" : null, dev: true,
           fetch: null, bundle: [] };
}

function cloneSel(sel) {
  return { ...sel, choices: { ...sel.choices }, link: { ...sel.link }, fetch: sel.fetch ? [...sel.fetch] : null,
           bundle: [...sel.bundle] };
}

// The linkage choice of an EP in the core (null if it has none), and the form it gives the EP there
const linkOf = (sel, c, ep) => (c && c.link ? sel.link[ep] || c.link.default : null);
const coreForm = (sel, c, ep) => (c.link && (c.link.choices[linkOf(sel, c, ep)] || {}).form) || c.form;

const formatSize = (mb) => (mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${mb} MB`);

const coreEps = (sel, cat) => Object.keys(cat.eps).filter((ep) => sel.choices[ep] === "core");
const pluginEps = (sel, cat) => Object.keys(cat.eps).filter((ep) => sel.choices[ep] === "plugin");

function needsCuda(sel, cat) {
  return coreEps(sel, cat).some((ep) => CUDA_EPS.includes(ep)) ||
    pluginEps(sel, cat).some((ep) => pluginOptions(cat, sel.triple, ep).some((p) => p.cuda_major));
}

// The plugin package of an EP for the selection's CUDA version
function pluginFor(sel, cat, ep) {
  const options = pluginOptions(cat, sel.triple, ep);
  return options.find((p) => !p.cuda_major || p.cuda_major === sel.cudaMajor) || null;
}

// Validate a selection; returns an error message or "".
function validate(sel, cat) {
  const info = platformInfo(cat, sel.triple);
  if (!info) return `unknown target ${sel.triple}`;
  const name = (ep) => cat.eps[ep].name;
  if (info.wasm_modes) {
    if (!info.wasm_modes[sel.wasmMode]) return "select a WebAssembly build mode";
    if (coreEps(sel, cat).length) return "the WebAssembly build mode decides the EPs of the core";
  }
  for (const ep of coreEps(sel, cat)) {
    if (!info.core_eps[ep]) return `${name(ep)} cannot be compiled into the core for ${sel.triple}`;
    const missing = (cat.requires[ep] || []).filter((r) => sel.choices[r] !== "core");
    if (missing.length) return `${name(ep)} in the core needs ${missing.map(name).join(", ")} in the core as well`;
    const c = info.core_eps[ep];
    if (c.link && !c.link.choices[linkOf(sel, c, ep)]) return `${name(ep)}: unknown ${c.link.label} choice`;
    const cuda = c.cuda;
    if (CUDA_EPS.includes(ep) && cuda && !cuda.includes(sel.cudaMajor)) {
      return `${name(ep)} in the core is built with CUDA ${cuda.join(" / ")} for this target, not CUDA ${sel.cudaMajor}`;
    }
  }
  const owner = {};
  const coreLibs = coreEps(sel, cat).filter((ep) => coreForm(sel, info.core_eps[ep], ep) === "provider")
    .map((ep) => info.core_eps[ep].lib);
  const parts = [["onnxruntime", ["onnxruntime", ...coreLibs]]];
  for (const ep of pluginEps(sel, cat)) {
    const options = pluginOptions(cat, sel.triple, ep);
    if (!options.length) return `${name(ep)} has no plugin EP package for ${sel.triple}`;
    const p = pluginFor(sel, cat, ep);
    if (!p) return `the ${name(ep)} plugin EP is built with CUDA ${options.map((o) => o.cuda_major).join(" / ")}, not CUDA ${sel.cudaMajor}`;
    parts.push([p.package, p.libs]);
  }
  for (const [pkg, libs] of parts) {
    for (const lib of libs) {
      if (owner[lib]) return `${pkg} and ${owner[lib]} both contain ${lib}`;
      owner[lib] = pkg;
    }
  }
  return "";
}

// The selection after setting an EP to "core", "plugin" or null (off); {sel, note} or {error}
function setChoice(sel, cat, ep, form) {
  const next = cloneSel(sel);
  const notes = [];
  const name = (e) => cat.eps[e].name;
  if (form) next.choices[ep] = form;
  else delete next.choices[ep];
  // TensorRT in the core brings the CUDA EP into the core (ORT builds them together)
  for (const [e, reqs] of Object.entries(cat.requires)) {
    if (next.choices[e] !== "core") continue;
    for (const r of reqs) {
      if (next.choices[r] === "core") continue;
      if (r === ep) {  // the user moved the required EP out of the core: move the dependent EP out as well
        const alt = pluginOptions(cat, next.triple, e).length ? "plugin" : null;
        if (alt) next.choices[e] = alt;
        else delete next.choices[e];
        notes.push(`${name(e)} ${alt ? "switched to its plugin EP" : "removed"}: in the core it needs ${name(r)} in the core.`);
      } else {
        next.choices[r] = "core";
        notes.push(`${name(r)} compiled into the core as well: ${name(e)} in the core needs it.`);
      }
    }
  }
  // a linkage choice applies to an EP in the core only: a plugin EP is a separate library
  const coreInfo = platformInfo(cat, next.triple).core_eps;
  for (const e of Object.keys(next.link)) {
    if (next.choices[e] === "core") continue;
    const c = coreInfo[e];
    if (c && c.link && next.link[e] !== c.link.default && next.choices[e] === "plugin") notes.push(`${name(e)}: ${c.link.plugin}.`);
    delete next.link[e];
  }
  // follow the CUDA version the selection can be built with
  if (needsCuda(next, cat) && validate(next, cat)) {
    for (const major of ["13", "12"]) {
      const trial = cloneSel(next);
      trial.cudaMajor = major;
      if (major !== next.cudaMajor && !validate(trial, cat)) {
        next.cudaMajor = major;
        notes.push(`Switched to CUDA ${major}.`);
        break;
      }
    }
  }
  const err = validate(next, cat);
  return err ? { error: err } : { sel: next, note: notes.join(" ") };
}

// Every available EP in its default form: the plugin EP where there is one, otherwise compiled into the core
function selectAll(sel, cat) {
  let next = cloneSel(sel);
  for (const row of epRows(cat, sel.triple)) {
    const form = row.plugins.length ? "plugin" : "core";
    const r = setChoice(next, cat, row.ep, form);
    if (r.sel) next = r.sel;
  }
  return next;
}

// Everything the request will contain
function summarize(sel, cat) {
  const info = platformInfo(cat, sel.triple);
  const sources = { cpu: ["built-in (core)"] };
  const long = [];
  if (info.wasm_modes && info.wasm_modes[sel.wasmMode]) {
    for (const ep of info.wasm_modes[sel.wasmMode].eps) sources[ep] = ["built-in (core)"];
  }
  let coreHeavy = false;
  for (const ep of coreEps(sel, cat)) {
    const c = info.core_eps[ep];
    if (!c) continue;
    const link = linkOf(sel, c, ep);
    sources[ep] = [`${coreForm(sel, c, ep) === "provider" ? "provider library" : "built-in"} (core` +
                   `${link ? `; ${c.link.label}: ${link === "shared" ? "dynamic" : "static"}` : ""})`];
    coreHeavy = coreHeavy || c.heavy;
  }
  if (coreHeavy) long.push("the core");
  const plugins = [];
  for (const ep of pluginEps(sel, cat)) {
    const p = pluginFor(sel, cat, ep);
    if (!p) continue;
    plugins.push(p);
    if (p.heavy) long.push(p.package);
    for (const e of Object.keys(p.eps)) (sources[e] = sources[e] || []).push(`plugin EP (${p.package})`);
  }
  // optional: EPs whose runtimes the bundle may contain (the rest is downloaded by fetch-restricted)
  const bundled = new Set(), optional = {}, fetchable = {}, system = new Set();
  let bundleSize = 0;
  for (const ep of Object.keys(sources)) {
    const tp = cat.third_party[ep] || {};
    const comps = tp.components || [];
    const inBundle = !!tp.optional && sel.bundle.includes(ep);
    if (tp.optional) optional[ep] = { size: tp.size_mb, components: comps, bundle: inBundle };
    if (inBundle) {
      comps.forEach((c) => bundled.add(c));
      bundleSize += tp.size_mb;
    }
    const parts = [...(tp.optional && !inBundle ? tp.optional : []),
                   ...(tp.fetch || []).filter((x) => sel.dev || !x.endsWith(":dev"))];
    if (parts.length) fetchable[ep] = { parts, components: comps };
    (tp.system || []).forEach((s) => system.add(s));
  }
  const fetched = {};  // component -> Set of parts ("runtime", "dev")
  for (const [ep, f] of Object.entries(fetchable)) {
    if (sel.fetch && !sel.fetch.includes(ep)) continue;
    for (const c of f.components) {
      for (const p of f.parts) (fetched[c] = fetched[c] || new Set()).add(p.endsWith(":dev") ? "dev" : "runtime");
    }
  }
  return { sources, plugins, long, bundled: [...bundled], bundleSize, optional, fetchable, fetched,
           system: [...system], cudaMajor: needsCuda(sel, cat) ? sel.cudaMajor : null };
}

function config(sel, cat) {
  const s = summarize(sel, cat);
  const info = platformInfo(cat, sel.triple);
  const eps = {};
  for (const ep of Object.keys(cat.eps)) if (sel.choices[ep]) eps[ep] = sel.choices[ep];
  return {
    schema: 4,
    triple: sel.triple,
    eps,
    cuda_major: s.cudaMajor,
    wasm_mode: platformInfo(cat, sel.triple).wasm_modes ? sel.wasmMode : null,
    dev: sel.dev,
    fetch: sel.fetch ? sel.fetch.filter((ep) => s.fetchable[ep]) : null,
    bundle: sel.bundle.filter((ep) => s.optional[ep]),
    link: Object.fromEntries(coreEps(sel, cat).filter((ep) => (info.core_eps[ep] || {}).link)
      .map((ep) => [ep, linkOf(sel, info.core_eps[ep], ep)])),
  };
}

function issueUrl(sel, cat) {
  const cfg = config(sel, cat);
  const core = ["cpu", ...coreEps(sel, cat)].map((ep) => cat.eps[ep].name);
  const parts = [cfg.wasm_mode ? `wasm ${cfg.wasm_mode}` : `core ${core.join("+")}`,
                 ...pluginEps(sel, cat).map((ep) => `${cat.eps[ep].name} plugin`)];
  const title = `[bundle] ${sel.triple}: ${parts.join(", ")}`.slice(0, 200);
  const params = new URLSearchParams({ template: "bundle.yml", title, config: JSON.stringify(cfg, null, 1) });
  return `https://github.com/${cat.repository}/issues/new?${params}`;
}


// ---------------------------------------------------------------------------
// User interface
// ---------------------------------------------------------------------------

if (typeof document !== "undefined") {
  const $ = (q, root = document) => root.querySelector(q);
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

  let cat = null;
  let sel = null;
  let note = "";

  function store() {
    try {
      localStorage.setItem("ortb-selection-v4", JSON.stringify(sel));
    } catch (e) { /* storage unavailable */ }
  }

  function restore() {
    try {
      const saved = JSON.parse(localStorage.getItem("ortb-selection-v4") || "null");
      if (saved && cat.platforms[saved.triple] && !validate(saved, cat)) return saved;
    } catch (e) { /* ignore */ }
    return null;
  }

  function update(next, msg = "") {
    sel = next;
    note = msg;
    render();
  }

  function renderTarget() {
    const platforms = [...new Set(cat.triples.map(platformOf))];
    const current = platformOf(sel.triple);
    const archs = cat.triples.filter((t) => platformOf(t) === current);
    const info = platformInfo(cat, sel.triple);
    return h("section", { class: "card" },
      h("h2", {}, "1. Target"),
      h("div", { class: "row" },
        h("label", {}, "Platform",
          h("select", { onchange: (e) => update(newSelection(cat, cat.triples.find((t) => platformOf(t) === e.target.value))) },
            platforms.map((p) => h("option", { value: p, selected: p === current }, p)))),
        h("label", {}, "Architecture",
          h("select", { onchange: (e) => update(newSelection(cat, e.target.value)) },
            archs.map((t) => h("option", { value: t, selected: t === sel.triple }, archOf(t))))),
        h("code", { class: "triple" }, sel.triple)),
      info.note || info.experimental ? h("p", { class: "hint" }, info.note,
        info.experimental ? h("span", { class: "tag warn" }, " experimental target: builds may fail") : null) : null);
  }

  function choiceButton(row, form, label, detail, tags) {
    const selected = (sel.choices[row.ep] || null) === form;
    let disabled = "";
    if (form === "plugin" && !row.plugins.length) disabled = "no plugin EP package for this target";
    else if (form === "core" && !row.core) disabled = "cannot be compiled into the core for this target";
    else if (!selected) {
      const r = setChoice(sel, cat, row.ep, form);
      if (r.error) disabled = r.error;
    }
    return h("button", {
      type: "button", class: `choice ${form || "off"}${selected ? " on" : ""}`, disabled: !!disabled,
      "aria-pressed": selected ? "true" : "false",
      title: disabled ? `Not available: ${disabled}` : detail,
      onclick: () => {
        const r = setChoice(sel, cat, row.ep, form);
        if (!r.error) update(r.sel, r.note);
      },
    }, h("span", { class: "label" }, label), tags.filter(Boolean).map((t) => h("span", { class: `tag${t.warn ? " warn" : ""}` }, t.text)));
  }

  function renderEps() {
    const info = platformInfo(cat, sel.triple);
    if (info.wasm_modes) {
      return h("section", { class: "card" },
        h("h2", {}, "2. WebAssembly build mode"),
        h("p", { class: "hint" }, "WebAssembly builds are exclusive build modes, so the EPs come with the mode: choose one."),
        h("div", { class: "modes" }, Object.keys(info.wasm_modes).map((m) =>
          h("label", { class: "check" },
            h("input", { type: "radio", name: "wasm-mode", checked: sel.wasmMode === m,
                         onchange: () => update({ ...cloneSel(sel), wasmMode: m }) }),
            ` ${WASM_MODES[m] || m}`))));
    }
    const rows = epRows(cat, sel.triple).map((row) => {
      const p = row.plugins.length ? (pluginFor(sel, cat, row.ep) || row.plugins[0]) : null;
      const c = row.core;
      const ep = cat.eps[row.ep];
      return h("tr", { class: sel.choices[row.ep] ? "active" : "" },
        h("th", { scope: "row" }, h("span", { class: "ep" }, ep.name), h("span", { class: "vendor" }, ep.vendor)),
        h("td", {}, h("div", { class: "choices" },
          choiceButton(row, null, "Off", "Leave this EP out", []),
          choiceButton(row, "plugin", "Plugin EP",
            p ? `Add the plugin EP package ${p.package} (a separate library you register at run time)` : "",
            [p && { text: p.package.replace(/^onnxruntime-/, "") },
             row.plugins.length > 1 && row.plugins.some((o) => o.cuda_major) && { text: `CUDA ${row.plugins.map((o) => o.cuda_major).join("/")}` },
             p && p.heavy && { text: "long build" }, p && p.experimental && { text: "experimental", warn: true }]),
          choiceButton(row, "core", "In core",
            c ? `Compile into the core library (${c.form === "provider" ? "provider library next to it" : "built-in"})${c.note ? ": " + c.note : ""}` : "",
            [c && { text: coreForm(sel, c, row.ep) === "provider" ? "provider library" : "built-in" },
             c && c.cuda && { text: `CUDA ${c.cuda.join("/")}` },
             c && c.heavy && { text: "long build" }, c && c.experimental && { text: "experimental", warn: true }])),
          c && c.link && sel.choices[row.ep] === "core" ? h("label", { class: "inline" }, `${c.link.label} `,
            h("select", { onchange: (e) => { const next = cloneSel(sel); next.link[row.ep] = e.target.value; update(next); } },
              Object.entries(c.link.choices).map(([k, v]) => h("option", { value: k, selected: k === linkOf(sel, c, row.ep) },
                `${k === "shared" ? "dynamic" : "static"}: ${v.text}`)))) : null,
          c && c.link && sel.choices[row.ep] === "plugin" ? h("p", { class: "ep-note" }, `${c.link.label}: ${c.link.plugin}`) : null,
          c && c.note && sel.choices[row.ep] === "core" ? h("p", { class: "ep-note" }, c.note) : null));
    });
    const cuda = needsCuda(sel, cat);
    return h("section", { class: "card" },
      h("h2", {}, "2. Execution providers"),
      h("p", { class: "hint" },
        "For every EP choose whether to leave it out, add its plugin EP package (a separate library, built once and ",
        "reused by every bundle), or compile it into the core library. CPU is always in the core. Options that do not ",
        "work with the current selection are disabled; hover them for the reason."),
      h("div", { class: "toolbar" },
        h("button", { type: "button", class: "button", onclick: () => update(selectAll(sel, cat), "Selected every EP this target offers (plugin EPs where there are any).") }, "Select all"),
        h("button", { type: "button", class: "button", onclick: () => update(newSelection(cat, sel.triple)) }, "Reset"),
        cuda ? h("label", { class: "inline" }, "CUDA ",
          h("select", { onchange: (e) => {
            const trial = { ...cloneSel(sel), cudaMajor: e.target.value };
            update(trial, validate(trial, cat) ? `Not possible with CUDA ${e.target.value}: ${validate(trial, cat)}` : "");
          } }, ["12", "13"].map((v) => h("option", { value: v, selected: v === sel.cudaMajor }, `${v}.x`)))) : null),
      note ? h("p", { class: "note", role: "status" }, note) : null,
      h("div", { class: "table-wrap" }, h("table", { class: "eps" }, h("tbody", {}, rows))));
  }

  function renderOptions(s) {
    const fetchEps = Object.keys(s.fetchable);
    const optionalEps = Object.keys(s.optional);
    const setBundle = (ep, on) => {
      const next = cloneSel(sel);
      next.bundle = on ? [...new Set([...next.bundle, ep])] : next.bundle.filter((x) => x !== ep);
      update(next);
    };
    return h("section", { class: "card" },
      h("h2", {}, "3. Options"),
      h("label", { class: "check" },
        h("input", { type: "checkbox", checked: sel.dev, onchange: (e) => update({ ...cloneSel(sel), dev: e.target.checked }) }),
        " Headers and link files of the backends (the bundle includes those it may redistribute; fetch-restricted downloads the others)"),
      optionalEps.length ? h("fieldset", {},
        h("legend", {}, "Third-party runtimes: in the bundle or downloaded on your machine"),
        h("p", { class: "hint" }, "Vendor runtimes that may be redistributed with an application, and large open-source ",
          "libraries. In the bundle they are ready to use; otherwise fetch-restricted downloads them from their vendors. ",
          "Sizes are estimates (unpacked)."),
        optionalEps.map((ep) => h("label", { class: "inline" },
          `${cat.eps[ep].name} (${s.optional[ep].components.map((c) => cat.components[c].name).join(", ")}) `,
          h("select", { onchange: (e) => setBundle(ep, e.target.value === "bundle") },
            h("option", { value: "fetch", selected: !s.optional[ep].bundle }, "Download on your machine"),
            h("option", { value: "bundle", selected: s.optional[ep].bundle },
              `In the bundle (about ${formatSize(s.optional[ep].size)})`)))),
        s.bundleSize ? h("p", { class: "hint" }, `In the bundle: about ${formatSize(s.bundleSize)} in total.`) : null) : null,
      fetchEps.length ? h("fieldset", {},
        h("legend", {}, "Fetched on your machine by fetch-restricted"),
        fetchEps.map((ep) => h("label", { class: "check" },
          h("input", { type: "checkbox", checked: !sel.fetch || sel.fetch.includes(ep), onchange: (e) => {
            const all = fetchEps.filter((x) => !sel.fetch || sel.fetch.includes(x));
            update({ ...cloneSel(sel), fetch: e.target.checked ? [...new Set([...all, ep])] : all.filter((x) => x !== ep) });
          } }),
          ` ${cat.eps[ep].name}: ${s.fetchable[ep].parts.map((p) => p.split(":")[1] === "dev" ? "headers / import libraries" : "runtime").join(", ")}`))) : null);
  }

  function renderSummary(s) {
    const err = validate(sel, cat);
    const comp = (c) => cat.components[c];
    const licenses = [...new Set([...s.bundled, ...Object.keys(s.fetched)])];
    const cfg = config(sel, cat);
    const ack = $("#ack") && $("#ack").checked;
    const rows = Object.entries(s.sources).map(([ep, srcs]) =>
      h("tr", {}, h("td", {}, cat.eps[ep] ? cat.eps[ep].name : ep), h("td", {}, srcs.join("; "))));
    return h("section", { class: "card" },
      h("h2", {}, "4. Request"),
      h("div", { class: "table-wrap" }, h("table", { class: "summary" },
        h("thead", {}, h("tr", {}, h("th", {}, "EP"), h("th", {}, "Provided as"))), h("tbody", {}, rows))),
      h("p", {}, "Every package is built for the request unless an earlier request built it from the same sources in ",
        "the last 90 days (a core is reused for the same set of EPs). ",
        s.long.length ? `First builds of ${s.long.join(", ")} take up to a few hours.` : "This selection builds in about an hour or less."),
      s.bundled.length ? h("p", {}, `Bundled third-party runtimes (about ${formatSize(s.bundleSize)}): `,
        s.bundled.map((c) => comp(c).name).join(", "), ".") : null,
      Object.keys(s.fetched).length ? h("p", {}, "Downloaded on your machine by fetch-restricted: ",
        Object.entries(s.fetched).map(([c, parts]) => `${comp(c).name} (${[...parts].map((p) => comp(c)[p]).join(", ")})`)
          .join("; "), ".") : null,
      s.system.length ? h("p", { class: "warn" }, "Install yourself: ",
        s.system.map((c) => `${comp(c).name}: ${comp(c).system}`).join("; "), ".") : null,
      licenses.length ? h("div", { class: "licenses" },
        h("h3", {}, "Third-party licenses"),
        h("ul", {}, licenses.map((c) => h("li", {}, h("a", { href: comp(c).license, target: "_blank", rel: "noopener" },
          comp(c).name))))) : null,
      h("label", { class: "check ack" },
        h("input", { type: "checkbox", id: "ack", checked: ack, onchange: () => render() }),
        licenses.length ? " I have read these licenses and accept them for the bundled and fetched files."
          : " I understand that the bundle is built from this repository's sources under their licenses."),
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
    const focus = document.activeElement && document.activeElement.id;
    $("#app").replaceChildren(renderTarget(), renderEps(), renderOptions(s), renderSummary(s));
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
    $("#source").replaceChildren(h("span", {}, "Build recipes: commit ", h("a", { href: src.url }, src.id),
      ` · ${new Date(src.created_at).toLocaleDateString()}`));
    const first = cat.triples.find((t) => t === "x86_64-pc-windows-msvc") || cat.triples[0];
    sel = restore() || newSelection(cat, first);
    render();
  }

  main();
}
