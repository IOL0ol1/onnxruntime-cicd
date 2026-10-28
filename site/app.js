// Bundle configurator: pick a target and, for every execution provider, whether to leave it out, add its plugin EP
// package, or compile it into the core; then create a bundle request issue. The rules mirror make_plan in
// ci/bundle.py, which validates the request again.
"use strict";

// ---------------------------------------------------------------------------
// Selection logic (no DOM)
// ---------------------------------------------------------------------------

const CUDA_EPS = ["cuda", "tensorrt", "nv-tensorrt-rtx"];
const CUDA_MAJORS = ["12", "13"];

const PLATFORMS = [
  ["Windows", (t) => /-windows-/.test(t) && !/^arm64(ec|x)-/.test(t)],
  ["Windows ARM64EC / ARM64X", (t) => /^arm64(ec|x)-/.test(t)],
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
const ARCHS = { x86_64: "x64", aarch64: "ARM64", armv7: "ARMv7", i686: "x86", wasm32: "wasm32", arm64ec: "ARM64EC",
                arm64x: "ARM64X" };
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
const epName = (cat, ep) => (cat.eps[ep] ? cat.eps[ep].name : ep);
// The third-party files of an EP on an OS (ci/catalog.py third_party_of)
function thirdPartyOf(cat, ep, os) {
  const { os: overrides, ...tp } = cat.third_party[ep] || {};
  return { ...tp, ...((overrides || {})[os] || {}) };
}

// The EPs a target offers, in display order (the ones it offers in the core only first, then the catalog order):
// {ep, core: info | null, plugins: [...]}
function epRows(cat, triple) {
  const info = platformInfo(cat, triple);
  const coreOnly = (r) => Number(!!r.core && !r.plugins.length);
  return Object.keys(cat.eps)
    .filter((ep) => ep !== "cpu")
    .map((ep) => ({ ep, core: info.core_eps[ep] || null, plugins: pluginOptions(cat, triple, ep) }))
    .filter((r) => r.core || r.plugins.length)
    .sort((a, b) => coreOnly(b) - coreOnly(a));
}

// The versions a plugin EP is offered in on a target (ci/catalog.py PLUGIN_VARIANTS), the default first:
// [{id, label, note}]; empty when it is one package
function variantsOf(cat, triple, ep) {
  const seen = new Map();
  for (const p of pluginOptions(cat, triple, ep)) {
    if (p.variant && !seen.has(p.variant)) seen.set(p.variant, { id: p.variant, label: p.variant_label, note: p.variant_note });
  }
  const preferred = (cat.default_variants || {})[ep];
  return [...seen.values()].sort((a, b) => (b.id === preferred) - (a.id === preferred));
}
const variantOf = (sel, cat, ep) => sel.variants[ep] || (cat.default_variants || {})[ep] || null;

// The AMD GPU targets MIGraphX can be built for on the OS of a target
const amdTargetsFor = (cat, info) => Object.keys(cat.gpu.amd.targets).filter((t) => cat.gpu.amd.targets[t].os.includes(info.os));

function newSelection(cat, triple) {
  const info = platformInfo(cat, triple);
  const choices = {};
  for (const ep of info.default_core) choices[ep] = "core";
  const amd = amdTargetsFor(cat, info);
  return { triple, choices, link: {}, variants: {}, amdHip: false, cudaMajor: "13", wasmMode: info.wasm_modes ? "cpu" : null,
           cudaArchs: [...cat.gpu.cuda.default.archs], cudaPtx: cat.gpu.cuda.default.ptx,
           amdTargets: cat.gpu.amd.default.filter((t) => amd.includes(t)) };
}

function cloneSel(sel) {
  return { ...sel, choices: { ...sel.choices }, link: { ...sel.link }, variants: { ...sel.variants },
           cudaArchs: [...sel.cudaArchs], amdTargets: [...sel.amdTargets] };
}

// The linkage choice of an EP in the core (null if it has none), and the form it gives the EP there
const linkOf = (sel, c, ep) => (c && c.link ? sel.link[ep] || c.link.default : null);
const coreForm = (sel, c, ep) => (c.link && (c.link.choices[linkOf(sel, c, ep)] || {}).form) || c.form;

const formatSize = (mb) => (mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${mb} MB`);
const capitalize = (s) => s.charAt(0).toUpperCase() + s.slice(1);

const coreEps = (sel, cat) => Object.keys(cat.eps).filter((ep) => sel.choices[ep] === "core");
const pluginEps = (sel, cat) => Object.keys(cat.eps).filter((ep) => sel.choices[ep] === "plugin");

function needsCuda(sel, cat) {
  return coreEps(sel, cat).some((ep) => CUDA_EPS.includes(ep)) ||
    pluginEps(sel, cat).some((ep) => pluginOptions(cat, sel.triple, ep).some((p) => p.cuda_major));
}

// The plugin package of an EP for the selection's CUDA version, the EP's version and (AMDGPU) the HIP backend
function pluginFor(sel, cat, ep) {
  const variant = variantOf(sel, cat, ep);
  return pluginOptions(cat, sel.triple, ep).find((p) => (!p.cuda_major || p.cuda_major === sel.cudaMajor) &&
    (!p.variant || p.variant === variant) && (p.with_hip === undefined || p.with_hip === !!sel.amdHip)) || null;
}

// Whether the plugin EP version of a selection comes with or without the HIP backend (the AMDGPU version)
const hipChoice = (sel, cat, ep) => pluginOptions(cat, sel.triple, ep)
  .some((p) => p.with_hip !== undefined && p.variant === variantOf(sel, cat, ep));

// The GPU architecture settings ("cuda", "amd") of the packages of a selection that compile device code
function gpuKinds(sel, cat) {
  const info = platformInfo(cat, sel.triple);
  const kinds = new Set();
  for (const ep of coreEps(sel, cat)) for (const k of (info.core_eps[ep] || {}).gpu || []) kinds.add(k);
  for (const ep of pluginEps(sel, cat)) for (const k of (pluginFor(sel, cat, ep) || {}).gpu || []) kinds.add(k);
  return kinds;
}

const bySm = (a, b) => Number(a) - Number(b);

// How a GPU of SM version sm runs the CUDA kernels of a selection: its own ("native"), those of an older SM version of
// the same major version ("compat"; not for the arch-specific kernels ORT compiles for sm_90 and newer), the PTX of the
// newest one, which the driver compiles when a session starts ("ptx"), or not at all ("none")
function cudaCoverage(sm, archs, ptx, exact) {
  if (archs.includes(sm)) return { kind: "native" };
  const major = (a) => Math.floor(Number(a) / 10);
  const older = archs.filter((a) => !exact.includes(a) && major(a) === major(sm) && Number(a) < Number(sm)).sort(bySm);
  if (older.length) return { kind: "compat", from: older[older.length - 1] };
  const newest = [...archs].sort(bySm).pop();
  if (ptx && newest && Number(newest) < Number(sm)) return { kind: "ptx", from: newest };
  return { kind: "none" };
}

// Validate a selection; returns an error message or "". The page only offers valid selections, so this is the check
// that setChoice uses to decide what a click does, and a guard for selections restored from storage.
function validate(sel, cat) {
  const info = platformInfo(cat, sel.triple);
  if (!info) return `unknown target ${sel.triple}`;
  const name = (ep) => epName(cat, ep);
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
    if (CUDA_EPS.includes(ep) && c.cuda && !c.cuda.includes(sel.cudaMajor)) {
      return `${name(ep)} in the core is built with CUDA ${c.cuda.join(" / ")} for this target, not CUDA ${sel.cudaMajor}`;
    }
  }
  const owner = {};
  const coreLibs = coreEps(sel, cat).filter((ep) => coreForm(sel, info.core_eps[ep], ep) === "provider")
    .map((ep) => info.core_eps[ep].lib);
  const parts = [["onnxruntime", ["onnxruntime", ...coreLibs]]];
  for (const ep of pluginEps(sel, cat)) {
    const options = pluginOptions(cat, sel.triple, ep);
    if (!options.length) return `${name(ep)} has no plugin EP package for ${sel.triple}`;
    const variants = variantsOf(cat, sel.triple, ep);
    if (variants.length && !variants.some((v) => v.id === variantOf(sel, cat, ep))) {
      return `the ${name(ep)} plugin EP has no ${variantOf(sel, cat, ep)} version for ${sel.triple}`;
    }
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
  const kinds = gpuKinds(sel, cat);
  if (kinds.has("cuda") && !sel.cudaArchs.length) return "select at least one CUDA GPU architecture";
  if (kinds.has("amd")) {
    if (!sel.amdTargets.length) return "select at least one AMD GPU target";
    const bad = sel.amdTargets.filter((t) => !amdTargetsFor(cat, info).includes(t));
    if (bad.length) return `MIGraphX cannot be built for ${bad.join(", ")} on this target`;
  }
  return "";
}

// The CUDA major versions a selection can be built with
const cudaChoices = (sel, cat) => CUDA_MAJORS.filter((m) => !validate({ ...cloneSel(sel), cudaMajor: m }, cat));

// Why a selection can be built with one CUDA major version only: the first EP that is built with that one only
function cudaLimit(sel, cat, major) {
  const info = platformInfo(cat, sel.triple);
  for (const ep of coreEps(sel, cat)) {
    const c = info.core_eps[ep];
    if (CUDA_EPS.includes(ep) && c && c.cuda && c.cuda.every((m) => m === major)) {
      return `${epName(cat, ep)} in the core is built with CUDA ${major} only for this target`;
    }
  }
  for (const ep of pluginEps(sel, cat)) {
    const majors = pluginOptions(cat, sel.triple, ep).map((p) => p.cuda_major).filter(Boolean);
    if (majors.length && majors.every((m) => m === major)) return `the ${epName(cat, ep)} plugin EP is built with CUDA ${major} only`;
  }
  return "";
}

// The selection after setting an EP to "core", "plugin" or null (off): {sel, effects} or {error}. effects lists what
// else changes: [{ep (null for the CUDA version), tag (short, for the button), text}]
function setChoice(sel, cat, ep, form) {
  const next = cloneSel(sel);
  const effects = [];
  const name = (e) => epName(cat, e);
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
        effects.push({ ep: e, tag: `${name(e)} → ${alt ? "plugin" : "off"}`,
                       text: `${name(e)} ${alt ? "switched to its plugin EP" : "left out"}: in the core it needs ${name(r)} in the core.` });
      } else {
        next.choices[r] = "core";
        effects.push({ ep: r, tag: `+ ${name(r)} in core`, text: `${name(r)} compiled into the core as well: ${name(e)} in the core needs it.` });
      }
    }
  }
  // a linkage choice applies to an EP in the core only: a plugin EP is a separate library
  for (const e of Object.keys(next.link)) if (next.choices[e] !== "core") delete next.link[e];
  // follow the CUDA version the selection can be built with
  if (needsCuda(next, cat) && validate(next, cat)) {
    const majors = cudaChoices(next, cat);
    const major = majors.includes("13") ? "13" : majors[0];
    if (major) {
      effects.push({ ep: null, tag: `CUDA ${major}`, text: `Switched to CUDA ${major}: the selection cannot be built with CUDA ${next.cudaMajor}.` });
      next.cudaMajor = major;
    }
  }
  const err = validate(next, cat);
  return err ? { error: err } : { sel: next, effects };
}

// The selection for another target that keeps what that target offers of sel: {sel, dropped: [descriptions]}
function retarget(sel, cat, triple) {
  const info = platformInfo(cat, triple);
  let next = newSelection(cat, triple);
  next.cudaMajor = sel.cudaMajor;
  next.cudaArchs = sel.cudaArchs.filter((a) => cat.gpu.cuda.archs[a]);
  if (!next.cudaArchs.length) next.cudaArchs = [...cat.gpu.cuda.default.archs];
  next.cudaPtx = sel.cudaPtx;
  const amd = sel.amdTargets.filter((t) => amdTargetsFor(cat, info).includes(t));
  if (amd.length) next.amdTargets = amd;
  next.variants = { ...sel.variants };
  next.amdHip = !!sel.amdHip;
  // an EP the previous target offered and the user left out stays out
  const before = cat.platforms[sel.triple] ? new Set(epRows(cat, sel.triple).map((r) => r.ep)) : new Set();
  for (const ep of Object.keys(next.choices)) if (before.has(ep) && !sel.choices[ep]) delete next.choices[ep];
  const rows = Object.fromEntries(epRows(cat, triple).map((r) => [r.ep, r]));
  const dropped = [];
  for (const ep of Object.keys(cat.eps)) {
    const form = sel.choices[ep];
    if (!form || next.choices[ep] === form) continue;
    const row = rows[ep];
    const what = `${epName(cat, ep)} ${form === "core" ? "in the core" : "plugin EP"}`;
    if (!row || !(form === "core" ? row.core : row.plugins.length)) {
      dropped.push(what);
      continue;
    }
    if (form === "plugin" && !variantsOf(cat, triple, ep).some((v) => v.id === next.variants[ep])) delete next.variants[ep];
    const r = setChoice(next, cat, ep, form);
    if (!r.sel) {
      dropped.push(what);
      continue;
    }
    next = r.sel;
    const c = row.core;
    if (form === "core" && c && c.link && c.link.choices[sel.link[ep]]) next.link[ep] = sel.link[ep];
  }
  return { sel: next, dropped };
}

// Everything the request will contain
function summarize(sel, cat) {
  const info = platformInfo(cat, sel.triple);
  const provided = [["CPU", "built-in (core)"]];  // [EP, how the bundle provides it]
  const contained = new Set(["cpu"]);  // the EPs and plugin EP backends in the bundle (they decide the third-party files)
  const long = [];
  if (info.wasm_modes && info.wasm_modes[sel.wasmMode]) {
    for (const ep of info.wasm_modes[sel.wasmMode].eps) {
      provided.push([epName(cat, ep), "built-in (core)"]);
      contained.add(ep);
    }
  }
  let coreHeavy = false;
  for (const ep of coreEps(sel, cat)) {
    const c = info.core_eps[ep];
    if (!c) continue;
    const link = linkOf(sel, c, ep);
    provided.push([epName(cat, ep), `${coreForm(sel, c, ep) === "provider" ? "provider library" : "built-in"} (core` +
                   `${link ? `; ${c.link.label}: ${link === "shared" ? "dynamic" : "static"}` : ""})`]);
    contained.add(ep);
    coreHeavy = coreHeavy || c.heavy;
  }
  if (coreHeavy) long.push("the core");
  const plugins = [];
  for (const ep of pluginEps(sel, cat)) {
    const p = pluginFor(sel, cat, ep);
    if (!p) continue;
    plugins.push(p);
    if (p.heavy) long.push(p.package);
    provided.push([epName(cat, ep), `plugin EP (${p.package}${p.variant_label ? `, ${p.variant_label}` : ""})`]);
    for (const e of Object.keys(p.eps)) contained.add(e);
  }
  const kinds = gpuKinds(sel, cat);
  // third-party files: vendor SDK files fetch-restricted downloads, and what the user installs
  const fetched = [], system = new Set();
  for (const ep of contained) {
    const tp = thirdPartyOf(cat, ep, info.os);
    if (tp.fetch) {
      const perTarget = (tp.download_mb_per_amd_target || 0) * (kinds.has("amd") ? sel.amdTargets.length : 0);
      fetched.push({ ep, components: tp.components, mb: (tp.download_mb || {})[info.os] + perTarget || undefined,
                     parts: [...new Set(tp.fetch.map((f) => (f.endsWith(":dev") ? "dev" : "runtime")))] });
    }
    (tp.system || []).forEach((s) => system.add(s));
  }
  const cudaMajor = needsCuda(sel, cat) ? sel.cudaMajor : null;
  return { provided, plugins, long, fetched, system: [...system], cudaMajor, kinds,
           requirements: requirements(sel, cat, plugins, cudaMajor, kinds), versions: versions(sel, cat, plugins, cudaMajor) };
}

// What the bundle needs on the user's machine (ci/bundle.py make_plan lists the same in the bundle's README)
function requirements(sel, cat, plugins, cudaMajor, kinds) {
  const info = platformInfo(cat, sel.triple);
  const out = [...info.requirements];
  if (info.wasm_modes && info.wasm_modes[sel.wasmMode]) out.push(...info.wasm_modes[sel.wasmMode].requirements);
  for (const ep of coreEps(sel, cat)) out.push(...((info.core_eps[ep] || {}).requirements || []));
  for (const p of plugins) out.push(...p.requirements);
  if (cudaMajor) {
    const g = cat.gpu.cuda;
    out.push(kinds.has("cuda") && sel.cudaPtx ? `${g.drivers[cudaMajor]}; ${g.ptx_note}` : g.drivers[cudaMajor]);
  }
  return [...new Set(out)];
}

// [component, version, link] of what the packages of a selection are built from: ONNX Runtime and the plugin EP
// repositories (commits of this repository's submodules), and the third-party SDKs (resolved when the catalog was made)
function versions(sel, cat, plugins, cudaMajor) {
  const info = platformInfo(cat, sel.triple);
  const rows = [];
  const source = (path) => {
    const src = cat.sources[path];
    if (src) rows.push([src.url.split("/").pop(), src.commit.slice(0, 12), src.tree]);
  };
  const ort = cat.sources.onnxruntime;
  if (ort) rows.push(["ONNX Runtime", `${cat.onnxruntime_version || ""} (${ort.commit.slice(0, 12)})`.trim(), ort.tree]);
  for (const path of [...new Set(plugins.flatMap((p) => p.sources))].filter((p) => p !== "onnxruntime")) source(path);
  // component -> the versions the selected packages pin it to ("": the newest, as resolved for the catalog)
  const pins = {};
  const use = (part) => part.sdks.forEach((c) => (pins[c] = pins[c] || new Set()).add((part.pins || {})[c] || ""));
  use(info);
  for (const ep of coreEps(sel, cat)) if (info.core_eps[ep]) use(info.core_eps[ep]);
  plugins.forEach(use);
  const resolved = (cat.sdk_versions || {})[sel.triple] || {};
  for (const c of Object.keys(pins).sort()) {
    const key = cat.sdk_cuda_dependent.includes(c) ? `${c}@${cudaMajor || "13"}` : c;
    const pinned = pins[c].size === 1 ? [...pins[c]][0] : "";
    rows.push([cat.sdk_names[c] || c, pinned || resolved[key] || "resolved by the build", null]);
  }
  return rows;
}

function config(sel, cat) {
  const s = summarize(sel, cat);
  const info = platformInfo(cat, sel.triple);
  const eps = {};
  for (const ep of Object.keys(cat.eps)) if (sel.choices[ep]) eps[ep] = sel.choices[ep];
  return {
    schema: 5,
    triple: sel.triple,
    eps,
    variants: Object.fromEntries(pluginEps(sel, cat).filter((ep) => variantsOf(cat, sel.triple, ep).length)
      .map((ep) => [ep, variantOf(sel, cat, ep)])),
    amd_hip: pluginEps(sel, cat).some((ep) => hipChoice(sel, cat, ep)) ? !!sel.amdHip : null,
    cuda_major: s.cudaMajor,
    wasm_mode: info.wasm_modes ? sel.wasmMode : null,
    link: Object.fromEntries(coreEps(sel, cat).filter((ep) => (info.core_eps[ep] || {}).link)
      .map((ep) => [ep, linkOf(sel, info.core_eps[ep], ep)])),
    cuda_archs: s.kinds.has("cuda") ? [...sel.cudaArchs].sort(bySm) : null,
    cuda_ptx: s.kinds.has("cuda") ? sel.cudaPtx : null,
    amd_gpu_targets: s.kinds.has("amd") ? amdTargetsFor(cat, info).filter((t) => sel.amdTargets.includes(t)) : null,
  };
}

function issueUrl(sel, cat) {
  const cfg = config(sel, cat);
  const core = ["cpu", ...coreEps(sel, cat)].map((ep) => epName(cat, ep));
  const plugin = (ep) => {
    const p = pluginFor(sel, cat, ep);
    return `${epName(cat, ep)} plugin${p && p.variant_label ? ` (${p.variant_label})` : ""}`;
  };
  const parts = [cfg.wasm_mode ? `wasm ${cfg.wasm_mode}` : `core ${core.join("+")}`, ...pluginEps(sel, cat).map(plugin)];
  const title = `[bundle] ${sel.triple}: ${parts.join(", ")}`.slice(0, 200);
  const params = new URLSearchParams({ template: "bundle.yml", title, config: JSON.stringify(cfg, null, 1) });
  return `https://github.com/${cat.repository}/issues/new?${params}`;
}


// ---------------------------------------------------------------------------
// User interface
// ---------------------------------------------------------------------------

if (typeof document !== "undefined") {
  const STORAGE_KEY = "ortb-selection-v6";
  const $ = (q, root = document) => root.querySelector(q);
  const h = (tag, attrs = {}, ...children) => {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === false || v === null || v === undefined) continue;
      if (k === "class") el.className = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat(Infinity)) if (c !== null && c !== undefined && c !== false) el.append(c);
    return el;
  };

  let cat = null;
  let sel = null;
  let note = "";      // what the last change did elsewhere (target switch, restore)
  let effects = {};   // EP (or "cuda") -> what the last change did to it

  function store() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(sel));
    } catch (e) { /* storage unavailable */ }
  }

  // The last selection (also one an older page stored), repaired where the catalog changed since
  function restore() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || localStorage.getItem("ortb-selection-v5") || "null");
      if (!saved || !cat.platforms[saved.triple]) return null;
      const s = { ...newSelection(cat, saved.triple), ...saved };
      s.variants = s.variants || {};
      s.link = s.link || {};
      if (!validate(s, cat) && s.cudaArchs.every((a) => cat.gpu.cuda.archs[a])) return { sel: s };
      const r = retarget(s, cat, saved.triple);
      return { sel: r.sel, note: `Restored your last selection${r.dropped.length ? `; no longer offered: ${r.dropped.join(", ")}` : ""}.` };
    } catch (e) {
      return null;
    }
  }

  function update(next, msg = "", fx = []) {
    sel = next;
    note = msg;
    effects = {};
    for (const e of fx) effects[e.ep || "cuda"] = e.text;
    render();
  }

  function switchTarget(triple) {
    if (triple === sel.triple) return;
    const r = retarget(sel, cat, triple);
    update(r.sel, r.dropped.length ? `${triple} does not offer: ${r.dropped.join(", ")}.` : "");
  }

  // One choice out of a few (shown like the EP buttons): options [{value, text, detail}]
  function segmented(label, options, current, onPick) {
    return h("div", { class: "seg", role: "group", "aria-label": label },
      h("span", { class: "seg-label" }, label),
      options.map((o) => h("button", {
        type: "button", class: `choice opt${o.value === current ? " on" : ""}`, "aria-pressed": o.value === current ? "true" : "false",
        title: o.detail || null, onclick: () => { if (o.value !== current) onPick(o.value); },
      }, h("span", { class: "label" }, o.text))));
  }

  function renderTarget() {
    const platforms = [...new Set(cat.triples.map(platformOf))];
    const current = platformOf(sel.triple);
    const archs = cat.triples.filter((t) => platformOf(t) === current);
    const info = platformInfo(cat, sel.triple);
    const pick = (platform) => {  // the triple of a platform with the current architecture, else its first one
      const triples = cat.triples.filter((t) => platformOf(t) === platform);
      return triples.find((t) => archOf(t) === archOf(sel.triple)) || triples[0];
    };
    return h("section", { class: "card" },
      h("h2", {}, "1. Target"),
      h("div", { class: "row" },
        h("label", {}, "Platform",
          h("select", { onchange: (e) => switchTarget(pick(e.target.value)) },
            platforms.map((p) => h("option", { value: p, selected: p === current }, p)))),
        h("label", {}, "Architecture",
          h("select", { onchange: (e) => switchTarget(e.target.value) },
            archs.map((t) => h("option", { value: t, selected: t === sel.triple }, archOf(t))))),
        h("code", { class: "triple" }, sel.triple)),
      info.note || info.experimental ? h("p", { class: "hint" }, info.note,
        info.experimental ? h("span", { class: "tag warn" }, " experimental target: builds may fail") : null) : null);
  }

  // A form of an EP: a button, or a placeholder where the target does not offer it. A button that changes other EPs
  // (or the CUDA version) says so before it is clicked.
  function choiceButton(row, form, label, detail, tags) {
    const offered = form === "plugin" ? row.plugins.length > 0 : form === "core" ? !!row.core : true;
    if (!offered) {
      const why = form === "plugin" ? "no plugin EP package for this target" : "cannot be compiled into the core for this target";
      return h("span", { class: "choice none", title: capitalize(why), "aria-label": `${label}: ${why}` },
        h("span", { class: "label" }, label), h("span", { class: "dash", "aria-hidden": "true" }, "—"));
    }
    const selected = (sel.choices[row.ep] || null) === form;
    const r = selected ? null : setChoice(sel, cat, row.ep, form);
    const blocked = r && r.error;
    const fx = (r && r.effects) || [];
    return h("button", {
      type: "button", class: `choice ${form || "off"}${selected ? " on" : ""}`, disabled: !!blocked,
      "aria-pressed": selected ? "true" : "false",
      title: blocked ? `Not available: ${blocked}` : [detail, ...fx.map((e) => e.text)].filter(Boolean).join("\n") || null,
      onclick: () => {
        const r2 = setChoice(sel, cat, row.ep, form);
        if (r2.sel) update(r2.sel, "", r2.effects);
      },
    }, h("span", { class: "label" }, label),
      tags.filter(Boolean).map((t) => h("span", { class: `tag${t.warn ? " warn" : ""}` }, t.text)),
      fx.map((e) => h("span", { class: "tag fx" }, e.tag)));
  }

  function renderEps() {
    const info = platformInfo(cat, sel.triple);
    if (info.wasm_modes) {
      return h("section", { class: "card" },
        h("h2", {}, "2. WebAssembly build mode"),
        h("p", { class: "hint" }, "WebAssembly builds are exclusive build modes, so the EPs come with the mode: choose one."),
        note ? h("p", { class: "note", role: "status" }, note) : null,
        h("div", { class: "modes" }, Object.keys(info.wasm_modes).map((m) =>
          h("label", { class: "check" },
            h("input", { type: "radio", name: "wasm-mode", checked: sel.wasmMode === m,
                         onchange: () => update({ ...cloneSel(sel), wasmMode: m }) }),
            ` ${WASM_MODES[m] || m}`))));
    }
    const rows = epRows(cat, sel.triple).map((row) => {
      const choice = sel.choices[row.ep];
      // the plugin package the row has, or would have after a click (which may switch the CUDA version)
      const after = choice === "plugin" || !row.plugins.length ? null : setChoice(sel, cat, row.ep, "plugin").sel;
      const p = row.plugins.length ? (pluginFor(after || sel, cat, row.ep) || row.plugins[0]) : null;
      const c = row.core;
      const ep = cat.eps[row.ep];
      const variants = variantsOf(cat, sel.triple, row.ep);
      const extras = [];
      if (c && c.link && choice === "core") {
        extras.push(segmented(c.link.label, Object.entries(c.link.choices).map(([k, v]) =>
          ({ value: k, text: k === "shared" ? "dynamic" : "static", detail: capitalize(v.text) })), linkOf(sel, c, row.ep),
        (v) => { const next = cloneSel(sel); next.link[row.ep] = v; update(next); }));
      }
      if (choice === "plugin" && variants.length > 1) {
        extras.push(segmented("Version", variants.map((v) => ({ value: v.id, text: v.label, detail: capitalize(v.note) })),
          variantOf(sel, cat, row.ep), (v) => { const next = cloneSel(sel); next.variants[row.ep] = v; update(next); }));
      }
      // the HIP backend of the AMDGPU version, below the description of the version
      const hip = choice === "plugin" && hipChoice(sel, cat, row.ep) && cat.hip_option ? h("label", { class: "check" },
          h("input", { type: "checkbox", checked: !!sel.amdHip,
                       onchange: (e) => update({ ...cloneSel(sel), amdHip: e.target.checked }) }),
          h("span", {}, ` ${cat.hip_option.label} `, h("span", { class: "tag warn" }, "experimental"),
            h("span", { class: "ep-note" }, ` ${capitalize(cat.hip_option.note)}.`))) : null;
      const notes = [];
      if (choice === "plugin" && p && p.variant_note) notes.push(`${capitalize(p.variant_note)}.`);
      if (choice === "plugin" && c && c.link) notes.push(`${c.link.label}: ${c.link.plugin}.`);
      if (choice === "core" && c && c.note) notes.push(`${capitalize(c.note)}.`);
      const fx = effects[row.ep];
      return h("tr", { class: `${choice ? "active" : ""}${fx ? " changed" : ""}` },
        h("th", { scope: "row" }, h("span", { class: "ep" }, ep.name), h("span", { class: "vendor" }, ep.vendor)),
        h("td", {}, h("div", { class: "choices" },
          choiceButton(row, null, "Off", "Leave this EP out", []),
          choiceButton(row, "plugin", "Plugin EP",
            p ? `Add the plugin EP package ${p.package} (a separate library you register at run time)` : "",
            [p && { text: p.package.replace(/^onnxruntime-/, "") },
             row.plugins.length > 1 && row.plugins.some((o) => o.cuda_major) && { text: `CUDA ${[...new Set(row.plugins.map((o) => o.cuda_major))].sort(bySm).join("/")}` },
             p && p.heavy && { text: "long build" }, p && p.experimental && { text: "experimental", warn: true }]),
          choiceButton(row, "core", "In core",
            c ? `Compile into the core library (${c.form === "provider" ? "provider library next to it" : "built-in"})${c.note ? ": " + c.note : ""}` : "",
            [c && { text: coreForm(sel, c, row.ep) === "provider" ? "provider library" : "built-in" },
             c && c.cuda && { text: `CUDA ${c.cuda.join("/")}` },
             c && c.heavy && { text: "long build" }, c && c.experimental && { text: "experimental", warn: true }])),
          extras,
          notes.map((n) => h("p", { class: "ep-note" }, n)),
          hip,
          fx ? h("p", { class: "ep-note fx", role: "status" }, fx) : null));
    });
    return h("section", { class: "card" },
      h("h2", {}, "2. Execution providers"),
      h("p", { class: "hint" },
        "For every EP choose whether to leave it out, add its plugin EP package (a separate library, built once and ",
        "reused by every bundle), or compile it into the core library. CPU is always in the core; the EPs that exist ",
        "only in the core come first. A dash marks a form the target does not offer; a button that also changes other ",
        "EPs says so on its label."),
      h("div", { class: "toolbar" },
        h("button", { type: "button", class: "button", onclick: () => update(newSelection(cat, sel.triple)) }, "Reset")),
      note ? h("p", { class: "note", role: "status" }, note) : null,
      h("div", { class: "table-wrap" }, h("table", { class: "eps" }, h("tbody", {}, rows))));
  }

  // The CUDA version, and the GPU architectures the CUDA and MIGraphX code of the selection is compiled for
  function renderGpu(s) {
    const info = platformInfo(cat, sel.triple);
    const toggle = (key, value, on) => {
      const next = cloneSel(sel);
      next[key] = on ? [...new Set([...next[key], value])] : next[key].filter((x) => x !== value);
      update(next);
    };
    const parts = [];
    if (s.cudaMajor) {
      const g = cat.gpu.cuda;
      const majors = cudaChoices(sel, cat);
      const version = majors.length > 1
        ? segmented("CUDA version", majors.map((m) => ({ value: m, text: `${m}.x` })), sel.cudaMajor,
          (m) => update({ ...cloneSel(sel), cudaMajor: m }))
        : h("p", { class: "current" }, h("strong", {}, `CUDA ${sel.cudaMajor}.x`),
          `: ${cudaLimit(sel, cat, sel.cudaMajor) || "the only version the selection can be built with"}.`);
      const status = {
        native: () => ({ text: "its own kernels" }),
        compat: (c) => ({ text: `runs the sm_${c.from} kernels` }),
        ptx: (c) => ({ text: `PTX of sm_${c.from}, compiled when a session starts`, warn: true }),
        none: () => ({ text: "not supported", warn: true }),
      };
      parts.push(h("fieldset", {},
        h("legend", {}, "NVIDIA CUDA"),
        version,
        effects.cuda ? h("p", { class: "ep-note fx", role: "status" }, effects.cuda) : null,
        s.kinds.has("cuda") ? [
          h("p", { class: "hint" }, "The CUDA code is compiled for the selected SM versions. A GPU runs the kernels of its ",
            "own SM version or of an older one of the same major version (not the arch-specific kernels of sm_90 and ",
            "newer); with PTX, newer GPUs compile the kernels of the newest selected version when a session starts. Every ",
            "architecture lengthens the CUDA builds and enlarges the libraries: many of them can push a first build past ",
            "the 6-hour job limit of GitHub-hosted runners. At least one stays selected."),
          Object.entries(g.archs).map(([sm, gpus]) => {
            const checked = sel.cudaArchs.includes(sm);
            const last = checked && sel.cudaArchs.length === 1;
            const st = status[cudaCoverage(sm, sel.cudaArchs, sel.cudaPtx, g.exact).kind](cudaCoverage(sm, sel.cudaArchs, sel.cudaPtx, g.exact));
            return h("label", { class: "check" },
              h("input", { type: "checkbox", checked, disabled: last, onchange: (e) => toggle("cudaArchs", sm, e.target.checked) }),
              h("span", {}, h("code", {}, `sm_${sm}`), ` ${gpus} `, h("span", { class: `tag${st.warn ? " warn" : ""}` }, st.text)));
          }),
          h("label", { class: "check" },
            h("input", { type: "checkbox", checked: sel.cudaPtx, onchange: (e) => update({ ...cloneSel(sel), cudaPtx: e.target.checked }) }),
            " PTX of the newest selected architecture, for newer GPUs (larger libraries)")]
          : h("p", { class: "hint" }, "No selected package compiles CUDA code for this target, so there are no GPU ",
            "architectures to choose.")));
    }
    if (s.kinds.has("amd")) {
      const g = cat.gpu.amd;
      const routed = info.os === "windows" && s.plugins.some((p) => p.variant === "amdgpu");
      parts.push(h("fieldset", {},
        h("legend", {}, "AMD GPU targets (MIGraphX)"),
        h("p", { class: "hint" }, "MIGraphX in the AMD GPU plugin EP is built for the selected targets; a GPU needs its own ",
          "target. At least one stays selected.",
          routed ? " On Windows the AMDGPU version runs RDNA2 GPUs on its DirectML backend, so gfx1030 matters only when an " +
            "application selects the MIGraphX backend." : ""),
        amdTargetsFor(cat, info).map((t) => {
          const checked = sel.amdTargets.includes(t);
          const last = checked && sel.amdTargets.length === 1;
          return h("label", { class: "check" },
            h("input", { type: "checkbox", checked, disabled: last, onchange: (e) => toggle("amdTargets", t, e.target.checked) }),
            h("span", {}, h("code", {}, t), ` ${g.targets[t].gpus}`));
        })));
    }
    return parts.length ? h("section", { class: "card" }, h("h2", {}, "3. GPU code"), parts) : null;
  }

  function renderSummary(s, number) {
    const err = validate(sel, cat);
    const comp = (c) => cat.components[c];
    const fetchedComps = [...new Set(s.fetched.flatMap((f) => f.components))];
    const licenses = fetchedComps;
    const cfg = config(sel, cat);
    const ack = $("#ack") && $("#ack").checked;
    const script = platformInfo(cat, sel.triple).os === "windows" ? "fetch-restricted.ps1" : "fetch-restricted.sh";
    return h("section", { class: "card" },
      h("h2", {}, `${number}. Request`),
      h("div", { class: "table-wrap" }, h("table", { class: "summary" },
        h("thead", {}, h("tr", {}, h("th", {}, "EP"), h("th", {}, "Provided as"))),
        h("tbody", {}, s.provided.map(([name, how]) => h("tr", {}, h("td", {}, name), h("td", {}, how)))))),
      h("p", {}, "Every package is built for the request unless an earlier request built it from the same sources in ",
        "the last 90 days (a core is reused for the same set of EPs). ",
        s.long.length ? `First builds of ${s.long.join(", ")} take up to a few hours.` : "This selection builds in about an hour or less."),
      s.fetched.length ? h("div", {},
        h("h3", {}, "Downloaded on your machine"),
        h("p", { class: "hint" }, "The vendors' licenses restrict redistributing these files, so they are not in the bundle: ",
          "its ", h("code", {}, script), " downloads each runtime together with its headers and import libraries, in the ",
          "versions the packages were built with. Skip it if you installed the SDKs yourself."),
        h("ul", { class: "plain" }, s.fetched.map((f) => h("li", {}, `${epName(cat, f.ep)}: `,
          f.components.map((c) => `${comp(c).name} (${f.parts.map((p) => comp(c)[p]).filter(Boolean).join(", ")})`).join("; "),
          f.mb ? ` — about ${formatSize(f.mb)} to download` : "")))) : null,
      s.system.length ? h("p", { class: "warn" }, "Install yourself: ",
        s.system.map((c) => `${comp(c).name}: ${comp(c).system}`).join("; "), ".") : null,
      s.requirements.length ? h("div", {},
        h("h3", {}, "Requirements"),
        h("ul", { class: "plain" }, s.requirements.map((r) => h("li", {}, r))),
        h("p", { class: "hint" }, "GPU and NPU drivers are not included. The bundle's README.md repeats these; for Linux it ",
          "also lists the glibc and libstdc++ versions its libraries were measured to need.")) : null,
      h("details", {},
        h("summary", {}, "Versions"),
        h("div", { class: "table-wrap" }, h("table", { class: "summary" },
          h("thead", {}, h("tr", {}, h("th", {}, "Built from"), h("th", {}, "Version"))),
          h("tbody", {}, s.versions.map(([name, version, link]) => h("tr", {}, h("td", {}, name),
            h("td", {}, link ? h("a", { href: link, target: "_blank", rel: "noopener" }, version) : version)))))),
        h("p", { class: "hint" }, cat.sdk_versions
          ? `Third-party SDK versions as resolved on ${new Date(cat.generated_at).toLocaleDateString()}; each package's BUILD_INFO.json records the ones it was built with.`
          : "Each package's BUILD_INFO.json records the third-party SDK versions it was built with.")),
      licenses.length ? h("div", { class: "licenses" },
        h("h3", {}, "Third-party licenses"),
        h("ul", {}, licenses.map((c) => h("li", {}, h("a", { href: comp(c).license, target: "_blank", rel: "noopener" },
          comp(c).name))))) : null,
      h("label", { class: "check ack" },
        h("input", { type: "checkbox", id: "ack", checked: ack, onchange: () => render() }),
        licenses.length ? " I have read these licenses and accept them."
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
    const gpu = renderGpu(s);
    const focus = document.activeElement && document.activeElement.id;
    $("#app").replaceChildren(renderTarget(), renderEps(), ...(gpu ? [gpu] : []), renderSummary(s, gpu ? 4 : 3));
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
    const restored = restore();
    sel = restored ? restored.sel : newSelection(cat, first);
    note = restored && restored.note ? restored.note : "";
    render();
  }

  main();
}
