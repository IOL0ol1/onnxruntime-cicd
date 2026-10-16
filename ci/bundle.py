#!/usr/bin/env python3
"""Bundle requests: turn a configuration from the configurator page (site/) into one ready-to-use package.

A bundle is one core (a prebuilt core package, or a custom core compiled with the selected in-tree EPs) plus plugin
EP packages, merged into one bin/ lib/ include/ tree, with the redistributable third-party runtimes they need, a
CMake package (lib/cmake/onnxruntime-bundle), plugins.json, an example, and fetch-restricted.(sh|ps1) for the
third-party files whose licenses do not allow redistributing them (headers and import libraries of the NVIDIA and
Qualcomm SDKs, the Vitis AI runtime, the CANN toolkit).

    python ci/bundle.py resolve --event $GITHUB_EVENT_PATH --plan plan.json --github-output $GITHUB_OUTPUT
    python ci/bundle.py resolve --config config.json --plan plan.json         # check a configuration locally
    python ci/bundle.py download --plan plan.json --dest DIR
    python ci/bundle.py assemble --plan plan.json --packages DIR --work WORK --out DIST

`resolve` validates the configuration against the catalog (ci/catalog.py) and writes the plan: the build job and
inputs hash of every package (ci/plan.py), and the artifact of an earlier run it can reuse; the workflow builds the
others. `download` fetches the package artifacts into DIR/<package>/, `assemble` writes the bundle archive into DIST.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


catalog = _load("ci_catalog", ROOT / "ci" / "catalog.py")
build = catalog.build
plan_mod = catalog.plan

CONFIG_SCHEMA = 2
CUSTOM_CORE = "onnxruntime-custom"  # package name of a compiled core
CUDA_EPS = {"cuda", "tensorrt", "nv-tensorrt-rtx"}


class RequestError(Exception):
    """A configuration that cannot be built; the message is shown to the requester."""


# ---------------------------------------------------------------------------
# resolve: configuration -> plan
# ---------------------------------------------------------------------------

def parse_issue(body: str) -> dict:
    """The configuration of an issue created from .github/ISSUE_TEMPLATE/bundle.yml."""
    m = re.search(r"```(?:json)?\s*\n(.*?)\n```", body or "", re.S) or re.search(r"(\{.*\})", body or "", re.S)
    if not m:
        raise RequestError("no configuration found: the issue must contain the JSON from the configurator page")
    try:
        config = json.loads(m.group(1))
    except ValueError as e:
        raise RequestError(f"the configuration is not valid JSON: {e}")
    if not re.search(r"^- \[[xX]\] ", body, re.M):
        raise RequestError("the license acknowledgement is not checked")
    return config


def custom_entry(triple: str, eps: list[str], cuda_major: str) -> dict:
    """Build recipe entry of a custom core: the runner of the target's core builds, with the packages and container of
    the matrix entries that build those EPs."""
    entries = [e for e in catalog.entries() if e["triple"] == triple and e["recipe"] == "ort"
               and e["variant"] not in ("cuda-plugin", "webgpu-plugin", "arm64x")]
    base = next((e for e in entries if e["package"] == "onnxruntime-all"), None) or \
        next(e for e in entries if e["package"] == "onnxruntime")
    apt, docker, heavy = set(base["apt"].split()), "", bool(base["heavy"])
    for ep in eps:
        carrier = next(e for e in entries if ep in catalog.core_eps(e))
        apt |= set(carrier["apt"].split())
        docker = docker or carrier["docker"]
        heavy = heavy or bool(carrier["heavy"])
    return {**plan_mod.DEFAULTS, "package": CUSTOM_CORE, "recipe": "ort", "variant": "custom", "triple": triple,
            "runner": base["runner"], "submodules": "onnxruntime", "cuda_major": cuda_major, "eps": ",".join(eps),
            "apt": " ".join(sorted(apt)), "docker": docker, "heavy": heavy}


def make_plan(config: dict, cat: dict) -> dict:
    """Validate a configuration against the catalog and work out what the bundle consists of (without the builds;
    see plan_builds)."""
    if config.get("schema") != CONFIG_SCHEMA:
        raise RequestError(f"unsupported configuration schema {config.get('schema')!r} (expected {CONFIG_SCHEMA}); "
                           "reload the configurator page and create a new request")
    triple = config.get("triple")
    if triple not in cat["triples"]:
        raise RequestError(f"unknown target triple {triple!r}")
    items = {(it["package"], it["triple"]): it for it in cat["items"]}

    def get(name: str, kind: str) -> dict:
        it = items.get((name, triple))
        if not it or it["kind"] != kind:
            raise RequestError(f"{name} is not a {kind} package for {triple}")
        return it

    core_name, compile_eps = config.get("core"), list(dict.fromkeys(config.get("compile") or []))
    if bool(core_name) == bool(compile_eps):
        raise RequestError("select either a prebuilt core package or the in-tree EPs of a custom core")
    plugins = [get(p, "plugin") for p in dict.fromkeys(config.get("plugins") or [])]
    core = get(core_name, "core") if core_name else None
    if compile_eps:
        allowed = cat["compile"].get(triple)
        if not allowed:
            raise RequestError(f"custom cores cannot be compiled for {triple}")
        for ep in list(compile_eps):
            compile_eps += [r for r in cat["requires"].get(ep, []) if r not in compile_eps]
        if bad := [ep for ep in compile_eps if ep not in allowed]:
            raise RequestError(f"{', '.join(bad)} cannot be compiled for {triple} (possible: {', '.join(allowed)})")
        compile_eps = [ep for ep in cat["eps"] if ep in compile_eps]
        core_provides = ["onnxruntime"] + [f"onnxruntime_providers_{catalog.PROVIDER_LIBS[ep]}"
                                           for ep in compile_eps if ep in catalog.PROVIDER_LIBS]
        core_eps = {"cpu": "builtin", **{ep: "provider" if ep in catalog.PROVIDER_LIBS else "builtin" for ep in compile_eps}}
    else:
        core_provides, core_eps = core["provides"], core["eps"]

    owner: dict[str, str] = {}
    for name, libs in [(core_name or CUSTOM_CORE, core_provides)] + [(p["package"], p["provides"]) for p in plugins]:
        for lib in libs:
            if lib in owner:
                raise RequestError(f"{name} and {owner[lib]} both contain {lib}; select only one of them")
            owner[lib] = name

    majors = {it["cuda_major"] for it in [core, *plugins] if it and it["cuda_major"]}
    if compile_eps and CUDA_EPS & set(compile_eps):
        majors.add(str(config.get("cuda_major") or (sorted(majors)[-1] if majors else "13")))
    if len(majors) > 1:
        raise RequestError(f"the selection mixes CUDA {' and '.join(sorted(majors))}; select packages of one CUDA version")
    cuda_major = majors.pop() if majors else ""

    eps: dict[str, list[str]] = {}  # EP -> "<form> (<package>)"
    for name, forms in [(core_name or CUSTOM_CORE, core_eps)] + [(p["package"], p["eps"]) for p in plugins]:
        for ep, form in forms.items():
            eps.setdefault(ep, []).append(f"{form} ({name})")
    dev = bool(config.get("dev", True))
    wanted = config.get("fetch")  # EPs whose restricted parts fetch-restricted gets (default: all)
    bundle_runtime, fetch, system, fetched, bundled = [], [], [], set(), set()
    for ep in eps:
        tp = cat["third_party"].get(ep, {})
        ep_fetch = [f for f in tp.get("fetch", []) if (dev or not f.endswith(":dev")) and (wanted is None or ep in wanted)]
        bundle_runtime += [b for b in tp.get("bundle", []) if b not in bundle_runtime]
        fetch += [f for f in ep_fetch if f not in fetch]
        system += [s for s in tp.get("system", []) if s not in system]
        fetched |= set(tp.get("components", [])) if ep_fetch else set()
        bundled |= set(tp.get("components", [])) if tp.get("bundle") else set()
    components = sorted({c for ep in eps for c in cat["third_party"].get(ep, {}).get("components", [])})

    entries = {(e["package"], e["triple"]): e for e in catalog.entries()}
    packages = ([{"package": core_name, "kind": "core", "entry": entries[(core_name, triple)]}] if core else
                [{"package": CUSTOM_CORE, "kind": "core", "entry": custom_entry(triple, compile_eps, cuda_major or "13")}])
    packages += [{"package": p["package"], "kind": "plugin", "entry": entries[(p["package"], triple)]} for p in plugins]
    normalized = {"schema": CONFIG_SCHEMA, "triple": triple, "core": core_name, "compile": compile_eps or None,
                  "plugins": [p["package"] for p in plugins], "cuda_major": cuda_major or None, "dev": dev,
                  "fetch": sorted({f.split(":")[0] for f in fetch}) if wanted is not None else None}
    return {
        "config": normalized,
        "source": cat["source"],
        "triple": triple,
        "core": core,
        "compile": {"eps": compile_eps, "cuda_major": cuda_major or "13"} if compile_eps else None,
        "plugins": plugins,
        "packages": packages,
        "eps": eps,
        "cuda_major": cuda_major or None,
        "bundle_runtime": bundle_runtime,
        "fetch_restricted": fetch,
        "fetch_components": sorted(fetched),
        "bundled_components": sorted(bundled),
        "system": system,
        "components": {c: cat["components"][c] for c in components + system},
    }


def plan_builds(plan: dict, reuse: bool = True) -> None:
    """Give every package of a plan its build job and inputs hash, and the earlier artifact it can reuse; name the
    bundle after the configuration and the packages' hashes."""
    inputs = plan_mod.Inputs()
    for p in plan["packages"]:
        p["build"] = plan_mod.build_entry(p["entry"], inputs)
        p["reuse"] = plan_mod.find_artifact(p["build"]["artifact"]) if reuse else None
    digest = json.dumps([plan["config"], [p["build"]["hash"] for p in plan["packages"]]], sort_keys=True)
    plan["key"] = hashlib.sha256(digest.encode()).hexdigest()[:12]
    plan["name"] = f"onnxruntime-bundle-{plan['key']}-{plan['triple']}"
    plan["summary"] = summary(plan)


def summary(plan: dict) -> str:
    """Markdown description of a plan for the request issue."""
    repo = os.environ.get("GITHUB_REPOSITORY") or "IOL0ol1/onnxruntime-cicd"
    lines = [f"**Target:** `{plan['triple']}`  ",
             f"**Recipes:** commit [{plan['source']['id']}]({plan['source']['url']})  "]
    if plan["cuda_major"]:
        lines.append(f"**CUDA:** {plan['cuda_major']}.x  ")
    lines += ["", "| Package | Build |", "|---|---|"]
    for p in plan["packages"]:
        name = p["package"] + (f" ({', '.join(plan['compile']['eps'])})" if p["package"] == CUSTOM_CORE else "")
        b, r = p.get("build", {}), p.get("reuse")
        how = (f"reused from [run {r['run']}](https://github.com/{repo}/actions/runs/{r['run']})" if r else
               f"built on `{b['runner']}`{' in ' + b['docker'] if b.get('docker') else ''}" if b else "")
        lines.append(f"| `{name}` | {how} |")
    lines += ["", "| EP | Provided as |", "|---|---|"]
    lines += [f"| {plan_ep_name(ep)} | {'; '.join(forms)} |" for ep, forms in plan["eps"].items()]
    if plan["bundle_runtime"]:
        lines += ["", f"Bundled third-party runtimes: {', '.join(plan['bundle_runtime'])} (redistributable)."]
    if plan["fetch_restricted"]:
        lines += [f"Fetched by `fetch-restricted` on your machine: {', '.join(plan['fetch_restricted'])}."]
    if plan["system"]:
        lines += [f"Install yourself: {', '.join(plan['components'][s]['name'] for s in plan['system'])}."]
    return "\n".join(lines)


def plan_ep_name(ep: str) -> str:
    return catalog.EPS.get(ep, (ep,))[0]


def write_outputs(path: Path, values: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for k, v in values.items():
            v = str(v)
            if "\n" in v:
                delim = f"EOF_{hashlib.sha256(v.encode()).hexdigest()[:16]}"
                f.write(f"{k}<<{delim}\n{v}\n{delim}\n")
            else:
                f.write(f"{k}={v}\n")


def cmd_resolve(args: argparse.Namespace) -> int:
    try:
        if args.event:
            event = json.loads(args.event.read_text(encoding="utf-8"))
            config = parse_issue(event["issue"]["body"])
        else:
            config = json.loads(args.config.read_text(encoding="utf-8"))
        plan = make_plan(config, catalog.catalog())
        plan_builds(plan, reuse=not args.no_reuse)
    except Exception as e:  # report every failure in the issue, not only invalid requests
        if not isinstance(e, RequestError):
            e = RequestError(f"internal error while resolving the request ({type(e).__name__}: {e})")
        print(f"::error::{e}")
        if args.github_output:
            write_outputs(args.github_output, {"ok": "false", "error": str(e)})
        return 0 if args.github_output else 1
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(json.dumps(plan, indent=1), encoding="utf-8")
    print(plan["summary"])
    builds = [p["build"] for p in plan["packages"] if not p["reuse"]]
    found = plan_mod.find_artifact(plan["name"]) if not args.no_reuse else None
    repo = os.environ.get("GITHUB_REPOSITORY") or "IOL0ol1/onnxruntime-cicd"
    outputs = {"ok": "true", "key": plan["key"], "name": plan["name"], "summary": plan["summary"],
               "existing_url": f"https://github.com/{repo}/actions/runs/{found['run']}/artifacts/{found['id']}"
               if found else "",
               "builds": json.dumps({"include": builds}), "has_builds": "true" if builds else "false"}
    if args.github_output:
        write_outputs(args.github_output, outputs)
    return 0


# ---------------------------------------------------------------------------
# assemble: plan + package archives -> bundle
# ---------------------------------------------------------------------------

def extract(archive: Path, dest: Path) -> Path:
    """Extract a package archive; returns its top-level directory."""
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(dest, filter="tar")  # keeps the library symlinks
    tops = [p for p in dest.iterdir()]
    return tops[0] if len(tops) == 1 and tops[0].is_dir() else dest


def package_archive(directory: Path) -> Path:
    """The package archive in directory, checked against its .sha256 file."""
    archives = [p for p in directory.iterdir() if p.name.endswith((".zip", ".tar.gz"))]
    if len(archives) != 1:
        raise SystemExit(f"expected one package archive in {directory}, found {[p.name for p in archives]}")
    archive = archives[0]
    sums = [archive.with_name(archive.name + ".sha256")]
    expected = next((line.split()[0] for f in sums if f.exists() for line in f.read_text(encoding="utf-8").splitlines()
                     if line.split()[-1:] == [archive.name]), None)
    if not expected:
        raise SystemExit(f"{archive.name}: no checksum found")
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"{archive.name}: sha256 {actual} does not match {expected}")
    return archive


def cmd_download(args: argparse.Namespace) -> int:
    """Download the package artifacts of a plan into DEST/<package>/ with the GitHub CLI: the reused ones from their
    earlier runs, the others from this run (GITHUB_RUN_ID)."""
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    repo = os.environ.get("GITHUB_REPOSITORY") or "IOL0ol1/onnxruntime-cicd"
    gh = shutil.which("gh") or "gh"
    for p in plan["packages"]:
        dest = args.dest / p["package"]
        dest.mkdir(parents=True, exist_ok=True)
        run = (p["reuse"] or {}).get("run") or os.environ.get("GITHUB_RUN_ID")
        if not run:
            raise SystemExit(f"{p['package']} was not reused and GITHUB_RUN_ID is not set")
        print(f"==> {p['package']}: {p['build']['artifact']} (run {run})", flush=True)
        subprocess.run([gh, "run", "download", run, "-R", repo, "-n", p["build"]["artifact"], "-D", str(dest)],
                       check=True)
    return 0


def pe_version(path: Path) -> tuple[int, ...] | None:
    """File version of a Windows PE file (from its VS_FIXEDFILEINFO), or None."""
    data = path.read_bytes()
    i = data.find(b"\xbd\x04\xef\xfe")
    if i < 0 or i + 16 > len(data):
        return None
    ms, ls = struct.unpack_from("<II", data, i + 8)
    return ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF


def same(a: Path, b: Path) -> bool:
    if a.is_symlink() or b.is_symlink():
        return a.is_symlink() and b.is_symlink() and os.readlink(a) == os.readlink(b)
    return a.stat().st_size == b.stat().st_size and a.read_bytes() == b.read_bytes()


def merge(src: Path, dest: Path, origin: str, owners: dict[str, str], conflicts: list[dict]) -> None:
    """Merge a package tree into the bundle. A file that two packages contain with different content is resolved:
    the newer version of a Windows library, otherwise the first package's copy (recorded in conflicts)."""
    for path in sorted(src.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        rel = path.relative_to(src)
        target = dest / "packages" / origin / rel if len(rel.parts) == 1 else dest / rel
        key = rel.as_posix()
        if target.exists() or target.is_symlink():
            if same(path, target):
                continue
            kept, reason = owners.get(key, "?"), "kept the first package's copy"
            if path.suffix.lower() == ".dll" and not path.is_symlink():
                new, old = pe_version(path), pe_version(target)
                if new and old and new > old:
                    kept, reason = origin, f"newer file version {'.'.join(map(str, new))} > {'.'.join(map(str, old))}"
                elif new and old:
                    reason = f"kept file version {'.'.join(map(str, old))} (other: {'.'.join(map(str, new))})"
            conflicts.append({"path": key, "packages": [owners.get(key, "?"), origin], "kept": kept, "reason": reason})
            print(f"::warning::{key} differs between {owners.get(key)} and {origin}: {reason}")
            if kept != origin:
                continue
            target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            target.symlink_to(os.readlink(path))
        else:
            shutil.copy2(path, target)
        owners[key] = origin


def find_library(stage: Path, stem: str) -> Path | None:
    """The (unversioned) shared library file of a plugin EP in the bundle."""
    pattern = re.compile(rf"^(lib)?{re.escape(stem)}\.(dll|so|dylib)$")
    for d in ("bin", "lib"):
        for f in sorted((stage / d).glob("*")) if (stage / d).is_dir() else []:
            if pattern.match(f.name):
                return f
    return None


def core_library(stage: Path) -> tuple[str, str, str] | None:
    """(kind, location, import library) of onnxruntime in the bundle, relative to the bundle root."""
    candidates = [("SHARED", "bin/onnxruntime.dll", "lib/onnxruntime.lib"), ("SHARED", "lib/libonnxruntime.dylib", ""),
                  ("SHARED", "lib/libonnxruntime.so", ""), ("SHARED", "lib/onnxruntime.framework/onnxruntime", ""),
                  ("STATIC", "lib/libonnxruntime_webassembly.a", "")]
    for kind, loc, implib in candidates:
        if (stage / loc).exists():
            return kind, loc, implib if implib and (stage / implib).exists() else ""
    return None


# Backend targets of the CMake package: (target, file that must exist, include directories, Windows import library
# glob, library name elsewhere). A target is defined only when its files are present (fetch-restricted adds some).
BACKENDS = [
    ("CUDA::cudart", "include/cuda/cuda_runtime.h", "include/cuda;include/cuda/cccl", "cudart.lib", "cudart"),
    ("CUDA::cublas", "include/cuda/cublas_v2.h", "include/cuda", "cublas.lib", "cublas"),
    ("CUDA::cublasLt", "include/cuda/cublasLt.h", "include/cuda", "cublasLt.lib", "cublasLt"),
    ("CUDA::cufft", "include/cuda/cufft.h", "include/cuda", "cufft.lib", "cufft"),
    ("CUDA::curand", "include/cuda/curand.h", "include/cuda", "curand.lib", "curand"),
    ("CUDA::nvrtc", "include/cuda/nvrtc.h", "include/cuda", "nvrtc.lib", "nvrtc"),
    ("CUDA::nvJitLink", "include/cuda/nvJitLink.h", "include/cuda", "nvJitLink.lib", "nvJitLink"),
    ("CUDNN::cudnn", "include/cudnn/cudnn.h", "include/cudnn", "cudnn.lib", "cudnn"),
    ("TensorRT::nvinfer", "include/tensorrt/NvInfer.h", "include/tensorrt", "nvinfer_[0-9]*.lib", "nvinfer"),
    ("TensorRT::nvonnxparser", "include/tensorrt/NvOnnxParser.h", "include/tensorrt", "nvonnxparser_[0-9]*.lib",
     "nvonnxparser"),
    ("TensorRT_RTX::tensorrt_rtx", "include/tensorrt_rtx/NvInfer.h", "include/tensorrt_rtx", "tensorrt_rtx_*.lib",
     "tensorrt_rtx"),
    ("TensorRT_RTX::onnxparser", "include/tensorrt_rtx/NvOnnxParser.h", "include/tensorrt_rtx",
     "tensorrt_onnxparser_rtx_*.lib", "tensorrt_onnxparser_rtx"),
    ("QNN::headers", "include/qnn/QnnInterface.h", "include/qnn", "", ""),
    ("OpenVINO::runtime", "include/openvino/openvino.hpp", "include", "openvino.lib", "openvino"),
    ("DirectML::DirectML", "include/DirectML.h", "include", "DirectML.lib", ""),
    ("MIGraphX::migraphx", "include/migraphx", "include", "migraphx_c.lib", "migraphx_c"),
    ("HIP::amdhip64", "include/hip", "include", "amdhip64.lib", ""),
    ("Vulkan::Vulkan", "include/vulkan/vulkan.h", "include", "vulkan-1.lib", ""),
    ("MLX::mlx", "include/mlx", "include", "", "mlx"),
]

CMAKE_HELPERS = r'''
# _onnxruntime_bundle_backend(<target> <required file> <include dirs> <Windows import library glob> <library name>)
function(_onnxruntime_bundle_backend target required includes win_lib unix_lib)
  set(_root "${ONNXRUNTIME_BUNDLE_ROOT}")
  if(TARGET ${target} OR NOT EXISTS "${_root}/${required}")
    return()
  endif()
  set(_inc "")
  foreach(_d IN LISTS includes)
    if(IS_DIRECTORY "${_root}/${_d}")
      list(APPEND _inc "${_root}/${_d}")
    endif()
  endforeach()
  set(_lib "")
  if(WIN32 AND win_lib)
    file(GLOB _lib "${_root}/lib/${win_lib}")
  elseif(NOT WIN32 AND unix_lib)
    file(GLOB _lib "${_root}/lib/lib${unix_lib}.so" "${_root}/lib/lib${unix_lib}.dylib"
                   "${_root}/lib/lib${unix_lib}.so.*" "${_root}/lib/lib${unix_lib}.*.dylib")
  endif()
  if((WIN32 AND win_lib) OR (NOT WIN32 AND unix_lib))
    if(NOT _lib)
      return()  # e.g. an import library that fetch-restricted has not downloaded yet
    endif()
    list(SORT _lib)
    list(GET _lib 0 _lib)
    add_library(${target} UNKNOWN IMPORTED)
    set_target_properties(${target} PROPERTIES IMPORTED_LOCATION "${_lib}")
  else()
    add_library(${target} INTERFACE IMPORTED)
  endif()
  set_target_properties(${target} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_inc}")
  set_property(GLOBAL APPEND PROPERTY ONNXRUNTIME_BUNDLE_BACKENDS ${target})
endfunction()

# onnxruntime_bundle_copy_runtime(<target>): copy the bundle's DLLs next to <target> after it is built (Windows; elsewhere
# the libraries are found through the run path CMake sets for imported libraries)
function(onnxruntime_bundle_copy_runtime target)
  if(WIN32)
    file(GLOB _dlls "${ONNXRUNTIME_BUNDLE_ROOT}/bin/*.dll")
    if(_dlls)
      add_custom_command(TARGET ${target} POST_BUILD
                         COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_dlls} "$<TARGET_FILE_DIR:${target}>"
                         VERBATIM)
    endif()
  endif()
endfunction()
'''


def cmake_config(stage: Path, plugins: list[dict]) -> str:
    lines = ["# CMake package of an ONNX Runtime bundle, generated by onnxruntime-cicd (ci/bundle.py).",
             "#",
             "#   find_package(onnxruntime-bundle CONFIG REQUIRED PATHS <bundle>/lib/cmake)",
             "#   target_link_libraries(app PRIVATE onnxruntime::onnxruntime)",
             "#   onnxruntime_bundle_copy_runtime(app)",
             "#",
             "# Backend targets (CUDA::cudart, TensorRT::nvinfer, QNN::headers, ...) are defined when their files are",
             "# present; the ONNXRUNTIME_BUNDLE_BACKENDS global property lists them. ONNXRUNTIME_BUNDLE_PLUGIN_EPS holds",
             "# <registration name>;<library> pairs for Ort::Env::RegisterExecutionProviderLibrary.",
             "include_guard(GLOBAL)",
             'get_filename_component(ONNXRUNTIME_BUNDLE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)',
             'set(ONNXRUNTIME_BUNDLE_RUNTIME_DIR "${ONNXRUNTIME_BUNDLE_ROOT}/'
             + ("bin" if (stage / "bin").is_dir() else "lib") + '")', ""]
    core = core_library(stage)
    if (stage / "lib/cmake/onnxruntime/onnxruntimeConfig.cmake").exists():
        lines += ['include("${ONNXRUNTIME_BUNDLE_ROOT}/lib/cmake/onnxruntime/onnxruntimeConfig.cmake")']
    if core:
        kind, loc, implib = core
        props = [f'IMPORTED_LOCATION "${{ONNXRUNTIME_BUNDLE_ROOT}}/{loc}"']
        if implib:
            props.append(f'IMPORTED_IMPLIB "${{ONNXRUNTIME_BUNDLE_ROOT}}/{implib}"')
        if ".framework/" in loc:
            props.append("FRAMEWORK TRUE")
        props.append('INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_BUNDLE_ROOT}/include;'
                     '${ONNXRUNTIME_BUNDLE_ROOT}/include/onnxruntime"')
        lines += ["if(NOT TARGET onnxruntime::onnxruntime)",
                  f"  add_library(onnxruntime::onnxruntime {kind} IMPORTED)",
                  "  set_target_properties(onnxruntime::onnxruntime PROPERTIES",
                  *[f"    {p}" for p in props[:-1]], f"    {props[-1]})", "endif()"]
    lines.append(CMAKE_HELPERS)
    for target, required, includes, win_lib, unix_lib in BACKENDS:
        lines.append(f'_onnxruntime_bundle_backend({target} "{required}" "{includes}" "{win_lib}" "{unix_lib}")')
    entries = [f'"{p["registration_name"]}" "${{ONNXRUNTIME_BUNDLE_ROOT}}/{p["library"]}"' for p in plugins if p["library"]]
    lines += ["", "set(ONNXRUNTIME_BUNDLE_PLUGIN_EPS", *[f"  {e}" for e in entries], ")", ""]
    return "\n".join(lines)


EXAMPLE_CMAKE = r'''cmake_minimum_required(VERSION 3.20)
project(onnxruntime_bundle_example CXX)

find_package(onnxruntime-bundle CONFIG REQUIRED PATHS "${CMAKE_CURRENT_LIST_DIR}/../lib/cmake" NO_DEFAULT_PATH)

# The plugin EP libraries of the bundle, as initializers for plugins.h
set(ORTB_PLUGIN_ENTRIES "")
list(LENGTH ONNXRUNTIME_BUNDLE_PLUGIN_EPS _n)
if(_n GREATER 0)
  math(EXPR _last "${_n} - 1")
  foreach(_i RANGE 0 ${_last} 2)
    math(EXPR _j "${_i} + 1")
    list(GET ONNXRUNTIME_BUNDLE_PLUGIN_EPS ${_i} _name)
    list(GET ONNXRUNTIME_BUNDLE_PLUGIN_EPS ${_j} _path)
    string(APPEND ORTB_PLUGIN_ENTRIES "  {\"${_name}\", ORT_TSTR(\"${_path}\")},\n")
  endforeach()
endif()
configure_file(plugins.h.in plugins.h @ONLY)

add_executable(list_eps main.cpp)
target_compile_features(list_eps PRIVATE cxx_std_17)
target_include_directories(list_eps PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(list_eps PRIVATE onnxruntime::onnxruntime)
onnxruntime_bundle_copy_runtime(list_eps)
'''

EXAMPLE_PLUGINS_H = r'''#pragma once
// Generated by CMake from plugins.h.in: the plugin EP libraries of the bundle.
#include <onnxruntime_c_api.h>

struct PluginLibrary {
  const char* name;
  const ORTCHAR_T* path;
};

static const PluginLibrary kPlugins[] = {
@ORTB_PLUGIN_ENTRIES@  {nullptr, nullptr},
};
'''

EXAMPLE_MAIN = r'''// Lists the execution providers of the bundle: the built-in and provider EPs of the core library, and the devices of
// the plugin EPs after registering their libraries.
#include <cstdio>
#include <string>

#include <onnxruntime_cxx_api.h>

#include "plugins.h"

int main() {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnxruntime-bundle");
  std::printf("ONNX Runtime %s\n\nEPs of the core library:\n", Ort::GetVersionString().c_str());
  for (const std::string& name : Ort::GetAvailableProviders()) {
    std::printf("  %s\n", name.c_str());
  }
  for (const PluginLibrary* p = kPlugins; p->name; ++p) {
    try {
      env.RegisterExecutionProviderLibrary(p->name, p->path);
    } catch (const Ort::Exception& e) {
      std::printf("cannot register plugin EP library %s: %s\n", p->name, e.what());
    }
  }
  std::printf("\nEP devices (plugin EPs need their hardware and drivers):\n");
  for (const Ort::ConstEpDevice& device : env.GetEpDevices()) {
    std::printf("  %s (%s)\n", device.EpName(), device.EpVendor());
  }
  return 0;
}
'''

FETCH_SH = r'''#!/bin/sh
# Downloads the third-party files this bundle may not contain from their vendors' official sources (no account
# needed) and puts them into the bundle: headers into include/, import libraries into lib/, runtimes next to the
# ONNX Runtime libraries. The vendors' licenses apply; they are listed before anything is downloaded.
# Needs Python 3.9 or later. Arguments are passed on (for example --yes, --dry-run).
set -eu
here=$(cd "$(dirname "$0")" && pwd)
py=$(command -v python3 || command -v python || true)
if [ -z "$py" ]; then
  echo "fetch-restricted: Python 3 is required" >&2
  exit 1
fi
exec "$py" -I "$here/tools/fetch_ep_runtime.py" --package-dir "$here" \
  --cache "${ONNXRUNTIME_BUNDLE_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/onnxruntime-bundle}" "$@"
'''

FETCH_PS1 = r'''# Downloads the third-party files this bundle may not contain from their vendors' official sources (no account
# needed) and puts them into the bundle: headers into include\, import libraries into lib\, runtimes next to the
# ONNX Runtime libraries. The vendors' licenses apply; they are listed before anything is downloaded.
# Needs Python 3.9 or later. Arguments are passed on (for example --yes, --dry-run).
$ErrorActionPreference = 'Stop'
$py = Get-Command python, py, python3 -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $py) { throw 'fetch-restricted: Python 3 is required' }
$cache = if ($env:ONNXRUNTIME_BUNDLE_CACHE) { $env:ONNXRUNTIME_BUNDLE_CACHE } else { Join-Path $env:LOCALAPPDATA 'onnxruntime-bundle' }
& $py.Source -I (Join-Path $PSScriptRoot 'tools\fetch_ep_runtime.py') --package-dir $PSScriptRoot --cache $cache @args
exit $LASTEXITCODE
'''


def readme(plan: dict, info: dict) -> str:
    windows = "windows" in plan["triple"]
    lines = [f"# {plan['name']}", "", "ONNX Runtime bundle built by onnxruntime-cicd for a bundle request.", "",
             plan["summary"], "", "## Layout", "",
             "- `include/` ONNX Runtime headers (`include/onnxruntime/`) and the backend headers",
             "- `lib/` " + ("import libraries" if windows else "shared libraries") + ", and the CMake package in"
             " `lib/cmake/onnxruntime-bundle/`"]
    if windows:
        lines.append("- `bin/` DLLs")
    lines += ["- `plugins.json` plugin EP libraries to register with `RegisterExecutionProviderLibrary`",
              "- `packages/<package>/` license, notices and BUILD_INFO.json of each package",
              "- `third_party_licenses/` licenses of the bundled third-party components",
              "- `examples/` a CMake example that lists the EPs", "- `BUNDLE_INFO.json` what this bundle contains"]
    if plan["fetch_restricted"]:
        script = "fetch-restricted.ps1" if windows else "./fetch-restricted.sh"
        lines += ["", "## Files you fetch yourself", "",
                  "These licenses do not allow redistributing the following files, so run the script once (it needs "
                  "Python 3) to download them from the vendors and put them into this bundle:", "",
                  f"    {script}", ""]
        lines += [f"- {plan['components'][c]['name']}: {plan['components'][c]['fetched']} "
                  f"({plan['components'][c]['license']})" for c in plan["fetch_components"]]
    if plan["system"]:
        lines += ["", "## Install yourself", ""]
        lines += [f"- {plan['components'][s]['name']}: {plan['components'][s]['system']}" for s in plan["system"]]
    lines += ["", "GPU and NPU drivers are not included.", "", "## CMake", "", "```cmake",
              'find_package(onnxruntime-bundle CONFIG REQUIRED PATHS "<bundle>/lib/cmake")',
              "target_link_libraries(app PRIVATE onnxruntime::onnxruntime)", "onnxruntime_bundle_copy_runtime(app)",
              "```", "", "The example: `cmake -S examples -B build && cmake --build build --config Release`.", ""]
    if info.get("conflicts"):
        lines += ["## Files that differed between packages", ""]
        lines += [f"- `{c['path']}`: {', '.join(c['packages'])}; kept {c['kept']} ({c['reason']})" for c in info["conflicts"]]
    return "\n".join(lines) + "\n"


def write_archive(stage: Path, out: Path, windows: bool) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    if windows:
        archive = out / f"{stage.name}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for f in sorted(p for p in stage.rglob("*") if p.is_file()):
                zf.write(f, f"{stage.name}/{f.relative_to(stage).as_posix()}")
    else:
        archive = out / f"{stage.name}.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=6) as tf:
            tf.add(stage, arcname=stage.name)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(archive.name + ".sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"{archive}  {archive.stat().st_size >> 20} MiB  sha256={digest}")
    return archive


def cmd_assemble(args: argparse.Namespace) -> int:
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    triple, windows = plan["triple"], "windows" in plan["triple"]
    stage = args.work.resolve() / plan["name"]
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    packages = [p["package"] for p in plan["packages"]]
    owners: dict[str, str] = {}
    conflicts: list[dict] = []
    build_infos: dict[str, dict] = {}
    for name in packages:
        archive = package_archive(args.packages / name)
        print(f"==> {name}: {archive.name}", flush=True)
        with tempfile.TemporaryDirectory(dir=args.work) as tmp:
            top = extract(archive, Path(tmp))
            merge(top, stage, name, owners, conflicts)
        info_file = stage / "packages" / name / "BUILD_INFO.json"
        build_infos[name] = json.loads(info_file.read_text(encoding="utf-8")) if info_file.exists() else {}

    sdks: dict[str, str] = {}
    for name in packages:
        for comp, version in build_infos[name].get("third_party_sdks", {}).items():
            if comp in sdks and sdks[comp] != version:
                print(f"::warning::{comp}: {name} was built with {version}, another package with {sdks[comp]}; "
                      f"using {max(sdks[comp], version)}")
            sdks[comp] = max(sdks.get(comp, version), version)
    licenses = [{"name": plan["components"][c]["name"], "license": plan["components"][c]["license"]}
                for c in plan["fetch_components"]]
    fetch_info = {"target_triple": triple, "cuda_major": plan["cuda_major"] or "13", "third_party_sdks": sdks,
                  "eps": plan["fetch_restricted"], "licenses": licenses}

    plugins = []
    for p in plan["plugins"]:
        lib = find_library(stage, p["plugin_libs"][0])
        if not lib:
            print(f"::warning::plugin library {p['plugin_libs'][0]} of {p['package']} not found in the bundle")
        plugins.append({"registration_name": p["package"], "package": p["package"], "eps": list(p["eps"]),
                        "library": lib.relative_to(stage).as_posix() if lib else None})
    (stage / "plugins.json").write_text(json.dumps({"plugins": plugins}, indent=2), encoding="utf-8")

    info = {"name": plan["name"], "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "request": {"issue": os.environ.get("BUNDLE_ISSUE_URL"), "run": os.environ.get("BUNDLE_RUN_URL")},
            "config": plan["config"], "source": plan["source"], "target_triple": triple, "eps": plan["eps"],
            "packages": [{"package": n, "version": build_infos[n].get("version"),
                          "sources": build_infos[n].get("sources")} for n in packages],
            "third_party": {"bundled": plan["bundle_runtime"], "fetch_restricted": plan["fetch_restricted"],
                            "system": plan["system"]},
            "conflicts": conflicts, "fetch_restricted": fetch_info}
    (stage / "BUNDLE_INFO.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    # redistributable vendor runtimes (CUDA, cuDNN, TensorRT), with the versions the packages were built with
    if plan["bundle_runtime"]:
        fetch = _load("fetch_ep_runtime", ROOT / "scripts" / "fetch_ep_runtime.py")
        argv = ["--package-dir", str(stage), "--ep", *plan["bundle_runtime"], "--target", triple,
                "--cuda-major", plan["cuda_major"] or "13", "--cache", str(args.cache), "--yes"]
        if fetch.main(argv) != 0:
            raise SystemExit("bundling the vendor runtimes failed")

    if plan["fetch_restricted"]:
        tools = stage / "tools"
        tools.mkdir(exist_ok=True)
        for script in ("fetch_ep_runtime.py", "download_ep_packages.py"):
            shutil.copy2(ROOT / "scripts" / script, tools / script)
        if any(f.startswith("qnn:") for f in plan["fetch_restricted"]):  # QAIRT's version and checksum
            rel = "eps/official/onnxruntime-qnn/qcom/packages.yml"
            snapshot = tools / "submodule-files" / rel
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_text(catalog.build.sdk.dl.read_submodule(rel), encoding="utf-8")
        (stage / "fetch-restricted.sh").write_text(FETCH_SH, encoding="utf-8", newline="\n")
        (stage / "fetch-restricted.sh").chmod(0o755)
        (stage / "fetch-restricted.ps1").write_text(FETCH_PS1, encoding="utf-8", newline="\r\n")

    cmake_dir = stage / "lib" / "cmake" / "onnxruntime-bundle"
    cmake_dir.mkdir(parents=True, exist_ok=True)
    (cmake_dir / "onnxruntime-bundleConfig.cmake").write_text(cmake_config(stage, plugins), encoding="utf-8")
    if core_library(stage) and core_library(stage)[0] == "SHARED":
        examples = stage / "examples"
        examples.mkdir(exist_ok=True)
        (examples / "CMakeLists.txt").write_text(EXAMPLE_CMAKE, encoding="utf-8")
        (examples / "plugins.h.in").write_text(EXAMPLE_PLUGINS_H, encoding="utf-8")
        (examples / "main.cpp").write_text(EXAMPLE_MAIN, encoding="utf-8")
    (stage / "README.md").write_text(readme(plan, info), encoding="utf-8")
    write_archive(stage, args.out, windows)
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Resolve and assemble bundle requests")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("resolve", help="validate a configuration and write the plan")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--event", type=Path, help="GitHub issues event (GITHUB_EVENT_PATH)")
    src.add_argument("--config", type=Path, help="configuration JSON file")
    r.add_argument("--plan", type=Path, required=True, help="output: plan.json")
    r.add_argument("--github-output", type=Path, help="append the plan's settings to this file ($GITHUB_OUTPUT); "
                                                      "an invalid request then sets ok=false and error= instead of failing")
    r.add_argument("--no-reuse", action="store_true",
                   help="build every package and the bundle again instead of reusing earlier artifacts")
    d = sub.add_parser("download", help="download the packages of a plan with the GitHub CLI")
    d.add_argument("--plan", type=Path, required=True)
    d.add_argument("--dest", type=Path, required=True)
    a = sub.add_parser("assemble", help="build the bundle from the plan and the downloaded packages")
    a.add_argument("--plan", type=Path, required=True)
    a.add_argument("--packages", type=Path, required=True, help="directory with <package>/<archive> for every package")
    a.add_argument("--work", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--cache", type=Path, default=ROOT / "packages", help="download cache of the vendor runtimes")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    return {"resolve": cmd_resolve, "download": cmd_download, "assemble": cmd_assemble}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
