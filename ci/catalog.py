#!/usr/bin/env python3
"""Describe what a bundle request can select: catalog.json for the configurator page (site/) and for ci/bundle.py.

The catalog lists, per target triple, every package of a package source (a GitHub Release, or the artifacts of a
successful build run) with the EPs it contains, how it provides each one (built into the core library, a shared
provider library next to it, or a plugin EP library) and the library files it brings, plus the in-tree EPs a custom
core can be compiled with. The EP sets of the core packages come from the recipes in ci/build.py, so they follow
the build matrix (ci/matrix.json) by themselves.

    python ci/catalog.py --out site/catalog.json                  # latest release, else the latest successful build
    python ci/catalog.py --source run:38780619388 --out catalog.json
    python ci/catalog.py --source none --out catalog.json        # no package source (offline; nothing available)

GITHUB_REPOSITORY and GITHUB_TOKEN are read from the environment (the token is optional for public repositories).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_bspec = importlib.util.spec_from_file_location("ci_build", ROOT / "ci" / "build.py")
build = importlib.util.module_from_spec(_bspec)
sys.modules["ci_build"] = build
_bspec.loader.exec_module(build)

SCHEMA = 1
DEFAULTS = {"variant": "default", "cuda_major": "13", "experimental": False}

# EP -> (display name, vendor). Order is the display order.
EPS = {
    "cpu": ("CPU", "Microsoft"),
    "xnnpack": ("XNNPACK", "Google"),
    "coreml": ("CoreML", "Apple"),
    "mlx": ("MLX", "Apple"),
    "nnapi": ("NNAPI", "Google"),
    "dml": ("DirectML", "Microsoft"),
    "webgpu": ("WebGPU", "Microsoft"),
    "jsep": ("JSEP (WebGPU / WebNN through JavaScript)", "Microsoft"),
    "webnn": ("WebNN", "W3C"),
    "cuda": ("CUDA", "NVIDIA"),
    "tensorrt": ("TensorRT", "NVIDIA"),
    "nv-tensorrt-rtx": ("TensorRT RTX", "NVIDIA"),
    "dnnl": ("oneDNN", "Intel"),
    "openvino": ("OpenVINO", "Intel"),
    "migraphx": ("MIGraphX", "AMD"),
    "amdgpu": ("AMDGPU", "AMD"),
    "vitisai": ("Vitis AI", "AMD"),
    "qnn": ("QNN", "Qualcomm"),
    "cann": ("CANN", "Huawei"),
    "vsinpu": ("VSINPU", "VeriSilicon"),
    "iree": ("IREE", "IREE"),
    "telum": ("Telum", "IBM"),
    "vulkan": ("Vulkan", "Community"),
}
# build.py argument -> in-tree EP
FLAG_EPS = {"--use_xnnpack": "xnnpack", "--use_webgpu": "webgpu", "--use_dml": "dml", "--use_coreml": "coreml",
            "--use_dnnl": "dnnl", "--use_vitisai": "vitisai", "--use_cuda": "cuda", "--use_tensorrt": "tensorrt",
            "--use_nv_tensorrt_rtx": "nv-tensorrt-rtx", "--use_openvino": "openvino", "--use_qnn": "qnn",
            "--use_migraphx": "migraphx", "--use_cann": "cann", "--use_vsinpu": "vsinpu", "--use_nnapi": "nnapi",
            "--use_jsep": "jsep", "--use_webnn": "webnn"}
# In-tree EPs built as a shared provider library (onnxruntime_providers_<name>) next to the core library; the
# others are linked into it. Two packages with the same provider library cannot be combined.
PROVIDER_LIBS = {"cuda": "cuda", "tensorrt": "tensorrt", "nv-tensorrt-rtx": "nv_tensorrt_rtx", "openvino": "openvino",
                 "qnn": "qnn", "migraphx": "migraphx", "vitisai": "vitisai", "dnnl": "dnnl", "cann": "cann"}
# The Apple framework builds use ORT's default_full_*_framework_build_settings.json, which enable these
APPLE_FRAMEWORK_EPS = ["coreml", "xnnpack"]
# Plugin EP packages: (recipe, variant) -> (EPs, plugin libraries as staged by ci/build.py). The first library is
# the one to register with RegisterExecutionProviderLibrary.
PLUGINS = {
    ("ort", "cuda-plugin"): (["cuda"], ["onnxruntime_providers_cuda"]),
    ("ort", "webgpu-plugin"): (["webgpu"], ["onnxruntime_providers_webgpu"]),
    ("ep-tensorrt", "default"): (["tensorrt"], ["onnxruntime_ep_tensorrt"]),
    ("ep-nv-tensorrt-rtx", "default"): (["nv-tensorrt-rtx"], ["onnxruntime_providers_nv_tensorrt_rtx"]),
    ("ep-amdgpu", "dml"): (["dml"], ["directx-ep"]),
    ("ep-amdgpu", "migraphx"): (["migraphx"], ["migraphx-ep"]),
    ("ep-amdgpu", "amdgpu"): (["amdgpu", "migraphx", "dml"],
                              ["amdgpu-ep", "migraphx-backend", "hip-backend", "directx-backend"]),
    ("ep-qnn", "default"): (["qnn"], ["onnxruntime_providers_qnn"]),
    ("ep-mlx", "default"): (["mlx"], ["onnxruntime_mlx_ep"]),
    ("ep-iree", "default"): (["iree"], ["onnxruntime_ep_iree"]),
    ("ep-telum", "default"): (["telum"], ["telum_plugin_ep"]),
    ("ep-vulkan", "default"): (["vulkan"], ["onnxruntime_vulkan_ep"]),
}

# Third-party components a bundle needs for an EP. "bundle": the bundle job adds the component's runtime files (its
# license allows redistributing them with an application); "fetch": fetch-restricted.(sh|ps1) in the bundle downloads
# these parts on the user's machine (runtime, or dev = headers and import libraries); "system": installed by the user.
# The other runtimes (DirectML, OpenVINO, QAIRT, TensorRT RTX, oneDNN, MIGraphX, ...) are already in the packages.
EP_THIRD_PARTY = {
    "cuda": {"bundle": ["cuda"], "fetch": ["cuda:dev"], "components": ["cuda", "cudnn"]},
    "tensorrt": {"bundle": ["tensorrt"], "fetch": ["tensorrt:dev"], "components": ["cuda", "tensorrt"]},
    "nv-tensorrt-rtx": {"fetch": ["nv-tensorrt-rtx:dev"], "components": ["cuda", "tensorrt-rtx"]},
    "qnn": {"fetch": ["qnn:dev"], "components": ["qairt"]},
    "openvino": {"fetch": ["openvino:dev"], "components": ["openvino"]},
    "vitisai": {"fetch": ["vitisai:runtime"], "components": ["vitisai"]},
    "cann": {"fetch": ["cann:runtime"], "components": ["cann"]},
    "migraphx": {"system": ["rocm"]},
    "amdgpu": {"system": ["rocm"]},
}
COMPONENTS = {
    "cuda": {"name": "NVIDIA CUDA Toolkit", "license": "https://docs.nvidia.com/cuda/eula/index.html",
             "bundled": "runtime libraries", "fetched": "headers and import libraries"},
    "cudnn": {"name": "NVIDIA cuDNN", "license": "https://docs.nvidia.com/deeplearning/cudnn/latest/reference/eula.html",
              "bundled": "runtime libraries", "fetched": "headers and import libraries"},
    "tensorrt": {"name": "NVIDIA TensorRT", "license": "https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/sla.html",
                 "bundled": "runtime libraries", "fetched": "headers and import libraries"},
    "tensorrt-rtx": {"name": "NVIDIA TensorRT for RTX",
                     "license": "https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/reference/sla.html",
                     "bundled": "runtime libraries", "fetched": "headers and import libraries"},
    "qairt": {"name": "Qualcomm AI Runtime (QAIRT)",
              "license": "https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk",
              "bundled": "runtime libraries", "fetched": "headers (the license allows object code only)"},
    "openvino": {"name": "Intel OpenVINO",
                 "license": "https://www.intel.com/content/www/us/en/developer/articles/license/end-user-license-agreement.html",
                 "bundled": "runtime files listed in its redist.txt", "fetched": "headers and import libraries"},
    "vitisai": {"name": "AMD Vitis AI runtime (Ryzen AI)", "license": "https://ryzenai.docs.amd.com/en/latest/licenses.html",
                "fetched": "runtime libraries (redistribution terms not verified)"},
    "cann": {"name": "Huawei CANN toolkit", "license": "https://www.hiascend.com/software/cann",
             "fetched": "the toolkit, installed by its vendor installer (not redistributable)"},
    "rocm": {"name": "AMD ROCm", "license": "https://rocm.docs.amd.com/en/latest/about/license.html",
             "system": "install ROCm (Linux /opt/rocm) or the HIP SDK (Windows) yourself"},
}
# In-tree EPs a custom core needs together (ORT's build.py builds the CUDA EP with TensorRT)
REQUIRES = {"tensorrt": ["cuda"]}
# Recipes whose core libraries a custom compile can reproduce (desktop targets of the main repository)
COMPILE_TRIPLES = {"x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu", "x86_64-pc-windows-msvc",
                   "aarch64-pc-windows-msvc", "aarch64-apple-darwin", "x86_64-apple-darwin"}


def entries() -> list[dict]:
    return [{**DEFAULTS, **e} for e in json.loads((ROOT / "ci" / "matrix.json").read_text(encoding="utf-8"))["include"]]


def core_eps(e: dict) -> list[str]:
    """The EPs of a core package entry, read from its recipe's build.py arguments (dry run, nothing downloaded)."""
    if e["recipe"] == "ort-apple":
        return ["cpu", *APPLE_FRAMEWORK_EPS]
    job = build.Job(e["recipe"], e["variant"], e["triple"], Path("work"), Path("stage"), Path("sdk"), e["cuda_major"],
                    build.DEFAULT_CUDA_ARCHS, True)
    with contextlib.redirect_stdout(io.StringIO()):
        flags, _ = build.ort_variant(job, build.ORT_SOURCES[e["recipe"]])
    return ["cpu", *dict.fromkeys(FLAG_EPS[f] for f in flags if f in FLAG_EPS)]


def uses_cuda(e: dict) -> bool:
    return any(c == "cuda" for c, _, _ in build.required_sdks(e["recipe"], e["variant"], e["triple"], e["cuda_major"]))


def item(e: dict) -> dict:
    """Catalog entry of one package / triple of the build matrix."""
    plugin = PLUGINS.get((e["recipe"], e["variant"]))
    if plugin:
        eps, libs = plugin
        if "windows" not in e["triple"]:  # the AMDGPU EP has a DirectML backend on Windows only
            eps, libs = [ep for ep in eps if ep != "dml"], [lib for lib in libs if lib != "directx-backend"]
        forms = {ep: "plugin" for ep in eps}
        provides = libs
    else:
        eps = core_eps(e)
        forms = {ep: "provider" if ep in PROVIDER_LIBS else "builtin" for ep in eps}
        provides = ["onnxruntime"] + [f"onnxruntime_providers_{PROVIDER_LIBS[ep]}" for ep in eps if ep in PROVIDER_LIBS]
        libs = []
    return {"package": e["package"], "triple": e["triple"], "kind": "plugin" if plugin else "core",
            "recipe": f"{e['recipe']}/{e['variant']}", "eps": forms, "provides": provides, "plugin_libs": libs,
            "cuda_major": e["cuda_major"] if uses_cuda(e) else None, "experimental": bool(e["experimental"]),
            "asset": None}


def compile_options(items: list[dict]) -> dict:
    """Per triple: the in-tree EPs a custom core can be built with, i.e. those of the main repository's prebuilt cores
    for that triple (each of them is known to build there)."""
    out: dict[str, list[str]] = {}
    for it in items:
        recipe, variant = it["recipe"].split("/")
        if it["kind"] == "core" and recipe == "ort" and it["triple"] in COMPILE_TRIPLES and variant not in ("arm64x",):
            eps = out.setdefault(it["triple"], [])
            eps += [ep for ep in it["eps"] if ep in build.CUSTOM_EPS and ep not in eps]
    return {t: [ep for ep in EPS if ep in eps] for t, eps in sorted(out.items())}


# ---------------------------------------------------------------------------
# Package sources
# ---------------------------------------------------------------------------

def api(path: str) -> dict | list:
    repo = os.environ.get("GITHUB_REPOSITORY") or "IOL0ol1/onnxruntime-cicd"
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}/{path}", headers=headers)
    return json.load(urllib.request.urlopen(req, timeout=60))


def release_assets(tag: str | None) -> tuple[dict, dict[tuple[str, str], dict]]:
    """Source info and {(package, triple): asset} of a release (the latest one when tag is None)."""
    rel = api(f"releases/tags/{urllib.parse.quote(tag)}") if tag else api("releases/latest")
    tag = rel["tag_name"]
    version = tag[1:] if tag.startswith("v") else tag.removeprefix("build-")
    known = {(e["package"], e["triple"]) for e in entries()}
    assets = {}
    for a in rel["assets"]:
        for ext in (".zip", ".tar.gz"):
            stem = a["name"].removesuffix(ext)
            if stem == a["name"] or f"-{version}-" not in stem:
                continue
            package, _, triple = stem.partition(f"-{version}-")
            if (package, triple) in known:
                assets[(package, triple)] = {"name": a["name"], "size": a["size"], "url": a["browser_download_url"]}
    src = {"type": "release", "id": tag, "version": version, "url": rel["html_url"], "created_at": rel["published_at"]}
    return src, assets


def run_artifacts(run_id: str | None) -> tuple[dict, dict[tuple[str, str], dict]]:
    """Source info and {(package, triple): artifact} of a build run (the latest successful one on main when None)."""
    if not run_id:
        runs = api("actions/workflows/build.yml/runs?branch=main&status=success&per_page=10")["workflow_runs"]
        run = next((r for r in runs if r["event"] != "pull_request"), None)
        if not run:
            raise LookupError("no successful build run on main")
        run_id = str(run["id"])
    else:
        run = api(f"actions/runs/{run_id}")
    by_name = {f"pkg-{e['package']}-{e['triple']}": (e["package"], e["triple"]) for e in entries()}
    arts, page = {}, 1
    while True:
        batch = api(f"actions/runs/{run_id}/artifacts?per_page=100&page={page}")["artifacts"]
        for a in batch:
            if a["name"] in by_name and not a["expired"]:
                arts[by_name[a["name"]]] = {"name": a["name"], "size": a["size_in_bytes"], "expires_at": a["expires_at"]}
        if len(batch) < 100:
            break
        page += 1
    src = {"type": "run", "id": run_id, "version": f"{run['head_sha'][:8]}", "url": run["html_url"],
           "created_at": run["created_at"]}
    return src, arts


def source(spec: str) -> tuple[dict | None, dict]:
    """spec: auto (latest release, else the latest successful build run), release[:TAG], run[:ID] or none."""
    kind, _, ident = spec.partition(":")
    if kind == "none":
        return None, {}
    if kind == "release" or (kind == "auto"):
        try:
            return release_assets(ident or None)
        except urllib.error.HTTPError as e:
            if kind == "release" or e.code != 404:
                raise
    if kind in ("run", "auto"):
        return run_artifacts(ident or None)
    raise SystemExit(f"unknown source {spec!r}")


def catalog(spec: str = "auto") -> dict:
    items = [item(e) for e in entries()]
    src, assets = source(spec)
    for it in items:
        it["asset"] = assets.get((it["package"], it["triple"]))
    triples = list(dict.fromkeys(it["triple"] for it in items))
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repository": os.environ.get("GITHUB_REPOSITORY") or "IOL0ol1/onnxruntime-cicd",
        "source": src,
        "triples": triples,
        "eps": {k: {"name": n, "vendor": v} for k, (n, v) in EPS.items()},
        "items": items,
        "compile": compile_options(items),
        "requires": REQUIRES,
        "third_party": EP_THIRD_PARTY,
        "components": COMPONENTS,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Generate catalog.json for bundle requests")
    ap.add_argument("--source", default="auto", help="auto (default), release[:TAG], run[:ID] or none")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cat = catalog(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cat, indent=1), encoding="utf-8")
    available = sum(1 for it in cat["items"] if it["asset"])
    src = cat["source"]
    print(f"{args.out}: {len(cat['items'])} package entries, {available} available from "
          f"{src['type'] + ' ' + src['id'] if src else 'no source'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
