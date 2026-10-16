#!/usr/bin/env python3
"""Describe what a bundle request can select: catalog.json for the configurator page (site/) and for ci/bundle.py.

The catalog lists, per target triple, every package the build recipes (ci/matrix.json) can build with the EPs it
contains, how it provides each one (built into the core library, a shared provider library next to it, or a plugin
EP library) and the library files it brings, plus the in-tree EPs a custom core can be compiled with. The EP sets of
the core packages come from the recipes in ci/build.py, so they follow the build matrix by themselves. Packages are
built on demand by bundle requests (ci/plan.py), so nothing here depends on earlier builds.

    python ci/catalog.py --out site/catalog.json
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_pspec = importlib.util.spec_from_file_location("ci_plan", ROOT / "ci" / "plan.py")
plan = importlib.util.module_from_spec(_pspec)
sys.modules["ci_plan"] = plan
_pspec.loader.exec_module(plan)
build = plan.build

SCHEMA = 2

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
    return plan.entries()


def core_eps(e: dict) -> list[str]:
    """The EPs of a core package entry, read from its recipe's build.py arguments (dry run, nothing downloaded)."""
    if e["recipe"] == "ort-apple":
        return ["cpu", *APPLE_FRAMEWORK_EPS]
    job = build.Job(e["recipe"], e["variant"], e["triple"], Path("work"), Path("stage"), Path("sdk"), e["cuda_major"],
                    build.DEFAULT_CUDA_ARCHS, True)
    with contextlib.redirect_stdout(io.StringIO()):
        flags, _ = build.ort_variant(job, build.ORT_SOURCES[e["recipe"]])
    return ["cpu", *dict.fromkeys(FLAG_EPS[f] for f in flags if f in FLAG_EPS)]


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
            "cuda_major": e["cuda_major"] if "cuda" in plan.sdks_of(e) else None,
            "experimental": bool(e["experimental"]), "heavy": bool(e["heavy"]) or "cuda" in plan.sdks_of(e)}


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


def source() -> dict:
    """The commit of this repository the catalog describes (the recipes and submodules packages are built from)."""
    repo = os.environ.get("GITHUB_REPOSITORY") or "IOL0ol1/onnxruntime-cicd"
    sha = os.environ.get("GITHUB_SHA") or plan.git("rev-parse", "HEAD")
    date = plan.git("show", "-s", "--format=%cI", sha)
    return {"type": "commit", "id": sha[:12], "url": f"https://github.com/{repo}/commit/{sha}", "created_at": date}


def catalog() -> dict:
    items = [item(e) for e in entries()]
    triples = list(dict.fromkeys(it["triple"] for it in items))
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repository": os.environ.get("GITHUB_REPOSITORY") or "IOL0ol1/onnxruntime-cicd",
        "source": source(),
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
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cat = catalog()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cat, indent=1), encoding="utf-8")
    print(f"{args.out}: {len(cat['items'])} package entries for {len(cat['triples'])} targets "
          f"(commit {cat['source']['id']})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
