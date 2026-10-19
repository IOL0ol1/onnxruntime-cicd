#!/usr/bin/env python3
"""Describe what a bundle request can select: catalog.json for the configurator page (site/) and for ci/bundle.py.

A bundle is composed EP by EP: every EP is either compiled into the core library (built-in, or a shared provider
library next to it) or added as a plugin EP package. The catalog lists, per target triple, the in-tree EPs its core
recipe can be built with (ci/matrix.json "cores", with their constraints) and the plugin EP packages (ci/matrix.json
"plugins"), with the library files they bring. Packages are built on demand by bundle requests (ci/plan.py), so
nothing here depends on earlier builds.

    python ci/catalog.py --out site/catalog.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import plan

SCHEMA = 5

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
# In-tree EPs built as a shared provider library (onnxruntime_providers_<name>) next to the core library; the
# others are linked into it. Two packages with the same provider library cannot be combined.
PROVIDER_LIBS = {"cuda": "cuda", "tensorrt": "tensorrt", "nv-tensorrt-rtx": "nv_tensorrt_rtx", "openvino": "openvino",
                 "qnn": "qnn", "migraphx": "migraphx", "vitisai": "vitisai", "dnnl": "dnnl", "cann": "cann"}
# Plugin EP packages: (recipe, variant) -> (EPs, plugin libraries as staged by ci/build.py). The first EP is the
# one the package provides (the others are backends of it), the first library the one to register with
# RegisterExecutionProviderLibrary.
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

# Third-party files a bundle needs for an EP, besides the open-source components the packages build from source and
# contain (oneDNN, TIM-VX, DXC, MIGraphX, MLX, ...):
#   "optional": runtimes the bundle may contain; the request chooses whether the assemble job adds them or the bundle's
#               fetch-restricted.(sh|ps1) downloads them on the user's machine. These are vendor runtimes whose licenses
#               allow redistributing them with an application, and large open-source libraries. "size_mb" estimates
#               them (unpacked).
#   "fetch":    files whose licenses do not allow redistributing them, always downloaded by fetch-restricted
#               (dev = headers and import libraries)
#   "system":   installed by the user
EP_THIRD_PARTY = {
    "cuda": {"optional": ["cuda"], "size_mb": 2300, "fetch": ["cuda:dev"], "components": ["cuda", "cudnn"]},
    "tensorrt": {"optional": ["tensorrt"], "size_mb": 2500, "fetch": ["tensorrt:dev"], "components": ["cuda", "tensorrt"]},
    "nv-tensorrt-rtx": {"optional": ["nv-tensorrt-rtx"], "size_mb": 200, "fetch": ["nv-tensorrt-rtx:dev"],
                        "components": ["cuda", "tensorrt-rtx"]},
    "qnn": {"optional": ["qnn"], "size_mb": 200, "fetch": ["qnn:dev"], "components": ["qairt"]},
    "openvino": {"optional": ["openvino"], "size_mb": 200, "fetch": ["openvino:dev"], "components": ["openvino"]},
    "dml": {"optional": ["dml"], "size_mb": 20, "fetch": ["dml:dev"], "components": ["directml"]},
    "iree": {"optional": ["iree"], "size_mb": 200, "components": ["iree-compiler"]},
    "vitisai": {"fetch": ["vitisai:runtime"], "components": ["vitisai"]},
    "cann": {"fetch": ["cann:runtime"], "components": ["cann"]},
    "migraphx": {"system": ["rocm"]},
    "amdgpu": {"system": ["rocm"]},
}
# What the parts of a component are: "runtime" (its runtime files), "dev" (headers and import libraries), "system"
COMPONENTS = {
    "cuda": {"name": "NVIDIA CUDA Toolkit", "license": "https://docs.nvidia.com/cuda/eula/index.html",
             "runtime": "runtime libraries", "dev": "headers and import libraries"},
    "cudnn": {"name": "NVIDIA cuDNN", "license": "https://docs.nvidia.com/deeplearning/cudnn/latest/reference/eula.html",
              "runtime": "runtime libraries", "dev": "headers and import libraries"},
    "tensorrt": {"name": "NVIDIA TensorRT", "license": "https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/sla.html",
                 "runtime": "runtime libraries", "dev": "headers and import libraries"},
    "tensorrt-rtx": {"name": "NVIDIA TensorRT for RTX",
                     "license": "https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/reference/sla.html",
                     "runtime": "runtime libraries", "dev": "headers and import libraries"},
    "qairt": {"name": "Qualcomm AI Runtime (QAIRT)",
              "license": "https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk",
              "runtime": "runtime libraries", "dev": "headers (the license allows object code only)"},
    "openvino": {"name": "Intel OpenVINO",
                 "license": "https://www.intel.com/content/www/us/en/developer/articles/license/end-user-license-agreement.html",
                 "runtime": "runtime files listed in its redist.txt", "dev": "headers and import libraries"},
    "directml": {"name": "Microsoft DirectML", "license": "https://www.nuget.org/packages/Microsoft.AI.DirectML",
                 "runtime": "DirectML.dll", "dev": "DirectML.h and the import library"},
    "iree-compiler": {"name": "IREE compiler", "license": "https://github.com/iree-org/iree/blob/main/LICENSE",
                      "runtime": "the compiler library the IREE EP loads at run time"},
    "vitisai": {"name": "AMD Vitis AI runtime (Ryzen AI)", "license": "https://ryzenai.docs.amd.com/en/latest/licenses.html",
                "runtime": "runtime libraries (redistribution terms not verified)"},
    "cann": {"name": "Huawei CANN toolkit", "license": "https://www.hiascend.com/software/cann",
             "runtime": "the toolkit, installed by its vendor installer (not redistributable)"},
    "rocm": {"name": "AMD ROCm", "license": "https://rocm.docs.amd.com/en/latest/about/license.html",
             "system": "install ROCm (Linux /opt/rocm) or the HIP SDK (Windows) yourself"},
}
# In-tree EPs a core needs together (ORT's build.py builds the CUDA EP with TensorRT)
REQUIRES = {"tensorrt": ["cuda"]}
# EPs compiled into the core when the page opens (where the core can have them); everything else is up to the user
DEFAULT_CORE_EPS = ["xnnpack"]
# Explanations the page shows for an EP in the core
CORE_NOTES = {
    "tensorrt": "brings the CUDA EP into the core as well (ORT builds them together)",
    "migraphx": "the core is built in the ROCm container; combinations with other EPs are untested",
    "vsinpu": "built with the TIM-VX x86 simulator driver",
    "cann": "the CANN toolkit is installed on your machine by fetch-restricted",
}
# Linkage choices of in-tree EPs in a desktop core, on the OSes ORT's CI builds them for (windows_webgpu.yml builds Dawn as
# a shared library, windows_qnn_x64.yml the QNN EP as both library kinds). Only for an EP in the core: a plugin EP is
# always a separate library, and ORT does not support a shared Dawn with the WebGPU plugin EP
# (onnxruntime_USE_EP_API_ADAPTERS and onnxruntime_BUILD_DAWN_SHARED_LIBRARY are mutually exclusive). A choice may change
# the EP's form in the core ("form"); the first choice is the default.
LINK_OPTIONS = {
    "webgpu": {"label": "Dawn", "os": ("windows", "linux"), "plugin": "the WebGPU plugin EP always links Dawn statically",
               "choices": {"static": {"text": "linked into the core library"},
                           "shared": {"text": "webgpu_dawn library next to the core library"}}},
    "qnn": {"label": "Linkage", "os": ("windows",), "plugin": "the QNN plugin EP is always a separate library",
            "choices": {"shared": {"text": "provider library onnxruntime_providers_qnn next to the core library",
                                   "form": "provider"},
                        "static": {"text": "linked into the core library", "form": "builtin"}}},
}
PLATFORM_NOTES = {
    "wasm": "WebAssembly builds are one of five exclusive build modes",
    "apple": "ORT's framework settings: CoreML everywhere, XNNPACK on iOS and its simulator only",
    "arm64x": "one ARM64X core serves ARM64 and ARM64EC processes",
    "s390x": "the core is cross-compiled on x86_64",
}


def plugin_item(e: dict) -> dict:
    """Catalog entry of a plugin EP package for one triple."""
    eps, libs = PLUGINS[(e["recipe"], e["variant"])]
    if "windows" not in e["triple"]:  # the AMDGPU EP has a DirectML backend on Windows only
        eps, libs = [ep for ep in eps if ep != "dml"], [lib for lib in libs if lib != "directx-backend"]
    sdks = plan.sdks_of(e)
    return {"package": e["package"], "triple": e["triple"], "ep": eps[0], "eps": {ep: "plugin" for ep in eps}, "libs": libs,
            "cuda_major": e["cuda_major"] if "cuda" in sdks else None,
            "experimental": bool(e["experimental"]), "heavy": bool(e["heavy"]) or "cuda" in sdks}


def platform_item(core: dict) -> dict:
    """What the core of a triple can contain."""
    eps = {}
    for ep, req in core["eps"].items():
        lib = f"onnxruntime_providers_{PROVIDER_LIBS[ep]}" if ep in PROVIDER_LIBS else None
        eps[ep] = {"form": "provider" if lib else "builtin", "lib": lib, "cuda": req.get("cuda"),
                   "experimental": bool(req.get("experimental")), "heavy": bool(req.get("heavy")),
                   "note": CORE_NOTES.get(ep, "")}
        opts = LINK_OPTIONS.get(ep)
        target_os = "windows" if "-windows-" in core["triple"] else "linux" if core["triple"].endswith("-linux-gnu") else ""
        if opts and core["platform"] == "desktop" and target_os in opts["os"]:
            eps[ep]["link"] = {"label": opts["label"], "default": next(iter(opts["choices"])), "choices": opts["choices"],
                               "plugin": opts["plugin"]}
    item = {"platform": core["platform"], "core_eps": {ep: eps[ep] for ep in EPS if ep in eps},
            "core_from": core.get("core_from"), "experimental": bool(core.get("experimental")),
            "note": PLATFORM_NOTES.get(core["platform"], ""),
            "default_core": [ep for ep in DEFAULT_CORE_EPS if ep in eps]}
    if core["platform"] == "wasm":
        item["wasm_modes"] = {m: {"eps": v["eps"]} for m, v in core["wasm_modes"].items()}
    return item


def source() -> dict:
    """The commit of this repository the catalog describes (the recipes and submodules packages are built from)."""
    sha = os.environ.get("GITHUB_SHA") or plan.git("rev-parse", "HEAD")
    date = plan.git("show", "-s", "--format=%cI", sha)
    return {"type": "commit", "id": sha[:12], "url": f"https://github.com/{plan.REPO}/commit/{sha}", "created_at": date}


def catalog() -> dict:
    for ep, opts in LINK_OPTIONS.items():
        if tuple(opts["choices"]) != plan.build.LINK_CHOICES[ep]:
            raise SystemExit(f"LINK_OPTIONS[{ep!r}] does not match ci/build.py LINK_CHOICES")
    platforms = {t: platform_item(c) for t, c in plan.cores().items()}
    plugins = [plugin_item(e) for e in plan.entries()]
    for p in plugins:
        if p["triple"] not in platforms:
            raise SystemExit(f"{p['package']} is built for {p['triple']}, which has no core recipe")
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repository": plan.REPO,
        "source": source(),
        "triples": list(platforms),
        "eps": {k: {"name": n, "vendor": v} for k, (n, v) in EPS.items()},
        "platforms": platforms,
        "plugins": plugins,
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
    print(f"{args.out}: {len(cat['triples'])} targets, {len(cat['plugins'])} plugin packages "
          f"(commit {cat['source']['id']})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
