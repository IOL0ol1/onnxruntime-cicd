#!/usr/bin/env python3
"""Describe what a bundle request can select: catalog.json for the configurator page (site/) and for ci/bundle.py.

A bundle is composed EP by EP: every EP is either compiled into the core library (built-in, or a shared provider
library next to it) or added as a plugin EP package. The catalog lists, per target triple, the in-tree EPs its core
recipe can be built with (ci/matrix.json "cores", with their constraints) and the plugin EP packages (ci/matrix.json
"plugins"), with the library files they bring. Packages are built on demand by bundle requests (ci/plan.py), so
nothing here depends on earlier builds. It also lists the GPU architectures a request can compile the CUDA and MIGraphX
code for, what the binaries need on the user's machine, and the versions they are built from.

    python ci/catalog.py --out site/catalog.json [--versions]

--versions resolves the third-party SDK versions the builds would use now (network access; the page shows them).
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import plan

SCHEMA = 6

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
    "amdgpu": ("AMD GPU", "AMD"),
    "hip": ("HIP", "AMD"),
    "vitisai": ("Vitis AI", "AMD"),
    "qnn": ("QNN", "Qualcomm"),
    "neuropilot": ("NeuroPilot", "MediaTek"),
    "cann": ("CANN", "Huawei"),
    "vsinpu": ("VSINPU", "VeriSilicon"),
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
    ("ep-amdgpu", "amdgpu"): (["amdgpu", "migraphx", "dml"], ["amdgpu-ep", "migraphx-backend", "directx-backend"]),
    ("ep-amdgpu", "amdgpu-hip"): (["amdgpu", "migraphx", "dml", "hip"],
                                  ["amdgpu-ep", "migraphx-backend", "hip-backend", "directx-backend"]),
    ("ep-qnn", "default"): (["qnn"], ["onnxruntime_providers_qnn"]),
    ("ep-mlx", "default"): (["mlx"], ["onnxruntime_mlx_ep"]),
    ("ep-openvino", "default"): (["openvino"], ["onnxruntime_providers_openvino_plugin"]),
    ("ep-neuropilot", "default"): (["neuropilot"], ["onnxruntime_providers_neuropilot"]),
}
# Plugin EP packages offered as versions of one EP: (recipe, variant) -> (EP, version, label, OS -> explanation, extra
# fields). A request picks one of them (DEFAULT_VARIANTS: the preselected one). The AMD packages share the code of the
# AMDGPU EP, which routes every session to one of its backends; the others contain one backend each. The AMDGPU
# version comes with or without the HIP backend ("with_hip", HIP_OPTION).
_AMDGPU_NOTES = {
    "windows": "one EP with the MIGraphX and DirectML backends that picks one per GPU: MIGraphX on RDNA3 and newer, "
               "DirectML on older GPUs. It always loads the MIGraphX backend, so it needs the ROCm libraries",
    "linux": "the AMDGPU EP, with its MIGraphX backend"}
PLUGIN_VARIANTS = {
    ("ep-amdgpu", "amdgpu"): ("amdgpu", "amdgpu", "AMDGPU", _AMDGPU_NOTES, {"with_hip": False}),
    ("ep-amdgpu", "amdgpu-hip"): ("amdgpu", "amdgpu", "AMDGPU", _AMDGPU_NOTES, {"with_hip": True}),
    ("ep-amdgpu", "migraphx"): ("amdgpu", "migraphx", "MIGraphX", {
        "windows": "the MIGraphX EP alone, without the routing and the other backends",
        "linux": "the MIGraphX EP alone, without the routing of the AMDGPU EP"}, {}),
    ("ep-amdgpu", "dml"): ("amdgpu", "dml", "DirectML", {
        "windows": "AMD's port of ORT's DirectML EP, for AMD GPUs only; needs no ROCm and no MIGraphX build"}, {}),
}
DEFAULT_VARIANTS = {"amdgpu": "amdgpu"}
# The HIP backend of the AMDGPU version (the packages with "with_hip"), a choice of its own on the page
HIP_OPTION = {"label": "HIP backend",
              "note": "the HIP EP of ROCm/hip-ep (hipgpu, prebuilt), which AMDGPU uses for LLM sessions of "
                      "onnxruntime-genai on Ryzen AI 300 / Max (gfx115x). fetch-restricted downloads it from the hip-ep "
                      "release; on Windows with the ROCm runtime it is built with, which is newer than the one of "
                      "MIGraphX and then serves the whole process. Not tested together"}
# In-tree EPs offered in the row of another EP: ORT's MIGraphX EP is the "in core" form of the AMD GPU EP
CORE_ROWS = {"migraphx": "amdgpu"}

# Third-party files a bundle needs for an EP (or a backend of a plugin EP package), besides the open-source components
# the packages build from source and contain (oneDNN, TIM-VX, DXC, MIGraphX, MLX, ...):
#   "fetch":  files of the vendor SDKs whose licenses restrict redistribution: never in a bundle, downloaded on the user's
#             machine by the bundle's fetch-restricted.(sh|ps1). The runtime ("<ep>" or "<ep>:runtime") comes with the
#             headers and import libraries ("<ep>:dev"), which are in the same vendor archives. "download_mb" estimates
#             the download per OS (x86_64, measured from the archive sizes), "download_mb_per_amd_target" what each
#             AMD GPU target of the request adds.
#   "system": installed by the user
#   "os":     the keys that differ on an OS ({OS: {key: value}}, see third_party_of)
EP_THIRD_PARTY = {
    "cuda": {"fetch": ["cuda", "cuda:dev"], "components": ["cuda", "cudnn"], "download_mb": {"windows": 2460, "linux": 2180}},
    "tensorrt": {"fetch": ["tensorrt", "tensorrt:dev"], "components": ["cuda", "tensorrt"],
                 "download_mb": {"windows": 2520, "linux": 8040}},
    "nv-tensorrt-rtx": {"fetch": ["nv-tensorrt-rtx", "nv-tensorrt-rtx:dev"], "components": ["cuda", "tensorrt-rtx"],
                        "download_mb": {"windows": 100, "linux": 95}},
    "qnn": {"fetch": ["qnn", "qnn:dev"], "components": ["qairt"], "download_mb": {"windows": 2970, "linux": 2970}},
    "openvino": {"fetch": ["openvino", "openvino:dev"], "components": ["openvino"],
                 "download_mb": {"windows": 200, "linux": 110}},
    "dml": {"fetch": ["dml", "dml:dev"], "components": ["directml"], "download_mb": {"windows": 195}},
    "vitisai": {"fetch": ["vitisai:runtime"], "components": ["vitisai"]},
    "cann": {"fetch": ["cann:runtime"], "components": ["cann"]},
    "neuropilot": {"fetch": ["neuropilot"], "components": ["neuron-op-capability"], "system": ["neuron-runtime"],
                   "download_mb": {"linux": 1, "android": 1}},
    # ROCm: a system install on Linux; on Windows TheRock's runtime wheels, which fetch-restricted unpacks into rocm/
    # (the AMD plugin packages with ROCm all contain the MIGraphX backend, so "migraphx" brings it)
    "migraphx": {"system": ["rocm"], "os": {"windows": {"system": [], "fetch": ["migraphx"], "components": ["rocm"],
                                                        "download_mb": {"windows": 800}, "download_mb_per_amd_target": 150}}},
    "amdgpu": {"system": ["rocm"], "os": {"windows": {"system": []}}},
    # the HIP backend of the AMDGPU EP: hip-ep's prebuilt HIP EP (MIT), downloaded with the version the package records;
    # its archives have no headers, only import libraries on Windows
    "hip": {"fetch": ["hip"], "components": ["hip-ep"], "download_mb": {"windows": 300, "linux": 475},
            "os": {"windows": {"fetch": ["hip", "hip:dev"]}}},
}


def third_party_of(cat: dict, ep: str, target_os: str) -> dict:
    """The third-party files of an EP on an OS (EP_THIRD_PARTY with its "os" overrides applied)."""
    tp = dict(cat["third_party"].get(ep, {}))
    return {**tp, **tp.pop("os", {}).get(target_os, {})}
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
    "vitisai": {"name": "AMD Vitis AI runtime (Ryzen AI)", "license": "https://ryzenai.docs.amd.com/en/latest/licenses.html",
                "runtime": "runtime libraries (redistribution terms not verified)"},
    "cann": {"name": "Huawei CANN toolkit", "license": "https://www.hiascend.com/software/cann",
             "runtime": "the toolkit, installed by its vendor installer (not redistributable)"},
    "hip-ep": {"name": "ROCm hip-ep", "license": "https://github.com/ROCm/hip-ep/blob/main/LICENSE",
               "runtime": "the HIP EP hipgpu, on Windows with the ROCm runtime it is built with",
               "dev": "import libraries"},
    "neuron-op-capability": {"name": "MediaTek NeuroPilot op capability library",
                             "license": "https://github.com/onnxruntime/onnxruntime-ep-neuropilot/releases",
                             "runtime": "libneuron_op_capability.so (prebuilt; its archive states no redistribution terms)"},
    "neuron-runtime": {"name": "MediaTek Neuron runtime", "license": "https://neuropilot.mediatek.com/",
                       "system": "part of the software of a device with a MediaTek NPU (libneuronusdk_adapter.mtk.so)"},
    "rocm": {"name": "AMD ROCm", "license": "https://rocm.docs.amd.com/en/latest/about/license.html",
             "runtime": "runtime libraries and the device libraries of the bundle's GPU targets (TheRock's wheels), "
                        "in rocm/",
             "system": "install the ROCm release the packages were built with yourself (see Requirements)"},
}
# In-tree EPs a core needs together (ORT's build.py builds the CUDA EP with TensorRT)
REQUIRES = {"tensorrt": ["cuda"]}
# EPs compiled into the core when the page opens (where the core can have them); everything else is up to the user
DEFAULT_CORE_EPS = ["xnnpack"]
# Explanations the page shows for an EP in the core
CORE_NOTES = {
    "tensorrt": "brings the CUDA EP into the core as well (ORT builds them together)",
    "migraphx": "ORT's own MIGraphX EP; the core is built in the ROCm container against its MIGraphX, and combinations "
                "with other EPs are untested",
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
}

# GPU architectures a request compiles the CUDA code of the CUDA / TensorRT EPs for: SM version -> GPUs. A GPU runs
# the kernels of its own SM version and, within the same major version, of an older one, except for the versions in
# CUDA_EXACT_SMS, which ORT compiles arch-specific (sm_XXa) kernels for that run on that SM version only. With PTX,
# the CUDA driver compiles the kernels of the newest selected version for newer GPUs when a session starts.
CUDA_ARCHS = {
    "75": "Turing: GeForce RTX 20 / GTX 16, T4",
    "80": "Ampere: A100, A30",
    "86": "Ampere: GeForce RTX 30, A10, A40, RTX A6000",
    "89": "Ada: GeForce RTX 40, L4, L40S",
    "90": "Hopper: H100, H200, GH200",
    "100": "Blackwell: B200, GB200",
    "103": "Blackwell Ultra: B300, GB300",
    "120": "Blackwell: GeForce RTX 50, RTX PRO 6000",
    "121": "Blackwell: DGX Spark (GB10)",
}
CUDA_EXACT_SMS = ["90", "100", "103", "110", "120", "121"]  # ORT's ARCHITECTURES_WITH_ACCEL (cuda_configuration.cmake)
DEFAULT_CUDA = {"archs": ["80"], "ptx": True}  # ci/build.py DEFAULT_CUDA_ARCHS
# GPU targets a request compiles MIGraphX (the AMD plugin EPs) for: gfx target -> (GPUs, OSes). A code object runs on
# its own gfx target only. On Windows, TheRock has the device libraries of the Radeon targets only.
AMD_GPU_TARGETS = {
    "gfx90a": ("CDNA2: Instinct MI210, MI250", ("linux",)),
    "gfx942": ("CDNA3: Instinct MI300X, MI300A, MI325X", ("linux",)),
    "gfx950": ("CDNA4: Instinct MI350X, MI355X", ("linux",)),
    "gfx1030": ("RDNA2: Radeon RX 6800 / 6900, PRO W6800", ("linux", "windows")),
    "gfx1100": ("RDNA3: Radeon RX 7900, PRO W7900 / W7800", ("linux", "windows")),
    "gfx1101": ("RDNA3: Radeon RX 7800 / 7700, PRO W7700", ("linux", "windows")),
    "gfx1102": ("RDNA3: Radeon RX 7600, PRO W7600", ("linux", "windows")),
    "gfx1150": ("RDNA3.5: Ryzen AI 300 (Radeon 890M / 880M)", ("linux", "windows")),
    "gfx1151": ("RDNA3.5: Ryzen AI Max (Radeon 8060S)", ("linux", "windows")),
    "gfx1200": ("RDNA4: Radeon RX 9060", ("linux", "windows")),
    "gfx1201": ("RDNA4: Radeon RX 9070, AI PRO R9700", ("linux", "windows")),
}
DEFAULT_AMD_GPU_TARGETS = plan.build.DEFAULT_MIGRAPHX_GPU_TARGETS.split(";")

# What the binaries need on the user's machine besides the GPU / NPU drivers. The Linux packages are built on Ubuntu
# 24.04 (runners and containers), so they may use symbols up to its glibc and libstdc++ (GCC 13); the bundle's README
# lists the versions its libraries actually need.
LINUX_REQUIREMENT = ("glibc 2.39 and the libstdc++ of GCC 13 or newer (built on Ubuntu 24.04): Ubuntu 24.04, "
                     "Debian 13, Fedora 40, RHEL 10 or a newer distribution")
WINDOWS_REQUIREMENT = ("Windows 10 or newer with the latest Microsoft Visual C++ v14 Redistributable (the libraries "
                       "are built with the Visual Studio 2026 toolset, which needs a runtime at least as new)")
APPLE_OS = {"iphoneos": "iOS", "iphonesimulator": "iOS Simulator", "macabi": "Mac Catalyst", "appletvos": "tvOS",
            "appletvsimulator": "tvOS Simulator", "xros": "visionOS", "xrsimulator": "visionOS Simulator"}
ANDROID_VERSIONS = {24: "7.0", 26: "8.0", 27: "8.1", 28: "9", 29: "10", 30: "11", 31: "12", 33: "13", 34: "14", 35: "15"}
WASM_REQUIREMENTS = {
    "cpu": ["a browser or runtime with WebAssembly SIMD; multi-threading needs a cross-origin isolated page"],
    "jsep": ["a browser with WebGPU for the WebGPU EP and WebNN for the WebNN EP",
             "multi-threading needs a cross-origin isolated page"],
    "webgpu": ["a browser with WebGPU for the WebGPU EP and WebNN for the WebNN EP",
               "multi-threading needs a cross-origin isolated page"],
    "webgpu-jspi": ["a browser with WebAssembly JavaScript Promise Integration (JSPI) and WebGPU",
                    "multi-threading needs a cross-origin isolated page"],
    "static": ["link it with Emscripten {emsdk} (the version ORT's build uses)"],
}
CORE_REQUIREMENTS = {  # EP in the core -> (OS, requirement); {rocm} is the ROCm release of the core's container
    "migraphx": ("linux", "ROCm {rocm} with MIGraphX (the release the core is built against)"),
    "openvino": ("linux", "oneTBB (libtbb12) from the distribution, which the OpenVINO runtime uses on Linux"),
}
ROCM_REQUIREMENTS = {  # OS -> requirement of the plugin EP packages with MIGraphX (built for AMD GPU targets)
    "linux": "the ROCm release of the rocm/migraphx-ci-ubuntu image the package is built in (its BUILD_INFO.json "
             "records it)",
    "windows": f"the ROCm {plan.build.WINDOWS_ROCM_VERSION} runtime, which fetch-restricted unpacks into rocm/: put "
               "rocm/_rocm_sdk_core/bin and rocm/_rocm_sdk_libraries/bin on the PATH of the application",
}
PLUGIN_REQUIREMENTS = {  # recipe or (recipe, variant) -> OS -> requirement of a plugin EP package
    "ep-openvino": {"linux": CORE_REQUIREMENTS["openvino"][1]},
    ("ep-amdgpu", "amdgpu-hip"): {
        "windows": "for the HIP backend, the ROCm runtime of hip-ep, which fetch-restricted puts next to amdgpu-ep.dll "
                   "and which then serves the whole process (newer than the ROCm of MIGraphX; not tested together)",
        "linux": "for the HIP backend, the ROCm release hip-ep is built with (its release notes name it)"},
    "ep-neuropilot": {"android": "Android 10 (API level 29) or newer, which the NeuroPilot plugin EP is built for"},
}
CUDA_DRIVERS = {"12": "NVIDIA driver R525 or newer (CUDA 12)", "13": "NVIDIA driver R580 or newer (CUDA 13)"}
CUDA_PTX_NOTE = "GPUs that run the PTX need a driver at least as new as the CUDA release the package is built with"
# Display names of the third-party SDKs the packages are built with (the versions are listed on the page)
SDK_NAMES = {"cuda": "CUDA", "cudnn": "cuDNN", "tensorrt": "TensorRT", "tensorrt-rtx": "TensorRT for RTX",
             "qairt": "Qualcomm AI Runtime (QAIRT)", "openvino": "OpenVINO", "cann": "CANN", "android-ndk": "Android NDK",
             "onnxruntime-sdk": "ONNX Runtime release (headers the plugin EP is built with)",
             "tim-vx": "TIM-VX", "rocm": "ROCm", "directml": "DirectML",
             "neuron-op-capability": "NeuroPilot op capability library", "hip-ep": "ROCm hip-ep"}


@functools.lru_cache(maxsize=None)
def submodule_text(rel: str) -> str:
    """A file of a submodule (fetched from GitHub when it is not checked out), or "" when it cannot be read."""
    try:
        return plan.build.sdk.dl.read_submodule(rel)
    except Exception as e:  # the requirements that depend on it are left out
        print(f"warning: cannot read {rel}: {e}", file=sys.stderr)
        return ""


def ort_build_default(option: str) -> str:
    """The default of an option of ORT's tools/ci_build/build.py, which the recipes do not set."""
    m = re.search(rf'"--{option}",[^)]*?default=("?)([^",)]+)\1', submodule_text("onnxruntime/tools/ci_build/build_args.py"))
    return m.group(2) if m else ""


def apple_deploy_target(triple: str) -> str:
    sysroot, _, settings = plan.build.APPLE_FRAMEWORKS[triple]
    try:
        params = json.loads(submodule_text(f"onnxruntime/tools/ci_build/github/apple/{settings}"))["build_params"]
    except ValueError:
        return ""
    return next((p.split("=", 1)[1] for p in params.get(sysroot, []) if p.startswith("--apple_deploy_target=")), "")


def os_of(triple: str) -> str:
    return plan.build.TRIPLES[triple][1]


def platform_requirements(core: dict) -> list[str]:
    """What the binaries of a target need on the user's machine (besides drivers and the EPs' requirements)."""
    triple, target_os = core["triple"], os_of(core["triple"])
    if target_os == "linux":
        return [LINUX_REQUIREMENT]
    if target_os == "windows":
        return [WINDOWS_REQUIREMENT]
    if target_os == "macos":
        return [f"macOS {plan.build.MACOS_DEPLOYMENT_TARGET} or newer"]
    if target_os == "apple":
        version = apple_deploy_target(triple)
        name = APPLE_OS[plan.build.APPLE_FRAMEWORKS[triple][0]]
        return [f"{name} {version} or newer"] if version else []
    if target_os == "android":
        api = ort_build_default("android_api")
        if not api.isdigit():
            return []
        version = ANDROID_VERSIONS.get(int(api))
        return [f"Android {version} (API level {api}) or newer" if version else f"Android API level {api} or newer"]
    return []


def rocm_of_image(image: str) -> str:
    """The ROCm release in the tag of a ROCm container image (rocm/dev-ubuntu-24.04:7.2.4-complete -> 7.2.4)."""
    m = re.search(r":(\d+(?:\.\d+)+)", image)
    return m.group(1) if m else ""


def core_ep_requirements(core: dict, ep: str, req: dict) -> list[str]:
    target_os, text = CORE_REQUIREMENTS.get(ep, ("", ""))
    if target_os != os_of(core["triple"]):
        return []
    return [text.format(rocm=rocm_of_image(req.get("docker", "")) or "(the core's container release)")]


def build_inputs(e: dict) -> dict:
    """The third-party SDKs (component names), the versions it pins of them, and the GPU architecture settings of a
    recipe entry."""
    needed = plan.recipe_inputs(e["recipe"], e["variant"], e["triple"], e["cuda_major"], e["eps"])[0]
    return {"sdks": sorted(plan.sdks_of(e)), "pins": {c: pin for c, _, pin in sorted(needed) if pin},
            "gpu": sorted(plan.gpu_kinds(e))}


def plugin_item(e: dict) -> dict:
    """Catalog entry of a plugin EP package for one triple."""
    eps, libs = PLUGINS[(e["recipe"], e["variant"])]
    if "windows" not in e["triple"]:  # the AMDGPU EP has a DirectML backend on Windows only
        eps, libs = [ep for ep in eps if ep != "dml"], [lib for lib in libs if lib != "directx-backend"]
    sdks, inputs = plan.sdks_of(e), build_inputs(e)
    reqs = [ROCM_REQUIREMENTS.get(os_of(e["triple"])) if "amd" in inputs["gpu"] else None,
            PLUGIN_REQUIREMENTS.get((e["recipe"], e["variant"]), PLUGIN_REQUIREMENTS.get(e["recipe"], {})).get(os_of(e["triple"]))]
    row, variant = eps[0], {}
    if v := PLUGIN_VARIANTS.get((e["recipe"], e["variant"])):
        row, variant = v[0], {"variant": v[1], "variant_label": v[2], "variant_note": v[3][os_of(e["triple"])], **v[4]}
    return {"package": e["package"], "triple": e["triple"], "ep": row, "eps": {ep: "plugin" for ep in eps}, "libs": libs,
            "cuda_major": e["cuda_major"] if "cuda" in sdks else None,
            "experimental": bool(e["experimental"]), "heavy": bool(e["heavy"]) or "cuda" in sdks,
            "sources": e["submodules"].split(), "requirements": [r for r in reqs if r], **variant, **inputs}


def platform_item(core: dict) -> dict:
    """What the core of a triple can contain, keyed by EP row (CORE_ROWS; "build_ep" is the in-tree EP it builds)."""
    eps = {}
    for ep, req in core["eps"].items():
        lib = f"onnxruntime_providers_{PROVIDER_LIBS[ep]}" if ep in PROVIDER_LIBS else None
        entry = plan.core_entry(core["triple"], [ep], (req.get("cuda") or ["13"])[-1])
        row = CORE_ROWS.get(ep, ep)
        eps[row] = {"form": "provider" if lib else "builtin", "lib": lib, "cuda": req.get("cuda"),
                    "experimental": bool(req.get("experimental")), "heavy": bool(req.get("heavy")),
                    "note": CORE_NOTES.get(ep, ""), "requirements": core_ep_requirements(core, ep, req),
                    **({"build_ep": ep} if row != ep else {}), **build_inputs(entry)}
        opts = LINK_OPTIONS.get(ep)
        target_os = "windows" if "-windows-" in core["triple"] else "linux" if core["triple"].endswith("-linux-gnu") else ""
        if opts and core["platform"] == "desktop" and target_os in opts["os"]:
            eps[row]["link"] = {"label": opts["label"], "default": next(iter(opts["choices"])), "choices": opts["choices"],
                                "plugin": opts["plugin"]}
    wasm = core["platform"] == "wasm"
    item = {"platform": core["platform"], "os": os_of(core["triple"]), "core_eps": {ep: eps[ep] for ep in EPS if ep in eps},
            "core_from": core.get("core_from"), "experimental": bool(core.get("experimental")),
            "note": PLATFORM_NOTES.get(core["platform"], ""),
            "default_core": [ep for ep in DEFAULT_CORE_EPS if ep in eps],
            "requirements": platform_requirements(core),
            "sdks": build_inputs(plan.core_entry(core["triple"], [], wasm_mode="cpu" if wasm else ""))["sdks"]}
    if wasm:
        emsdk = ort_build_default("emsdk_version") or "(ORT's emsdk version)"
        item["wasm_modes"] = {m: {"eps": v["eps"], "requirements": [r.format(emsdk=emsdk) for r in WASM_REQUIREMENTS[m]]}
                              for m, v in core["wasm_modes"].items()}
    return item


def sources() -> dict[str, dict]:
    """The submodules the packages are built from: path -> commit and repository."""
    out = subprocess.run(["git", "config", "--file", ".gitmodules", "--get-regexp", r"\.url$"], capture_output=True,
                         text=True, cwd=plan.ROOT, check=True).stdout.split()
    result = {}
    for key, url in zip(out[::2], out[1::2]):
        path = plan.git("config", "--file", ".gitmodules", key.replace(".url", ".path"))
        commit = plan.git("rev-parse", f"HEAD:{path}")
        repo = url.removesuffix(".git")
        result[path] = {"commit": commit, "url": repo, "tree": f"{repo}/tree/{commit}"}
    return result


def sdk_versions(platforms: dict, plugins: list[dict]) -> dict[str, dict[str, str]]:
    """The third-party SDK versions the builds would use now, per triple: component -> version, "<component>@<CUDA
    major>" for the components whose version depends on it. Each build resolves them again and records its own in
    BUILD_INFO.json."""
    inputs, result = plan.Inputs(), {}
    entries = [(t, plan.core_entry(t, [info.get("build_ep", ep)], cuda)) for t, p in platforms.items()
               for ep, info in p["core_eps"].items() for cuda in (info["cuda"] or ["13"])]
    entries += [(p["triple"], next(e for e in plan.entries() if e["package"] == p["package"] and e["triple"] == p["triple"]))
                for p in plugins]
    for triple, e in entries:
        cuda = e["cuda_major"]
        for comp, plat, pin in plan.recipe_inputs(e["recipe"], e["variant"], e["triple"], cuda, e["eps"])[0]:
            dependent = comp in plan.CUDA_DEPENDENT
            key = (comp, plat, cuda if dependent else "13", pin)
            inputs.sdk(*key)
            if version := inputs.versions.get(key):
                result.setdefault(triple, {})[f"{comp}@{cuda}" if dependent else comp] = version
    return result


def source() -> dict:
    """The commit of this repository the catalog describes (the recipes and submodules packages are built from)."""
    sha = os.environ.get("GITHUB_SHA") or plan.git("rev-parse", "HEAD")
    date = plan.git("show", "-s", "--format=%cI", sha)
    return {"type": "commit", "id": sha[:12], "url": f"https://github.com/{plan.REPO}/commit/{sha}", "created_at": date}


def cuda_archs_value(archs: list[str], ptx: bool) -> str:
    """CMAKE_CUDA_ARCHITECTURES of a selection: the real kernels of every SM version, and the PTX of the newest."""
    archs = sorted(archs, key=int)
    return ";".join([f"{a}-real" for a in archs] + ([f"{archs[-1]}-virtual"] if ptx and archs else []))


def gpu_options() -> dict:
    """The GPU architectures a request can select, with the defaults (the ci/build.py defaults)."""
    default = cuda_archs_value(DEFAULT_CUDA["archs"], DEFAULT_CUDA["ptx"])
    if default != plan.build.DEFAULT_CUDA_ARCHS:
        raise SystemExit("DEFAULT_CUDA does not match ci/build.py DEFAULT_CUDA_ARCHS")
    if bad := [t for t in DEFAULT_AMD_GPU_TARGETS if t not in AMD_GPU_TARGETS]:
        raise SystemExit(f"AMD_GPU_TARGETS lacks the default targets {bad}")
    return {"cuda": {"archs": CUDA_ARCHS, "exact": CUDA_EXACT_SMS, "default": DEFAULT_CUDA,
                     "drivers": CUDA_DRIVERS, "ptx_note": CUDA_PTX_NOTE},
            "amd": {"targets": {t: {"gpus": g, "os": list(o)} for t, (g, o) in AMD_GPU_TARGETS.items()},
                    "default": DEFAULT_AMD_GPU_TARGETS}}


def catalog(versions: bool = False) -> dict:
    """The catalog; versions=True also resolves the third-party SDK versions (network access)."""
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
        "default_variants": DEFAULT_VARIANTS,
        "hip_option": HIP_OPTION,
        "requires": REQUIRES,
        "third_party": EP_THIRD_PARTY,
        "components": COMPONENTS,
        "gpu": gpu_options(),
        "onnxruntime_version": submodule_text("onnxruntime/VERSION_NUMBER").strip() or None,
        "sources": sources(),
        "sdk_names": SDK_NAMES,
        "sdk_cuda_dependent": sorted(plan.CUDA_DEPENDENT),
        "sdk_versions": sdk_versions(platforms, plugins) if versions else None,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Generate catalog.json for bundle requests")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--versions", action="store_true", help="resolve the third-party SDK versions (network access)")
    args = ap.parse_args(argv)
    cat = catalog(args.versions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cat, indent=1), encoding="utf-8")
    print(f"{args.out}: {len(cat['triples'])} targets, {len(cat['plugins'])} plugin packages "
          f"(commit {cat['source']['id']})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
