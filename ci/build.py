#!/usr/bin/env python3
"""CI helper: build ONNX Runtime and its EPs from a recipe and stage the outputs (bin/ lib/ include/).

Each package build of a bundle request makes one call (through .github/actions/build-entry), e.g. a core:

    python ci/build.py --recipe ort --variant custom --eps cuda,openvino --triple x86_64-unknown-linux-gnu \\
        --work $RUNNER_TEMP/work --sdk-root $RUNNER_TEMP/sdk --stage stage

Third-party SDKs required by a recipe are downloaded automatically by ci/sdk.py (official sources, no account
needed), so the workflow does not prepare them separately. --dry-run only prints the commands, which is handy
for checking recipes locally.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import sdk

ROOT = Path(__file__).resolve().parent.parent

# target triple -> (SDK platform, operating system)
TRIPLES = {
    "x86_64-unknown-linux-gnu": ("linux-x86_64", "linux"),
    "aarch64-unknown-linux-gnu": ("linux-aarch64", "linux"),
    "x86_64-pc-windows-msvc": ("windows-x86_64", "windows"),
    "aarch64-pc-windows-msvc": ("windows-arm64", "windows"),
    "arm64ec-pc-windows-msvc": ("windows-arm64x", "windows"),      # ARM64EC (x64-compatible ABI on ARM64)
    "arm64x-pc-windows-msvc": ("windows-arm64x", "windows"),       # ARM64X (ARM64 + ARM64EC in one binary)
    "aarch64-apple-darwin": ("macos-arm64", "macos"),
    "x86_64-apple-darwin": ("macos-arm64", "macos"),               # cross-compiled on an Apple Silicon host
    "aarch64-apple-ios": ("macos-arm64", "apple"),
    "aarch64-apple-ios-sim": ("macos-arm64", "apple"),
    "x86_64-apple-ios": ("macos-arm64", "apple"),                  # iOS simulator on x86_64
    "aarch64-apple-ios-macabi": ("macos-arm64", "apple"),          # Mac Catalyst
    "x86_64-apple-ios-macabi": ("macos-arm64", "apple"),
    "aarch64-apple-tvos": ("macos-arm64", "apple"),
    "aarch64-apple-tvos-sim": ("macos-arm64", "apple"),
    "x86_64-apple-tvos": ("macos-arm64", "apple"),                 # tvOS simulator on x86_64
    "aarch64-apple-visionos": ("macos-arm64", "apple"),
    "aarch64-apple-visionos-sim": ("macos-arm64", "apple"),
    "aarch64-linux-android": ("linux-x86_64", "android"),         # cross-compiled with the NDK
    "armv7-linux-androideabi": ("linux-x86_64", "android"),
    "i686-linux-android": ("linux-x86_64", "android"),
    "x86_64-linux-android": ("linux-x86_64", "android"),
    "wasm32-unknown-emscripten": ("linux-x86_64", "wasm"),        # cross-compiled with emsdk
}
SHARED_EXT = {"linux": ".so", "android": ".so", "macos": ".dylib", "windows": ".dll"}
ANDROID_ABI = {"aarch64-linux-android": "arm64-v8a", "armv7-linux-androideabi": "armeabi-v7a",
               "i686-linux-android": "x86", "x86_64-linux-android": "x86_64"}
# Apple triple -> (sysroot, arch, build settings file in onnxruntime/tools/ci_build/github/apple)
APPLE_FRAMEWORKS = {
    "aarch64-apple-ios": ("iphoneos", "arm64", "default_full_apple_framework_build_settings.json"),
    "aarch64-apple-ios-sim": ("iphonesimulator", "arm64", "default_full_apple_framework_build_settings.json"),
    "x86_64-apple-ios": ("iphonesimulator", "x86_64", "default_full_apple_framework_build_settings.json"),
    "aarch64-apple-ios-macabi": ("macabi", "arm64", "default_full_ios_framework_build_settings.json"),
    "x86_64-apple-ios-macabi": ("macabi", "x86_64", "default_full_ios_framework_build_settings.json"),
    "aarch64-apple-tvos": ("appletvos", "arm64", "default_full_tvos_framework_build_settings.json"),
    "aarch64-apple-tvos-sim": ("appletvsimulator", "arm64", "default_full_tvos_framework_build_settings.json"),
    "x86_64-apple-tvos": ("appletvsimulator", "x86_64", "default_full_tvos_framework_build_settings.json"),
    "aarch64-apple-visionos": ("xros", "arm64", "default_vision_os_framework_build_settings.json"),
    "aarch64-apple-visionos-sim": ("xrsimulator", "arm64", "default_vision_os_framework_build_settings.json"),
}
DEFAULT_CUDA_ARCHS = "80-real;80-virtual"  # ORT drops a list that has only -virtual entries
# ROCm (TheRock wheels) used for the Windows MIGraphX source build (same as the AMDMIGraphX windows.yaml CI)
WINDOWS_ROCM_VERSION = "7.13.0"
DEFAULT_MIGRAPHX_GPU_TARGETS = "gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"
# OpenVINO of the OpenVINO plugin EP, which needs a newer one (cmake/openvino.cmake) than ORT's in-tree EP is built with
OPENVINO_PLUGIN_VERSION = "2026.4.1"
# Oldest macOS the desktop macOS packages support, as in ORT's official macOS packages (mac-cpu-packing-jobs.yml);
# without it the binaries require the macOS version of the runner's SDK
MACOS_DEPLOYMENT_TARGET = "14.0"
# AMDMIGraphX sources (a submodule on its develop branch, which the AMD plugin EPs follow)
MIGRAPHX_SRC = ROOT / "deps" / "amdmigraphx"


@dataclass
class Job:
    recipe: str
    variant: str
    triple: str
    work: Path
    stage: Path
    sdk_root: Path
    cuda_major: str
    cuda_archs: str
    dry_run: bool
    amd_gpu_targets: str = DEFAULT_MIGRAPHX_GPU_TARGETS
    platform: str = ""
    os: str = ""
    sdks: dict[str, Path] = field(default_factory=dict)
    # in-tree EPs of a core (--eps, see ci/bundle.py); "<ep>:<choice>" selects a linkage other than the default
    # (LINK_CHOICES), which ends up in link
    eps: list[str] = field(default_factory=list)
    link: dict[str, str] = field(default_factory=dict)
    # (component, platform, pinned version or ""), used by ci/plan.py for the build inputs hash
    needed: set[tuple[str, str, str]] = field(default_factory=set)
    # the GPU architecture settings the recipe compiles device code for ("cuda", "amd"); also part of the inputs hash
    gpu: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.platform, self.os = TRIPLES[self.triple]
        self.link = dict(e.split(":", 1) for e in self.eps if ":" in e)
        self.eps = [e.split(":", 1)[0] for e in self.eps]
        if bad := [f"{ep}:{c}" for ep, c in self.link.items() if c not in LINK_CHOICES.get(ep, ())]:
            raise SystemExit(f"unknown linkage {bad}; known: {LINK_CHOICES}")

    def need(self, *names: str, pins: dict[str, str] | None = None) -> None:
        names = tuple(n for n in names if n not in self.sdks)  # each SDK is prepared once per job
        if not names:
            return
        self.needed.update((n, self.platform, (pins or {}).get(n, "")) for n in names)
        if self.dry_run:
            self.sdks.update({n: self.sdk_root / n for n in names})
            print(f"[dry-run] prepare SDKs: {', '.join(names)} (platform {self.platform})")
            return
        self.sdks.update(sdk.prepare(list(names), self.platform, self.sdk_root, self.cuda_major, pins=pins))

    def cuda_architectures(self) -> str:
        """CMAKE_CUDA_ARCHITECTURES of the CUDA code the recipe compiles."""
        self.gpu.add("cuda")
        return self.cuda_archs

    def gpu_targets(self) -> str:
        """GPU_TARGETS (AMD gfx targets) of the MIGraphX build."""
        self.gpu.add("amd")
        return self.amd_gpu_targets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(job: Job, cmd: list, cwd: Path | None = None, env: dict | None = None) -> None:
    cmd = [str(c) for c in cmd]
    print(f"$ {shlex.join(cmd)}" + (f"   (cwd={cwd})" if cwd else ""), flush=True)
    if job.dry_run:
        return
    subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})}, check=True)


def copy_files(job: Job, patterns: list[str], base: Path, dest: Path) -> list[Path]:
    """Copy files matching the globs into dest (keeping symlinks) and return them; fail if nothing matches."""
    if job.dry_run:
        print(f"[dry-run] copy {patterns} @ {base} -> {dest}")
        return []
    copied = []
    dest.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for f in glob.glob(str(base / pattern), recursive=True):
            src = Path(f)
            if src.is_file() or src.is_symlink():
                target = dest / src.name
                if src.is_symlink() and os.name != "nt":
                    target.unlink(missing_ok=True)
                    target.symlink_to(os.readlink(src))
                else:
                    shutil.copy2(src, target)
                copied.append(target)
    if not copied:
        raise SystemExit(f"no build outputs found: {patterns} @ {base}")
    return copied


def stage_shared_libs(job: Job, base: Path, names: list[str]) -> None:
    """Stage shared libraries: on Windows .dll goes to bin/ and .lib/.pdb to lib/; elsewhere everything goes to lib/."""
    if job.os == "windows":
        copy_files(job, [f"**/{n}.dll" for n in names], base, job.stage / "bin")
        for ext in ("lib", "pdb"):
            try:
                copy_files(job, [f"**/{n}.{ext}" for n in names], base, job.stage / "lib")
            except SystemExit:
                pass  # plugin EPs usually have no import library, and PDBs are optional
    else:
        ext = SHARED_EXT[job.os]
        copy_files(job, [f"**/*{n}{ext}*" for n in names], base, job.stage / "lib")


def stage_headers(job: Job, src: Path, dest: str) -> None:
    """Copy a backend's interface headers (a directory tree) into include/<dest>/ of the package."""
    if job.dry_run:
        print(f"[dry-run] headers {src} -> include/{dest}")
        return
    if not src.is_dir():
        print(f"::warning::backend headers not found: {src}", flush=True)
        return
    shutil.copytree(src, job.stage / "include" / dest, dirs_exist_ok=True)


def stage_dev_files(job: Job, patterns: list[str], base: Path, dest: Path) -> None:
    """Copy a backend's import libraries / headers; a missing one is reported, but does not fail the build."""
    try:
        copy_files(job, patterns, base, dest)
    except SystemExit:
        print(f"::warning::backend development files not found: {patterns} in {base}", flush=True)


def stage_license(job: Job, name: str, base: Path, patterns: list[str]) -> None:
    """Copy the license texts of a bundled third-party component into third_party_licenses/<name>/."""
    try:
        copy_files(job, patterns, base, job.stage / "third_party_licenses" / name)
    except SystemExit:
        print(f"warning: no license text of {name} found in {base}", file=sys.stderr, flush=True)


def record_sdk_version(job: Job, component: str, version: str) -> None:
    """Record the version of a third-party component the build used in sdk-versions.json (written by ci/sdk.py for the
    SDKs it prepares), which ci/package.py copies into BUILD_INFO.json."""
    if job.dry_run:
        return
    path = job.sdk_root / "sdk-versions.json"
    versions = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    versions[component] = version
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(versions, indent=2), encoding="utf-8")


def set_origin_rpath(job: Job, lib_dir: Path, extra: str = "/opt/rocm/lib") -> None:
    """Let the libraries built here find each other in the package directory: the build-tree run paths point into the
    CI work directory. Only for libraries this project builds; vendor libraries are left unmodified."""
    if job.dry_run:
        return
    # the PyPI package ships a patchelf binary; PIP_BREAK_SYSTEM_PACKAGES allows it in the Ubuntu container images
    run(job, [sys.executable, "-m", "pip", "install", "--quiet", "patchelf"], env={"PIP_BREAK_SYSTEM_PACKAGES": "1"})
    candidates = [shutil.which("patchelf"), Path(sysconfig.get_path("scripts")) / "patchelf",
                  Path(sysconfig.get_path("scripts", f"{os.name}_user")) / "patchelf"]
    patchelf = next(str(c) for c in candidates if c and Path(c).exists())
    for f in sorted(lib_dir.glob("*.so*")):
        if f.is_file() and not f.is_symlink() and f.read_bytes()[:4] == b"\x7fELF":
            run(job, [patchelf, "--set-rpath", f"$ORIGIN:{extra}" if extra else "$ORIGIN", f])


def record_directml(job: Job) -> None:
    """Record the version of the Microsoft.AI.DirectML NuGet package the build restored. Like the other vendor runtimes
    that may only be redistributed with an application (CUDA, TensorRT, QAIRT, OpenVINO, ...), DirectML is not packaged:
    a bundle adds that version or its fetch-restricted script downloads it (ci/bundle.py)."""
    if job.dry_run:  # the resolve job has no submodules checked out
        return
    if job.recipe == "ep-amdgpu":
        path = ROOT / "eps/official/onnxruntime-ep-amdgpu/src/directx/CMakeLists.txt"
        pattern = r"install_nuget_package\(Microsoft\.AI\.DirectML\s+([^)\s]+)\)"
    else:
        path, pattern = ROOT / "onnxruntime/packages.config", r'id="Microsoft\.AI\.DirectML"\s+version="([^"]+)"'
    if m := re.search(pattern, path.read_text(encoding="utf-8")):
        record_sdk_version(job, "directml", m.group(1))
    else:
        raise SystemExit(f"the Microsoft.AI.DirectML version is not found in {path}")


def stage_ort_headers(job: Job, src: Path) -> None:
    """Copy the public ORT C/C++ API headers (for packages that are not produced by cmake --install)."""
    copy_files(job, ["include/onnxruntime/core/session/onnxruntime_*.h",
                     "include/onnxruntime/core/session/onnxruntime_*.inc"], src, job.stage / "include" / "onnxruntime")



def vs_generator() -> str:
    """CMake Visual Studio generator of the MSVC environment set up by msvc-dev-cmd (hosted runners ship VS 2026)."""
    major = os.environ.get("VisualStudioVersion", "18.0").split(".")[0]
    return {"17": "Visual Studio 17 2022"}.get(major, "Visual Studio 18 2026")


# ---------------------------------------------------------------------------
# ONNX Runtime main repository (tools/ci_build/build.py)
# ---------------------------------------------------------------------------

ORT_SRC = ROOT / "onnxruntime"


def cuda_flags(job: Job) -> list[str]:
    job.need("cuda", "cudnn")
    flags = ["--use_cuda", "--cuda_home", job.sdks["cuda"], "--cudnn_home", job.sdks["cudnn"], "--nvcc_threads", "1",
             "--cmake_extra_defines", f"CMAKE_CUDA_ARCHITECTURES={job.cuda_architectures()}"]
    if job.os == "windows":
        # ORT's build.py passes <cuda_home>/bin/nvcc without the .exe suffix, which CMake rejects as "not a full path
        # to an existing compiler tool"; extra defines come later on the command line and override it
        flags += ["--cmake_extra_defines", f"CMAKE_CUDA_COMPILER={job.sdks['cuda'] / 'bin' / 'nvcc.exe'}"]
    return flags


def tensorrt_flags(job: Job) -> list[str]:
    """TensorRT EP; ORT's build.py always builds the CUDA EP with it."""
    flags = cuda_flags(job)
    job.need("tensorrt")
    return flags + ["--use_tensorrt", "--tensorrt_home", job.sdks["tensorrt"]]


def dawn_backends(job: Job) -> list[str]:
    return {"windows": ["onnxruntime_ENABLE_DAWN_BACKEND_D3D12=1", "onnxruntime_ENABLE_DAWN_BACKEND_VULKAN=1"],
            "linux": ["onnxruntime_ENABLE_DAWN_BACKEND_VULKAN=1"]}.get(job.os, [])


def webgpu_flags(job: Job) -> list[str]:
    """WebGPU as an internal EP (static_lib), with the Dawn backends of the target. Dawn is linked into the core library
    unless the core links it as a shared library (webgpu:shared, the build of ORT's windows_webgpu.yml)."""
    defines = dawn_backends(job) + (["onnxruntime_BUILD_DAWN_SHARED_LIBRARY=ON"] if job.link.get("webgpu") == "shared" else [])
    return ["--use_webgpu", *(["--cmake_extra_defines", *defines] if defines else [])]


def nv_tensorrt_rtx_flags(job: Job) -> list[str]:
    job.need("cuda", "tensorrt-rtx")
    # onnxruntime_providers_nv.cmake runs find_package(CUDAToolkit), which does not see --cuda_home
    return ["--use_nv_tensorrt_rtx", "--tensorrt_rtx_home", job.sdks["tensorrt-rtx"], "--cuda_home", job.sdks["cuda"],
            "--cmake_extra_defines", f"CUDAToolkit_ROOT={job.sdks['cuda']}"]


def openvino_flags(job: Job) -> list[str]:
    job.need("openvino")  # OpenVINO_DIR / INTEL_OPENVINO_DIR are set by ci/sdk.py
    return ["--use_openvino", "CPU"]


def qnn_flags(job: Job) -> list[str]:
    """The QNN EP as a provider library (shared_lib, the default) or linked into the core library (qnn:static)."""
    job.need("qairt")
    kind = "static_lib" if job.link.get("qnn") == "static" else "shared_lib"
    return ["--use_qnn", kind, "--qnn_home", job.sdks["qairt"]]


def migraphx_flags(job: Job) -> list[str]:
    # built inside the rocm/dev-ubuntu container; ROCm and MIGraphX live in /opt/rocm
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    record_rocm_version(job, rocm)
    return ["--use_migraphx", "--migraphx_home", rocm]


def record_rocm_version(job: Job, rocm: Path, version: str = "") -> None:
    """Record the ROCm release the build used: the user installs the same one (ci/catalog.py COMPONENTS)."""
    info = rocm / ".info" / "version"
    if not version and info.exists():
        version = info.read_text(encoding="utf-8").strip().split("-")[0]
    if version:
        record_sdk_version(job, "rocm", version)


def cann_flags(job: Job) -> list[str]:
    job.need("cann")
    return ["--use_cann", "--cann_home", job.sdks["cann"]]


def vsinpu_flags(job: Job) -> list[str]:
    tim_vx = build_tim_vx(job)
    if not job.dry_run:
        os.environ["TIM_VX_INSTALL"] = str(tim_vx)
    return ["--use_vsinpu"]


# In-tree EP -> build.py arguments, for the cores of bundle requests (the eps of the core variants: custom, arm64x,
# android; the Apple frameworks select coreml / xnnpack in their settings instead). TensorRT includes the CUDA
# EP: ORT's build.py always builds them together.
CUSTOM_EPS: dict[str, Callable[[Job], list[str]]] = {
    "xnnpack": lambda job: ["--use_xnnpack"],
    "webgpu": webgpu_flags,
    "dml": lambda job: ["--use_dml"],
    "coreml": lambda job: ["--use_coreml"],
    "nnapi": lambda job: ["--use_nnapi"],
    "dnnl": lambda job: ["--use_dnnl"],
    "vitisai": lambda job: ["--use_vitisai"],
    "cuda": cuda_flags,
    "tensorrt": tensorrt_flags,
    "nv-tensorrt-rtx": nv_tensorrt_rtx_flags,
    "openvino": openvino_flags,
    "qnn": qnn_flags,
    "migraphx": migraphx_flags,
    "cann": cann_flags,
    "vsinpu": vsinpu_flags,
}
# Linkage choices of in-tree EPs (the default first; ci/catalog.py LINK_OPTIONS offers them in the page)
LINK_CHOICES = {"webgpu": ("static", "shared"), "qnn": ("shared", "static")}


def eps_flags(job: Job) -> list[str]:
    """build.py arguments of the selected in-tree EPs (job.eps) of a core."""
    unknown = sorted(set(job.eps) - CUSTOM_EPS.keys())
    if unknown:
        raise SystemExit(f"unknown in-tree EPs {unknown}; known: {', '.join(CUSTOM_EPS)}")
    eps = [ep for ep in job.eps if not (ep == "cuda" and "tensorrt" in job.eps)]  # TensorRT brings the CUDA EP
    return [flag for ep in eps for flag in CUSTOM_EPS[ep](job)]


def allow_onnx_printer_on_tvos(src: Path) -> None:
    """ONNX's printer.cc uses std::to_chars, which Apple's SDKs mark unavailable before iOS / tvOS 16.3. ORT's ONNX
    patch adds a fallback for iOS only; extend its condition to tvOS (the tvOS frameworks target 15.1). The two
    patched lines keep their count, so the patch's hunk headers stay valid."""
    path = src / "cmake/patches/onnx/onnx.patch"
    text = path.read_text(encoding="utf-8")
    old = ("+#if defined(__APPLE__) && defined(__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__) && \\\n"
           "+    __ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__ < 160300\n")
    new = ("+#if defined(__APPLE__) && ((defined(__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__ < 160300) || \\\n"
           "+    (defined(__ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__) && __ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__ < 160300))\n")
    if old in text:
        path.write_text(text.replace(old, new), encoding="utf-8", newline="")


def allow_cann_with_cuda(src: Path) -> None:
    """The provider bridge defines AllocateBufferWithOptions in both its CUDA and its CANN section, so a build with
    both providers fails with a redefinition; keep only the CUDA copy (both forward to the same host function)."""
    path = src / "onnxruntime/core/providers/shared_library/provider_bridge_provider.cc"
    text = path.read_text(encoding="utf-8")
    cann = text.index("#if defined(USE_CANN)\n")
    start = text.index("void* AllocateBufferWithOptions(", cann)
    if text.rfind("#if !defined(USE_CUDA)", cann, start) != -1:
        return  # already patched
    end = text.index("\n}\n", start) + 3
    path.write_text(text[:start] + "#if !defined(USE_CUDA) && !defined(USE_CUDA_PROVIDER_INTERFACE)\n" + text[start:end]
                    + "#endif\n" + text[end:], encoding="utf-8", newline="")


def ort_variant(job: Job, src: Path) -> tuple[list[str], Callable[[Job, Path], None] | None]:
    """Return (extra build.py arguments, custom staging function). A None staging function means cmake --install.

    Cores: custom (desktop targets), arm64x and android take the in-tree EPs of --eps; the wasm variants are
    the WebAssembly build modes. cuda-plugin and webgpu-plugin build plugin EP packages from the main repository."""
    v = job.variant
    if v == "custom":
        arch = ["--osx_arch", "x86_64"] if job.triple == "x86_64-apple-darwin" else []  # cross-compiled on arm64
        return arch + eps_flags(job), None
    if v == "arm64x":
        return eps_flags(job), None
    if v == "android":
        job.need("android-ndk")
        sdk_path = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT") or ""
        return ["--android", "--android_sdk_path", sdk_path, "--android_ndk_path", job.sdks["android-ndk"],
                "--android_abi", ANDROID_ABI[job.triple], *eps_flags(job)], None
    if v == "cuda-plugin":
        flags = cuda_flags(job) + ["--cmake_extra_defines", "onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON"]
        if job.os == "windows":
            # The plugin target passes /bigobj only to the CUDA host compiler, but cudnn_flash_attention.cc needs it
            # too (C1128); ORT's build.py sets CMAKE_CXX_FLAGS itself, so add it to the Release flags (MSVC defaults)
            flags += ["--cmake_extra_defines", "CMAKE_CXX_FLAGS_RELEASE=/O2 /Ob2 /DNDEBUG /bigobj"]
        return flags, \
            lambda j, b: stage_shared_libs(j, b, ["onnxruntime_providers_cuda"])
    if v == "webgpu-plugin":
        names = ["onnxruntime_providers_webgpu"] + (["dxcompiler"] if job.os == "windows" else [])  # same as the official package
        backends = dawn_backends(job)
        return ["--use_webgpu", "shared_lib", "--wgsl_template", "static",
                *(["--cmake_extra_defines", *backends] if backends else [])], \
            lambda j, b: stage_shared_libs(j, b, names)
    if v == "wasm-static":
        return ["--build_wasm_static_lib", "--enable_wasm_simd", "--disable_rtti"], \
            lambda j, b: (copy_files(j, ["**/libonnxruntime_webassembly.a"], b, j.stage / "lib"), stage_ort_headers(j, src))
    if v.startswith("wasm"):
        # Like the official Release builds, catch exceptions only at the API boundary (JSPI uses native wasm
        # exceptions instead): catching them everywhere makes the LTO link need far more memory than the runners have
        catch = ["--enable_wasm_api_exception_catching"]
        extra = {"wasm": catch,
                 "wasm-jsep": ["--use_jsep", "--use_webnn", *catch],
                 "wasm-webgpu": ["--use_webgpu", "--use_webnn", *catch],
                 "wasm-webgpu-jspi": ["--use_webgpu", "--use_webnn", "--enable_wasm_jspi"]}[v]
        return ["--build_wasm", "--enable_wasm_simd", "--enable_wasm_threads", "--target", "onnxruntime_webassembly",
                "--disable_rtti", *extra], \
            lambda j, b: copy_files(j, ["ort-wasm-*.wasm", "ort-wasm-*.mjs"], b, j.stage / "lib")
    raise SystemExit(f"unknown ORT variant: {v}")


def build_ort(job: Job) -> None:
    src = ORT_SRC
    flags, stager = ort_variant(job, src)
    if "--use_cann" in flags and "--use_cuda" in flags and not job.dry_run:
        allow_cann_with_cuda(src)
    build_dir = job.work / "build"
    base = [sys.executable, src / "tools/ci_build/build.py", "--config", "Release", "--update", "--build", "--parallel",
            "--skip_tests", "--skip_submodule_sync", "--compile_no_warning_as_error",
            # the tests are not run here, and cmake --install would package onnx_test_runner
            "--cmake_extra_defines", "onnxruntime_BUILD_UNIT_TESTS=OFF"]
    if not job.variant.startswith("wasm"):
        base.append("--build_shared_lib")
    if job.os == "linux":
        # cmake --install drops the $ORIGIN run path the providers are linked with, so the oneDNN provider would not find
        # the libdnnl.so.3 packaged next to it
        base += ["--cmake_extra_defines", "CMAKE_INSTALL_RPATH=$ORIGIN"]
    if job.os == "windows" and not job.triple.startswith("x86_64"):
        # The MSVC 14.5x armasm64 rejects the /arch:armv8.2 flag KleidiAI passes to its .S kernels
        base.append("--no_kleidiai")
    if hasattr(os, "geteuid") and os.geteuid() == 0:  # the rocm/dev-ubuntu container runs as root
        base.append("--allow_running_as_root")
    if job.variant == "arm64x":
        # Same two-pass build as the official win-arm64x package (templates/build-win-arm64x-steps.yml):
        # an ARM64 pass generates the .def/.lib, then the ARM64EC pass produces the real ARM64X binaries.
        vs = ["--cmake_generator", vs_generator()]
        run(job, base + vs + ["--arm64", "--buildasx", "--build_dir", build_dir / "arm64"] + flags, cwd=src)
        run(job, base + vs + ["--arm64ec", "--buildasx", "--build_dir", build_dir] + flags, cwd=src)
    elif job.os == "windows" and not job.triple.startswith("x86_64") and "--use_xnnpack" in flags:
        # With Ninja on an ARM64 host CMAKE_SYSTEM_PROCESSOR is "ARM64", which ORT's and XNNPACK's CMake do not
        # recognize, so no NEON kernels get built; the Visual Studio generator passes -A ARM64 instead
        # MSVC has no arm_fp16.h, which XNNPACK's FP16 scalar kernels include
        flags = flags + ["--cmake_extra_defines", "XNNPACK_ENABLE_ARM_FP16_SCALAR=OFF"]
        run(job, base + ["--cmake_generator", vs_generator(), "--build_dir", build_dir] + flags, cwd=src)
    else:
        # On Windows the workflow sets up the MSVC environment with msvc-dev-cmd first
        run(job, base + ["--cmake_generator", "Ninja", "--build_dir", build_dir] + flags, cwd=src)
    config_dir = build_dir / "Release"
    if stager:
        stager(job, config_dir)
    else:
        install_ort(job, config_dir)
    if "--use_dnnl" in flags:  # oneDNN (Apache-2.0) is built from source for its provider, but not installed with it
        stage_shared_libs(job, config_dir / "dnnl" / "install", ["dnnl"])
        stage_license(job, "oneDNN", config_dir / "dnnl" / "src" / "dnnl" / "src", ["LICENSE", "THIRD-PARTY-PROGRAMS"])
    if job.os == "windows" and "--use_webgpu" in flags and job.variant != "webgpu-plugin":
        # Dawn loads the DXC shader compiler it builds (dxcompiler.dll) at run time, as in the WebGPU plugin package.
        # Ninja puts it into the config directory, the Visual Studio generator (Windows ARM64) into Release/ below it.
        copy_files(job, ["dxcompiler.dll", "Release/dxcompiler.dll"], config_dir, job.stage / "bin")
    if job.os == "windows" and "--use_webgpu" in flags:
        stage_license(job, "DirectXShaderCompiler", build_dir,
                      [f"Release/_deps/dawn-src/third_party/directx-shader-compiler/src/{n}"
                       for n in ("LICENSE.TXT", "ThirdPartyNotices.txt")])
    if job.link.get("webgpu") == "shared":  # Dawn (BSD-3-Clause) as a separate library next to the core library
        stage_shared_libs(job, config_dir, ["webgpu_dawn"])
        stage_license(job, "Dawn", build_dir, ["Release/_deps/dawn-src/LICENSE"])
    if "--use_dml" in flags:
        record_directml(job)
    if "--use_vsinpu" in flags:  # TIM-VX (MIT), built from source by build_tim_vx
        copy_files(job, ["lib/libtim-vx.so*"], job.work / "tim-vx-install", job.stage / "lib")
        stage_license(job, "TIM-VX", job.sdks["tim-vx"], ["LICENSE"])


def install_ort(job: Job, config_dir: Path) -> None:
    run(job, ["cmake", "--install", config_dir, "--prefix", job.stage, "--config", "Release"])
    if job.os == "windows" and not job.dry_run:
        # The providers are MODULE libraries, which install() puts into lib/; they are loaded from next to onnxruntime.dll
        for dll in (job.stage / "lib").glob("*.dll"):
            shutil.move(dll, job.stage / "bin" / dll.name)


def build_ort_apple(job: Job) -> None:
    """Apple mobile platforms: one sysroot/arch dynamic framework per job, built the same way as the official pods."""
    src = ORT_SRC
    sysroot, arch, settings = APPLE_FRAMEWORKS[job.triple]
    settings = src / "tools/ci_build/github/apple" / settings
    build_dir = job.work / "build"
    if sysroot.startswith("appletv") and not job.dry_run:
        allow_onnx_printer_on_tvos(src)
    if not job.dry_run:
        data = json.loads(settings.read_text(encoding="utf-8"))
        # the in-tree EPs of the core (--eps) instead of the official CoreML + XNNPACK
        selectable = {"--use_coreml": "coreml", "--use_xnnpack": "xnnpack"}
        for key, params in data["build_params"].items():
            data["build_params"][key] = [p for p in params if p not in selectable or selectable[p] in job.eps]
        unknown = sorted(set(job.eps) - set(selectable.values()))
        if unknown:
            raise SystemExit(f"the Apple frameworks support coreml / xnnpack, not {unknown}")
        data["build_params"]["base"] += [p for p, ep in selectable.items()
                                         if ep in job.eps and p not in data["build_params"]["base"]]
        if sysroot == "macabi":
            # ORT's build.py turns warnings into errors for Catalyst (unlike iOS / tvOS / visionOS), and the CoreML EP
            # uses APIs newer than the 14.0 deployment target without availability guards
            data["build_params"]["macabi"].append("--compile_no_warning_as_error")
        settings = job.work / settings.name
        settings.parent.mkdir(parents=True, exist_ok=True)
        settings.write_text(json.dumps(data, indent=2), encoding="utf-8")
    run(job, [sys.executable, src / "tools/ci_build/github/apple/build_apple_framework.py",
              settings, "--build_dir", build_dir, "--config", "Release",
              "--build_dynamic_framework", "--only_build_single_sysroot_arch_framework", sysroot, arch], cwd=src)
    config_dir = build_dir / "intermediates" / f"{sysroot}_{arch}" / "Release"
    framework = config_dir / f"Release-{sysroot}" / "onnxruntime.framework"
    if sysroot == "macabi" and not framework.exists():  # Catalyst builds use Makefiles, which add no -<sysroot> dir
        framework = config_dir / "onnxruntime.framework"
    print(f"stage {framework}", flush=True)
    if job.dry_run:
        return
    (job.stage / "lib").mkdir(parents=True, exist_ok=True)
    shutil.copytree(framework, job.stage / "lib" / "onnxruntime.framework", symlinks=True, dirs_exist_ok=True)
    stage_ort_headers(job, src)


# ---------------------------------------------------------------------------
# Open-source dependencies that must be built first
# ---------------------------------------------------------------------------

def build_tim_vx(job: Job) -> Path:
    """On x86_64, TIM-VX builds against the VIV simulator driver shipped in its repository (prebuilt-sdk/x86_64_linux)."""
    job.need("tim-vx")
    src, install = job.sdks["tim-vx"], job.work / "tim-vx-install"
    viv = src / "prebuilt-sdk" / "x86_64_linux"
    if not job.dry_run:  # linking ORT needs the VIV driver libraries that libtim-vx depends on
        os.environ["VIVANTE_SDK_DIR"] = str(viv)
        os.environ["LD_LIBRARY_PATH"] = f"{viv / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    run(job, ["cmake", "-G", "Ninja", "-S", src, "-B", job.work / "tim-vx-build", "-DCMAKE_BUILD_TYPE=Release",
              f"-DCMAKE_INSTALL_PREFIX={install}", "-DTIM_VX_BUILD_EXAMPLES=OFF", "-DTIM_VX_ENABLE_TEST=OFF"])
    run(job, ["cmake", "--build", job.work / "tim-vx-build", "--target", "install"])
    return install


def build_migraphx_linux(job: Job) -> tuple[Path, Path]:
    """Build MIGraphX (develop, which the AMD EPs follow) with its GPU backend, the same way AMDMIGraphX's ci.yaml does:
    inside its rocm/migraphx-ci-ubuntu image, which has ROCm in /opt/rocm and the dependencies in /usr/local.
    Returns (MIGraphX install prefix, ROCm root)."""
    src, b, install = MIGRAPHX_SRC, job.work / "migraphx-build", job.work / "migraphx-install"
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    run(job, ["cmake", "-S", src, "-B", b, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
              f"-DCMAKE_C_COMPILER={rocm / 'llvm/bin/clang'}", f"-DCMAKE_CXX_COMPILER={rocm / 'llvm/bin/clang++'}",
              "-DMIGRAPHX_ENABLE_GPU=On", "-DMIGRAPHX_ENABLE_PYTHON=Off", "-DBUILD_TESTING=Off",
              f"-DGPU_TARGETS={job.gpu_targets()}", f"-DCMAKE_INSTALL_PREFIX={install}"])
    run(job, ["cmake", "--build", b, "--config", "Release"])
    run(job, ["cmake", "--install", b])
    record_rocm_version(job, rocm)
    return install, rocm


def install_rocm_wheels(job: Job, devices: bool = True) -> tuple[Path, Path]:
    """Install ROCm from TheRock's Python wheels, as AMDMIGraphX's windows.yaml does: unlike the multi-arch SDK tarball,
    the device-<gfx> packages also bring the device libraries HIP code needs (devices=False leaves them out, for builds
    that compile no device code). Returns (ROCm root, CMake prefix)."""
    targets = job.gpu_targets().split(";") if devices else []
    extras = ",".join(["libraries", "devel", *(f"device-{t}" for t in targets)])
    # The SDK unpacks to several GB: keep it in a virtual environment in the work directory (the runner's large data
    # disk) rather than in the Python installation, whose disk (C:) runs out of space
    venv = job.work / "rocm-venv"
    run(job, [sys.executable, "-m", "venv", venv])
    scripts = venv / ("Scripts" if os.name == "nt" else "bin")
    run(job, [scripts / "python", "-m", "pip", "install", "--index-url", "https://repo.amd.com/rocm/whl-multi-arch/",
              f"rocm[{extras}]=={WINDOWS_ROCM_VERSION}"])
    rocm_sdk = scripts / "rocm-sdk"
    run(job, [rocm_sdk, "init"])
    record_rocm_version(job, Path(), WINDOWS_ROCM_VERSION)
    if job.dry_run:
        return job.sdk_root / "rocm", job.sdk_root / "rocm" / "lib" / "cmake"
    path = lambda kind: Path(subprocess.run([str(rocm_sdk), "path", kind], capture_output=True, text=True,
                                            check=True).stdout.strip())
    return path("--root"), path("--cmake")


def build_migraphx_windows(job: Job, deps_only: bool = False) -> tuple[Path, Path]:
    """Build MIGraphX with its GPU backend on Windows, following AMDMIGraphX's .github/workflows/windows.yaml.

    ROCm comes from TheRock's Python wheels; MIGraphX's own dependencies (including rocMLIR) are built by rbuild.
    deps_only stops after the dependencies (the first stage of build_migraphx). Returns (MIGraphX install prefix,
    ROCm root).
    """
    src = MIGRAPHX_SRC
    install, cget = job.work / "migraphx-install", job.work / "cget"
    # the MIGraphX install of the build cache is reused for the same commit and GPU targets
    marker, built = install / ".migraphx-commit", f"{git_commit(src)} {job.gpu_targets()}"
    if not job.dry_run and marker.exists() and marker.read_text(encoding="utf-8").strip() == built:
        print(f"MIGraphX {built} is already built (build cache): {install}", flush=True)
        # building against it needs ROCm, but not the device libraries: the EP repository compiles no device code
        return install, (install_rocm_wheels(job, devices=False)[0] if not deps_only else Path())
    run(job, [sys.executable, "-m", "pip", "install", "https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz"])
    if not job.dry_run and (cget / "hash").exists():
        # cget may come from the Actions cache (see build_cache). As in AMDMIGraphX's windows.yaml, compare the
        # dependency hash ourselves: rbuild prepare on Windows keeps stale dependencies when it does not match.
        expected = subprocess.run(["rbuild", "hash", "-s", "gh-win-gpu"], cwd=src, capture_output=True, text=True,
                                  check=True).stdout.strip()
        if (cget / "hash").read_text(encoding="utf-8").strip() != expected:
            print(f"MIGraphX dependency hash changed ({expected}); rebuilding the dependencies", flush=True)
            shutil.rmtree(cget)
        elif deps_only:
            print(f"the MIGraphX dependencies {expected} are already built (build cache): {cget}", flush=True)
            return install, Path()
    rocm, rocm_cmake = install_rocm_wheels(job, devices=not deps_only)
    path = os.pathsep.join([str(rocm / "lib" / "llvm" / "bin"), str(rocm / "bin"), os.environ.get("PATH", "")])
    # rocMLIR has long paths; build the dependencies in a short directory like the upstream CI does
    short = Path(job.work.anchor) / "b"
    run(job, ["rbuild", "prepare", "-d", cget, "-s", "gh-win-gpu", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
              "-DLLVM_DISABLE_ASSEMBLY_FILES=On", "-DLLVM_ENABLE_DIA_SDK=Off"], cwd=src,
        env={"PATH": path, "CC": "clang-cl", "CXX": "clang-cl", "CXXFLAGS": "/std:c++17", "CGET_BUILD_PATH": str(short)})
    if deps_only:
        return install, rocm
    # as on Linux: without BUILD_TESTING=Off the build also compiles MIGraphX's test executables
    run(job, ["rbuild", "build", "-d", cget, "-s", "gh-win-gpu", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
              "-DBUILD_TESTING=Off", "-DMIGRAPHX_ENABLE_PYTHON=Off",
              f"-DGPU_TARGETS={job.gpu_targets()}", f"-DCMAKE_INSTALL_PREFIX={install}"], cwd=src,
        env={"PATH": path, "CC": "clang", "CXX": "clang++", "CMAKE_PREFIX_PATH": f"{rocm_cmake};{cget}"})
    run(job, ["cmake", "--install", src / "build", "--prefix", install])
    if not job.dry_run:
        marker.write_text(built, encoding="utf-8")  # lets a cached install be reused (see build_cache)
    return install, rocm


# ---------------------------------------------------------------------------
# Plugin EP repositories
# ---------------------------------------------------------------------------

def ort_home(job: Job) -> Path:
    """Official ORT prebuilt package, completed with the layout plugin CMake projects expect (include/onnxruntime, lib64, CMake config)."""
    job.need("onnxruntime-sdk")
    home = job.sdks["onnxruntime-sdk"]
    if job.dry_run:
        return home
    nested = home / "include" / "onnxruntime"
    nested.mkdir(exist_ok=True)
    for h in (home / "include").iterdir():
        if h.is_file() and not (nested / h.name).exists():
            shutil.copy2(h, nested / h.name)
    if job.os == "linux" and not (home / "lib64").exists():
        (home / "lib64").symlink_to("lib")
    cmake_dir = home / "lib" / "cmake" / "onnxruntime"
    version_file = cmake_dir / "onnxruntimeConfigVersion.cmake"
    if not cmake_dir.exists():  # the Windows prebuilt package has no CMake config, so add a minimal imported target
        cmake_dir.mkdir(parents=True)
        lib = "onnxruntime.lib" if job.os == "windows" else "libonnxruntime.dylib" if job.os == "macos" else "libonnxruntime.so"
        dll = "IMPORTED_LOCATION \"${_root}/lib/onnxruntime.dll\"\n    IMPORTED_IMPLIB" if job.os == "windows" else "IMPORTED_LOCATION"
        (cmake_dir / "onnxruntimeConfig.cmake").write_text(
            'get_filename_component(_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)\n'
            "if(NOT TARGET onnxruntime::onnxruntime)\n"
            "  add_library(onnxruntime::onnxruntime SHARED IMPORTED)\n"
            "  set_target_properties(onnxruntime::onnxruntime PROPERTIES\n"
            f'    {dll} "${{_root}}/lib/{lib}"\n'
            '    INTERFACE_INCLUDE_DIRECTORIES "${_root}/include;${_root}/include/onnxruntime")\n'
            "endif()\n", encoding="utf-8")
    if not version_file.exists() and (home / "VERSION_NUMBER").exists():  # for find_package(onnxruntime <version>)
        version = (home / "VERSION_NUMBER").read_text(encoding="utf-8").strip()
        version_file.write_text(
            f'set(PACKAGE_VERSION "{version}")\n'
            "if(PACKAGE_FIND_VERSION VERSION_GREATER PACKAGE_VERSION)\n"
            "  set(PACKAGE_VERSION_COMPATIBLE FALSE)\n"
            "else()\n"
            "  set(PACKAGE_VERSION_COMPATIBLE TRUE)\n"
            "  if(PACKAGE_FIND_VERSION STREQUAL PACKAGE_VERSION)\n"
            "    set(PACKAGE_VERSION_EXACT TRUE)\n"
            "  endif()\n"
            "endif()\n", encoding="utf-8")
    return home


def need_plugin_ort(job: Job, path: str, pattern: str) -> None:
    """Prepare the ORT release (ort_home) that a plugin's own build names, the first group of pattern in the file path
    of its submodule: its code follows the experimental API of that release, which later releases change."""
    try:  # read at the gitlink commit when the submodule is not checked out (the resolve job)
        m = re.search(pattern, sdk.dl.read_submodule(path))
    except RuntimeError:
        m = None
    job.need("onnxruntime-sdk", pins={"onnxruntime-sdk": m.group(1)} if m else None)


def build_ep_tensorrt(job: Job) -> None:
    job.need("cuda", "cudnn", "tensorrt")
    src, b = ROOT / "eps/official/onnxruntime-ep-tensorrt", job.work / "build"
    trt, cuda = job.sdks["tensorrt"], job.sdks["cuda"]
    if job.os == "linux" and not job.dry_run:  # on Linux the CMake project links ${TENSORRT_HOME}/libnvinfer.so.N directly
        for so in (trt / "lib").glob("lib*.so*"):
            if not (trt / so.name).exists():
                (trt / so.name).symlink_to(so)
    nvcc = cuda / "bin" / ("nvcc.exe" if job.os == "windows" else "nvcc")
    run(job, ["cmake", "-G", "Ninja", "-S", src, "-B", b, "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX={job.stage}",
              f"-DCMAKE_CUDA_ARCHITECTURES={job.cuda_architectures()}", f"-DCMAKE_CUDA_COMPILER={nvcc}",
              f"-DTENSORRT_HOME={trt}", f"-DORT_HOME={ort_home(job)}", f"-DTRT_MAJOR_VERSION={trt_major(job)}"])
    run(job, ["cmake", "--build", b, "--parallel"])
    # not cmake --install: it also installs the protobuf / ONNX / flatbuffers dependencies and the TensorRT DLLs
    stage_shared_libs(job, b, ["onnxruntime_ep_tensorrt"])


def trt_major(job: Job) -> str:
    version_file = job.sdk_root / "sdk-versions.json"
    if job.dry_run or not version_file.exists():
        return "10"
    return json.loads(version_file.read_text(encoding="utf-8")).get("tensorrt", "10").split(".")[0]


def build_ep_nv_tensorrt_rtx(job: Job) -> None:
    job.need("cuda", "tensorrt-rtx")
    src, b = ROOT / "eps/vendor/nvidia-tensorrt-rtx-ep", job.work / "build"
    need_plugin_ort(job, "eps/vendor/nvidia-tensorrt-rtx-ep/cmake/onnxruntime.cmake", r'ONNXRUNTIME_VERSION "([\d.]+)"')
    # as in docs/build-guide.md; the unit tests run their executable after linking (gtest_discover_tests), which
    # fails on a runner without an NVIDIA GPU
    args = [f"-DCUDAToolkit_ROOT={job.sdks['cuda']}", f"-DONNXRUNTIME_ROOT={ort_home(job)}",
            f"-DTRT_RTX_ROOT={job.sdks['tensorrt-rtx']}", f"-DPython3_EXECUTABLE={sys.executable}",
            "-DBUILD_TESTS=OFF", "-DBUILD_BENCHMARK=OFF"]
    if job.os == "windows":
        # the Visual Studio generator; Windows ARM64 is cross-compiled on an x64 host, with a host protoc
        arm64 = job.triple.startswith("aarch64")
        args += ["-G", vs_generator(), "-A", "ARM64" if arm64 else "x64"]
        args += ["-DUSE_PRECOMPILED_HOST_PROTOC=ON"] if arm64 else []
        run(job, ["cmake", "-S", src, "-B", b, *args])
        run(job, ["cmake", "--build", b, "--config", "Release", "--parallel"])
    else:  # Ninja runs CMAKE_BUILD_PARALLEL_LEVEL jobs
        run(job, ["cmake", "-S", src, "-B", b, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", *args])
        run(job, ["cmake", "--build", b])
    stage_shared_libs(job, b, ["onnxruntime_providers_nv_tensorrt_rtx"])


def migraphx_staged(recipe: str, variant: str, triple: str) -> bool:
    """Whether a package build needs the Windows MIGraphX that build_migraphx builds in jobs of its own beforehand."""
    return recipe == "ep-amdgpu" and variant != "dml" and TRIPLES[triple][1] == "windows"


def build_migraphx(job: Job) -> None:
    """The MIGraphX the Windows AMD plugin EPs are built with, in two stages that bundle.yml runs as jobs of their own
    before the packages: the dependencies rbuild builds (variant "deps", about 3.5 hours; rocMLIR is an LLVM build),
    then MIGraphX itself (variant "build", about 2 hours). Each stage fits the 6-hour job limit of a hosted runner,
    and both AMD packages restore the result from the build cache (build_cache) instead of building it twice."""
    if job.os != "windows" or job.variant not in ("deps", "build"):
        raise SystemExit("the migraphx recipe builds the Windows MIGraphX in the stages deps and build")
    if not job.dry_run:  # a path of the build cache entry (build_cache), also when only the dependencies are built
        (job.work / "migraphx-install").mkdir(parents=True, exist_ok=True)
    build_migraphx_windows(job, deps_only=job.variant == "deps")


def build_ep_amdgpu(job: Job) -> None:
    src, b = ROOT / "eps/official/onnxruntime-ep-amdgpu", job.work / "build"
    cmd = [sys.executable, src / "tools/ci_build/build.py", "--build_dir", b, "--config", "Release", "--update", "--build",
           "--parallel", "--cmake_generator", "Ninja", "--onnxrt_home", ort_home(job)]
    if job.variant == "dml":
        cmd += ["--use_dml"]
        names = ["directx-ep"]
    else:
        migraphx, rocm = build_migraphx_windows(job) if job.os == "windows" else build_migraphx_linux(job)
        cmd += ["--migraphx_home", migraphx, "--hip_path", rocm]
        if job.variant in ("amdgpu", "amdgpu-hip"):  # AMDGPU EP; implies MIGraphX (and DirectML on Windows)
            cmd += ["--use_amdgpu"]
            names = ["amdgpu-ep", "migraphx-backend", "directx-backend"]
            if job.variant == "amdgpu-hip":
                # hip-backend, the shim that loads hip-ep's prebuilt HIP EP (hipgpu), which fetch-restricted downloads:
                # record the hip-ep release the package is paired with
                names.append("hip-backend")
                if not job.dry_run:
                    hip_ep = sdk.dl.component("hip-ep")
                    record_sdk_version(job, "hip-ep", hip_ep.resolve(sdk.dl.Context(job.platform, job.cuda_major,
                                                                                     "ubuntu24", {}), hip_ep.default).version)
            else:  # amdgpu-ep fails to load when it is built with the HIP backend and hipgpu is missing
                cmd += ["--cmake_extra_defines", "USE_HIP=OFF"]
        else:
            cmd += ["--use_migraphx"]
            names = ["migraphx-ep"]
    if job.os == "windows":
        # With VSINSTALLDIR set, the script's _build_environment() replaces the whole environment with an empty one
        # (no PATH, no compiler); without it, the script loads vcvars64.bat of the latest Visual Studio itself.
        os.environ.pop("VSINSTALLDIR", None)
    run(job, cmd, cwd=src)
    stage_shared_libs(job, b / "Release", names)
    if job.os == "windows" and job.variant in ("dml", "amdgpu", "amdgpu-hip"):  # the DirectML EP / backend
        record_directml(job)
    if job.variant != "dml":  # MIGraphX (MIT) is built from its develop branch here, so bundle it with its license
        if job.os == "windows":
            copy_files(job, ["bin/migraphx*.dll"], migraphx, job.stage / "bin")
            stage_dev_files(job, ["lib/migraphx*.lib"], migraphx, job.stage / "lib")
        else:
            copy_files(job, ["lib/libmigraphx*.so*", "lib/migraphx/lib/*.so*"], migraphx, job.stage / "lib")
            set_origin_rpath(job, job.stage / "lib")
        stage_headers(job, migraphx / "include" / "migraphx", "migraphx")
        stage_license(job, "MIGraphX", MIGRAPHX_SRC, ["LICENSE*"])
    if job.variant in ("amdgpu", "amdgpu-hip"):  # the HIP headers, and the import library on Windows (MIT)
        stage_headers(job, rocm / "include" / "hip", "hip")
        if job.os == "windows":
            stage_dev_files(job, ["lib/amdhip64.lib"], rocm, job.stage / "lib")
        stage_license(job, "HIP", rocm, ["share/doc/hip/LICENSE*", "share/doc/*hip*/LICENSE*"])


QNN_TASKS = {
    "x86_64-pc-windows-msvc": "build_ort_windows_x86_64",
    "aarch64-pc-windows-msvc": "build_ort_windows_arm64",
    "arm64ec-pc-windows-msvc": "build_ort_windows_arm64ec",
    "arm64x-pc-windows-msvc": "build_ort_windows_arm64x",
    "x86_64-unknown-linux-gnu": "build_ort_linux_x86_64",
    "aarch64-unknown-linux-gnu": "build_ort_linux_aarch64_manylinux_2_34",
    "aarch64-linux-android": "build_ort_android_aarch64",
}


def build_ep_qnn(job: Job) -> None:
    src = ROOT / "eps/official/onnxruntime-qnn"
    job.need("qairt")
    cmd = [sys.executable, "qcom/build_and_test.py", "--config", "Release", "--qairt-sdk", job.sdks["qairt"]]
    if job.os != "android":  # Android needs ORT as an AAR, so the repository script builds it from source
        # Build against the ORT release the repository's own CI uses: its code follows that release's experimental
        # API, which later releases changed.
        try:  # read at the gitlink commit when the submodule is not checked out (the resolve job)
            packages = sdk.dl.read_submodule("eps/official/onnxruntime-qnn/qcom/packages.yml")
            ort = sdk.dl.yaml_block(packages, "ort_prebuilt_windows_x64")["version"]
            job.need("onnxruntime-sdk", pins={"onnxruntime-sdk": ort})
        except RuntimeError:
            job.need("onnxruntime-sdk")
        # the extracted package root: onnxruntime_prebuilt.cmake copies files from <root>/lib
        cmd += ["--ort-prebuilt", job.sdks["onnxruntime-sdk"]]
    if job.os == "linux" and not job.dry_run:
        # The Linux builds run qcom/scripts/linux/build.sh in a container that mounts only this repository, where git
        # (`rev-parse --show-toplevel`, `submodule sync`) must work; the .git files of a submodule checkout point
        # outside of it, so give this repository and its submodules their own git dirs
        nested = subprocess.run(["git", "submodule", "foreach", "--recursive", "--quiet", 'echo "$displaypath"'],
                                cwd=src, capture_output=True, text=True, check=True).stdout.splitlines()
        for repo in [src, *(src / d for d in nested)]:  # outer first; the originals stay in place for the nested ones
            gitfile = repo / ".git"
            if gitfile.is_file():
                gitdir = (repo / gitfile.read_text(encoding="utf-8").split(":", 1)[1].strip()).resolve()
                gitfile.unlink()
                shutil.copytree(gitdir, gitfile, symlinks=True)
                subprocess.run(["git", "config", "--file", str(gitfile / "config"), "--unset", "core.worktree"],
                               check=False)
        # The aarch64 build mounts the package cache into its container; create it first, or Docker creates it as root
        # and the container user cannot write to it
        (Path.home() / ".ort-package-cache").mkdir(exist_ok=True)
    if job.triple == "aarch64-pc-windows-msvc" and platform.machine() == "AMD64":
        cmd += ["--target-py-version", "None"]  # build.ps1 cannot build an ARM64 wheel on an x64 host
    run(job, cmd + [QNN_TASKS[job.triple]], cwd=src)
    stage_shared_libs(job, src / "build", ["onnxruntime_providers_qnn"])
    try:
        copy_files(job, ["**/dist/*.whl"], src / "build", job.stage / "python")
    except SystemExit:
        pass


def build_ep_mlx(job: Job) -> None:
    src = ROOT / "eps/official/onnxruntime-ep-mlx/rust"
    prefix = job.work / "mlx-prefix"
    include = ort_home(job) / "include"
    mlx_build = job.work / "mlx-build"
    run(job, ["bash", "scripts/setup_mlx.sh", prefix], cwd=src, env={"MLX_BUILD_ROOT": str(mlx_build)})
    env = {"MLX_PREFIX": str(prefix), "MLXC_PREFIX": str(prefix), "ORT_INCLUDE_DIR": str(include),
           "MACOSX_DEPLOYMENT_TARGET": "14.0", "CARGO_TARGET_DIR": str(job.work / "target")}
    run(job, ["cargo", "build", "--release"], cwd=src, env=env)
    copy_files(job, ["release/libonnxruntime_mlx_ep.dylib"], job.work / "target", job.stage / "lib")
    copy_files(job, ["lib/libmlx*.dylib", "lib/mlx.metallib"], prefix, job.stage / "lib")  # MLX is MIT-licensed and may be redistributed
    stage_headers(job, prefix / "include" / "mlx", "mlx")
    # libmlx statically links fmt, nlohmann/json and gguflib and is compiled with the metal-cpp headers
    for name, sub in (("mlx", "mlx-source"), ("mlx-c", "source"), ("fmt", "fmt-source"), ("nlohmann-json", "json-source"),
                      ("gguflib", "gguflib-source"), ("metal-cpp", "metal-cpp-source/metal-cpp")):
        stage_license(job, name, mlx_build / sub, ["LICENSE*"])


def build_ep_openvino(job: Job) -> None:
    job.need("openvino", pins={"openvino": OPENVINO_PLUGIN_VERSION})  # INTEL_OPENVINO_DIR / OpenVINO_DIR: ci/sdk.py
    src, b = ROOT / "eps/official/onnxruntime-ep-openvino", job.work / "build"
    target = "onnxruntime_providers_openvino_plugin"
    need_plugin_ort(job, "eps/official/onnxruntime-ep-openvino/cmake/onnxruntime.cmake",
                    r"find_package\(onnxruntime ([\d.]+)")
    env = {"ONNXRUNTIME_ROOT_DIR": str(ort_home(job))}
    run(job, ["cmake", "-G", "Ninja", "-S", src, "-B", b, "-DCMAKE_BUILD_TYPE=Release",
              "-DOVEP_PLUGIN_BUILD_TESTS=OFF", "-DOVEP_PLUGIN_BUILD_TOOLS=OFF"], env=env)
    # only the plugin: cmake --install and the default target also stage the OpenVINO runtime (on Windows re-signed
    # copies with embedded manifests), which fetch-restricted downloads unmodified on the user's machine instead
    run(job, ["cmake", "--build", b, "--parallel", "--target", target], env=env)
    stage_shared_libs(job, b, [target])
    if job.os == "windows":
        # The plugin DLL declares the openvino_runtime private assembly (SxS), which this manifest next to it describes:
        # the OpenVINO and TBB DLLs in the same directory
        manifest = b / "sxs_manifests" / "openvino_runtime_Release.manifest"
        if job.dry_run:
            print(f"[dry-run] copy {manifest} -> bin/openvino_runtime.manifest")
        else:
            shutil.copy2(manifest, job.stage / "bin" / "openvino_runtime.manifest")
    else:
        set_origin_rpath(job, job.stage / "lib", extra="")  # the build-tree run path points into the SDK directory


def build_ep_neuropilot(job: Job) -> None:
    job.need("neuron-op-capability")
    src, b = ROOT / "eps/official/onnxruntime-ep-neuropilot", job.work / "build"
    # the prebuilt op capability library of the target (the build links against it; fetch-restricted downloads it)
    archive = "android-arm64-v8a" if job.os == "android" else "linux-aarch64"
    prebuilt = job.sdks["neuron-op-capability"]
    if not job.dry_run:
        prebuilt = next(p.parent for p in prebuilt.rglob("libneuron_op_capability.so") if p.parent.name.endswith(archive))
    # only ORT's headers are needed: ORT loads the EP library, which resolves the API at run time
    args = ["-DCMAKE_BUILD_TYPE=Release", f"-DORT_INCLUDE_DIR={ORT_SRC / 'include/onnxruntime/core/session'}",
            f"-DNEURON_OP_CAPABILITY_LIB_DIR={prebuilt}", "-DORTNeuronEp_INSTALL=OFF"]
    if job.os == "android":
        job.need("android-ndk")
        args += [f"-DCMAKE_TOOLCHAIN_FILE={src / 'cmake/toolchain-android.cmake'}",
                 f"-DANDROID_NDK_HOME={job.sdks['android-ndk']}", "-DANDROID_ABI=arm64-v8a"]
    run(job, ["cmake", "-G", "Ninja", "-S", src, "-B", b, *args])
    run(job, ["cmake", "--build", b, "--parallel"])
    lib = copy_files(job, ["**/libonnxruntime_providers_neuropilot.so"], b, job.stage / "lib")
    if job.os == "android" and not job.dry_run:  # Android builds keep their debug symbols (tens of MB)
        strip = next((Path(job.sdks["android-ndk"]) / "toolchains/llvm/prebuilt").glob("*/bin/llvm-strip"))
        run(job, [strip, "--strip-unneeded", *lib])


RECIPES: dict[str, Callable[[Job], None]] = {
    "ort": build_ort, "ort-apple": build_ort_apple,
    "ep-tensorrt": build_ep_tensorrt, "ep-nv-tensorrt-rtx": build_ep_nv_tensorrt_rtx, "ep-amdgpu": build_ep_amdgpu,
    "ep-qnn": build_ep_qnn, "ep-mlx": build_ep_mlx, "ep-openvino": build_ep_openvino, "ep-neuropilot": build_ep_neuropilot,
    "migraphx": build_migraphx,
}


def recipe_inputs(recipe: str, variant: str, triple: str, cuda_major: str = "13",
                  eps: str = "") -> tuple[set[tuple[str, str, str]], set[str]]:
    """Walk through a recipe in dry-run mode and return the required (component, platform, pinned version) tuples and
    the GPU architecture settings it compiles device code for (Job.gpu)."""
    import contextlib
    import io
    job = Job(recipe, variant, triple, Path("work"), Path("stage"), Path("sdk"), cuda_major, DEFAULT_CUDA_ARCHS, True,
              eps=[e for e in eps.split(",") if e])
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            RECIPES[recipe](job)
        except SystemExit:
            pass  # missing outputs are expected in dry-run mode
    return job.needed, job.gpu


def required_sdks(recipe: str, variant: str, triple: str, cuda_major: str = "13",
                  eps: str = "") -> set[tuple[str, str, str]]:
    """The required (component, platform, pinned version) tuples of a recipe (see recipe_inputs)."""
    return recipe_inputs(recipe, variant, triple, cuda_major, eps)[0]


def watch_memory(interval: int = 30) -> None:
    """Print memory and swap use while they are low, so a job killed for lack of memory shows it in its log."""
    def info() -> dict[str, int]:
        fields = (line.split() for line in Path("/proc/meminfo").read_text().splitlines())
        return {f[0].rstrip(":"): int(f[1]) // 1024 for f in fields}

    def loop() -> None:
        while True:
            time.sleep(interval)
            m = info()
            if m["MemAvailable"] < m["MemTotal"] // 4:
                print(f"[memory] available {m['MemAvailable']} MiB of {m['MemTotal']} MiB, "
                      f"swap free {m['SwapFree']} MiB of {m['SwapTotal']} MiB", flush=True)

    threading.Thread(target=loop, daemon=True).start()


def git_commit(path: Path) -> str:
    out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def build_cache(job: Job) -> tuple[str, str, list[Path], Path] | None:
    """(key, restore key prefix, paths, marker) of the build state a recipe can reuse from the Actions cache, or None.

    Only open-source build results are cached (the repository is public, and caches of a public repository can be
    restored by pull-request workflows). The key hashes everything the cached files depend on, so a version update
    produces a new key instead of reusing old files; the cache is saved only when the marker file exists, i.e. the
    cached step finished.
    """
    if job.recipe == "migraphx" or migraphx_staged(job.recipe, job.variant, job.triple):
        # MIGraphX's third-party dependencies (rocMLIR, ...) built by `rbuild prepare`: about 3.5 hours. They depend on
        # the files rbuild hashes for them, the ROCm (clang) version and the MSVC toolset, as in AMDMIGraphX's own CI,
        # but not on the GPU targets (rbuild prepare does not get them). The deps stage of build_migraphx saves them
        # alone; the MIGraphX install is cached with them once it is complete, keyed by the submodule commit and the GPU
        # targets. Another commit or set of targets restores the dependencies of the same hash (restore key prefix)
        # and builds only MIGraphX again. Every entry has the same paths: actions/cache restores only entries of the
        # same paths.
        digest = hashlib.sha256()
        for name in ("requirements.txt", "dev-requirements.txt", "rbuild.ini"):
            path = MIGRAPHX_SRC / name
            digest.update(name.encode() + b"\0" + (path.read_bytes() if path.exists() else b""))
        digest.update(f"{WINDOWS_ROCM_VERSION}|{os.environ.get('VCToolsVersion', '')}".encode())
        prefix = f"migraphx-{job.triple}-{digest.hexdigest()[:20]}-"
        targets = hashlib.sha256(job.gpu_targets().encode()).hexdigest()[:8]
        cget, install = job.work / "cget", job.work / "migraphx-install"
        if job.recipe == "migraphx" and job.variant == "deps":
            return f"{prefix}deps", prefix, [cget, install], cget / "hash"
        return f"{prefix}{git_commit(MIGRAPHX_SRC)[:12]}-{targets}", prefix, [cget, install], install / ".migraphx-commit"
    return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Build ORT / EPs from a recipe and stage the outputs")
    ap.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    ap.add_argument("--variant", default="default")
    ap.add_argument("--triple", required=True, choices=sorted(TRIPLES))
    ap.add_argument("--work", type=Path, required=True, help="build working directory")
    ap.add_argument("--stage", type=Path, required=True, help="staging directory (input of ci/package.py)")
    ap.add_argument("--sdk-root", type=Path, required=True, help="directory where third-party SDKs are extracted")
    ap.add_argument("--cuda-major", default="13", choices=("12", "13"))
    ap.add_argument("--eps", default="", help="comma-separated in-tree EPs of --variant custom: " + ", ".join(CUSTOM_EPS))
    ap.add_argument("--cuda-archs", default=os.environ.get("CUDA_ARCHITECTURES") or DEFAULT_CUDA_ARCHS,
                    help="CMAKE_CUDA_ARCHITECTURES (defaults to the CUDA_ARCHITECTURES environment variable)")
    ap.add_argument("--amd-gpu-targets", default=os.environ.get("MIGRAPHX_GPU_TARGETS") or DEFAULT_MIGRAPHX_GPU_TARGETS,
                    help="GPU_TARGETS of the MIGraphX build (defaults to the MIGRAPHX_GPU_TARGETS environment variable)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cache-info", type=Path, metavar="FILE",
                    help="only append key=, restore-keys=, path= and marker= of the recipe's reusable build state "
                         "(see build_cache) "
                         "to FILE, e.g. $GITHUB_OUTPUT; nothing for recipes without one")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    job = Job(args.recipe, args.variant, args.triple, args.work.resolve(), args.stage.resolve(),
              args.sdk_root.resolve(), args.cuda_major, args.cuda_archs, args.dry_run, args.amd_gpu_targets,
              eps=[e for e in args.eps.split(",") if e])
    if job.os == "macos":  # CMake and the other build tools read it (CMAKE_OSX_DEPLOYMENT_TARGET)
        os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", MACOS_DEPLOYMENT_TARGET)
    if args.cache_info:
        if cache := build_cache(job):
            key, restore, paths, marker = cache
            with args.cache_info.open("a", encoding="utf-8") as f:
                f.write(f"key={key}\nrestore-keys={restore}\nmarker={marker}\n")
                f.write("path<<EOF_PATHS\n" + "\n".join(str(p) for p in paths) + "\nEOF_PATHS\n")
        return 0
    if Path("/proc/meminfo").exists() and not args.dry_run:
        watch_memory()
    RECIPES[args.recipe](job)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
