#!/usr/bin/env python3
"""CI helper: build ONNX Runtime and its EPs from a recipe and stage the outputs (bin/ lib/ include/).

Each GitHub Actions matrix entry makes one call:

    python ci/build.py --recipe ort --variant cuda --triple x86_64-unknown-linux-gnu \\
        --work $RUNNER_TEMP/work --sdk-root $RUNNER_TEMP/sdk --stage stage

Third-party SDKs required by a recipe are downloaded automatically by ci/sdk.py (official sources, no account
needed), so the workflow does not prepare them separately. --dry-run only prints the commands, which is handy
for checking recipes locally.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import platform
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

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("ci_sdk", Path(__file__).with_name("sdk.py"))
sdk = importlib.util.module_from_spec(_spec)
sys.modules["ci_sdk"] = sdk
_spec.loader.exec_module(sdk)

# target triple -> (SDK platform, operating system)
TRIPLES = {
    "x86_64-unknown-linux-gnu": ("linux-x86_64", "linux"),
    "aarch64-unknown-linux-gnu": ("linux-aarch64", "linux"),
    "s390x-unknown-linux-gnu": ("linux-x86_64", "linux"),          # cross-compiled on an x86_64 host
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
    platform: str = ""
    os: str = ""
    sdks: dict[str, Path] = field(default_factory=dict)
    # (component, platform, pinned version or ""), used by ci/plan.py for the SDK archive matrix
    needed: set[tuple[str, str, str]] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.platform, self.os = TRIPLES[self.triple]

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


def stage_license(job: Job, name: str, base: Path, patterns: list[str]) -> None:
    """Copy the license texts of a bundled third-party component into third_party_licenses/<name>/."""
    try:
        copy_files(job, patterns, base, job.stage / "third_party_licenses" / name)
    except SystemExit:
        print(f"warning: no license text of {name} found in {base}", file=sys.stderr, flush=True)


def bundle_vendor_runtime(job: Job, eps: list[str]) -> None:
    """Bundle the runtime libraries of vendor SDKs whose licenses allow distributing them as part of an application
    (QAIRT, TensorRT for RTX with the CUDA runtime), with the license texts. scripts/fetch_ep_runtime.py selects the
    files; the archives are already in the SDK download cache. CUDA libraries beyond cudart, cuDNN and TensorRT are
    redistributable too, but would push the packages past the 2 GiB release asset limit; users fetch them with the
    script instead."""
    if job.dry_run:
        print(f"[dry-run] bundle vendor runtime for {', '.join(eps)}")
        return
    spec = importlib.util.spec_from_file_location("fetch_ep_runtime", ROOT / "scripts" / "fetch_ep_runtime.py")
    fetch = importlib.util.module_from_spec(spec)
    sys.modules["fetch_ep_runtime"] = fetch  # its dataclasses look their module up there
    spec.loader.exec_module(fetch)
    versions = json.loads((job.sdk_root / "sdk-versions.json").read_text(encoding="utf-8"))
    pins = [f"{k}={v}" for k, v in fetch.pins_from_build_info(versions).items()]
    dest = job.stage / ("bin" if job.os == "windows" else "lib")
    args = ["--ep", *eps, "--target", job.triple, "--cuda-major", job.cuda_major, "--dest", str(dest),
            "--cache", str(job.sdk_root / "_downloads"), "--licenses-dir", str(job.stage / "third_party_licenses")]
    if fetch.main(args + (["--pin", *pins] if pins else [])) != 0:
        raise SystemExit(f"bundling the {', '.join(eps)} runtime failed")


def set_origin_rpath(job: Job, lib_dir: Path, extra: str = "/opt/rocm/lib") -> None:
    """Let the libraries built here find each other in the package directory: the build-tree run paths point into the
    CI work directory. Only for libraries this project builds; vendor libraries are left unmodified."""
    if job.dry_run:
        return
    # the PyPI package ships a patchelf binary; PIP_BREAK_SYSTEM_PACKAGES allows it in the Ubuntu container images
    run(job, [python(), "-m", "pip", "install", "--quiet", "patchelf"], env={"PIP_BREAK_SYSTEM_PACKAGES": "1"})
    candidates = [shutil.which("patchelf"), Path(sysconfig.get_path("scripts")) / "patchelf",
                  Path(sysconfig.get_path("scripts", f"{os.name}_user")) / "patchelf"]
    patchelf = next(str(c) for c in candidates if c and Path(c).exists())
    for f in sorted(lib_dir.glob("*.so*")):
        if f.is_file() and not f.is_symlink() and f.read_bytes()[:4] == b"\x7fELF":
            run(job, [patchelf, "--set-rpath", f"$ORIGIN:{extra}", f])


def bundle_openvino(job: Job) -> None:
    """The OpenVINO runtime with its device plugins, next to the OpenVINO provider. The prebuilt runtime is under the
    Intel OpenVINO Distribution License, which allows redistributing the files its docs/licensing/redist.txt lists
    (Windows: runtime/bin/intel64/Release and oneTBB; Linux: runtime/lib/intel64, oneTBB comes from the system) together
    with that license."""
    ov = job.sdks["openvino"]
    if job.os == "windows":
        tbb = [f"runtime/3rdparty/tbb/bin/{n}.dll" for n in ("tbb12", "tbbmalloc", "tbbmalloc_proxy", "tbbbind_2_5")]
        copy_files(job, ["runtime/bin/intel64/Release/*", *tbb], ov, job.stage / "bin")
    else:
        copy_files(job, ["runtime/lib/intel64/*"], ov, job.stage / "lib")
    stage_license(job, "OpenVINO", ov, [f"docs/licensing/{n}" for n in (
        "EULA.txt", "redist.txt", "readme.txt", "LICENSE", "Apache_license.txt", "runtime-third-party-programs.txt",
        "onetbb_third-party-programs.txt", "onednn_third-party-programs.txt")] + ["runtime/3rdparty/tbb/TBB-LICENSE"])


def bundle_directml(job: Job, build_dir: Path) -> None:
    """DirectML.dll from the Microsoft.AI.DirectML NuGet package the build restored (redistributable with applications)."""
    arch = {"x86_64": "x64", "aarch64": "arm64"}.get(job.triple.split("-")[0], "x64")
    copy_files(job, [f"**/Microsoft.AI.DirectML*/bin/{arch}-win/DirectML.dll"], build_dir, job.stage / "bin")
    stage_license(job, "DirectML", build_dir, ["**/Microsoft.AI.DirectML*/LICENSE.txt",
                                               "**/Microsoft.AI.DirectML*/ThirdPartyNotices.txt"])


def stage_ort_headers(job: Job, src: Path) -> None:
    """Copy the public ORT C/C++ API headers (for packages that are not produced by cmake --install)."""
    copy_files(job, ["include/onnxruntime/core/session/onnxruntime_*.h",
                     "include/onnxruntime/core/session/onnxruntime_*.inc"], src, job.stage / "include" / "onnxruntime")


def python() -> str:
    return sys.executable


def vs_generator() -> str:
    """CMake Visual Studio generator of the MSVC environment set up by msvc-dev-cmd (hosted runners ship VS 2026)."""
    major = os.environ.get("VisualStudioVersion", "18.0").split(".")[0]
    return {"17": "Visual Studio 17 2022"}.get(major, "Visual Studio 18 2026")


# ---------------------------------------------------------------------------
# ONNX Runtime main repository and forks (tools/ci_build/build.py)
# ---------------------------------------------------------------------------

ORT_SOURCES = {
    "ort": ROOT / "onnxruntime",
    "ort-intel": ROOT / "eps/vendor/intel-onnxruntime-openvino",
}


def cuda_flags(job: Job) -> list[str]:
    job.need("cuda", "cudnn")
    flags = ["--use_cuda", "--cuda_home", job.sdks["cuda"], "--cudnn_home", job.sdks["cudnn"], "--nvcc_threads", "1",
             "--cmake_extra_defines", f"CMAKE_CUDA_ARCHITECTURES={job.cuda_archs}"]
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


def macos_flags(job: Job) -> list[str]:
    return ["--use_coreml"] + (["--osx_arch", "x86_64"] if job.triple == "x86_64-apple-darwin" else [])


def dawn_backends(job: Job) -> list[str]:
    return {"windows": ["onnxruntime_ENABLE_DAWN_BACKEND_D3D12=1", "onnxruntime_ENABLE_DAWN_BACKEND_VULKAN=1"],
            "linux": ["onnxruntime_ENABLE_DAWN_BACKEND_VULKAN=1"]}.get(job.os, [])


def all_builtin_flags(job: Job) -> list[str]:
    """Every built-in EP the target supports.

    Built-in means linked into the onnxruntime library itself (XNNPACK, WebGPU as static_lib, DirectML, CoreML) or
    a shared provider that exports only the legacy bridge entry point GetProvider (oneDNN, TensorRT, CANN).
    Providers that also export CreateEpFactories (CUDA, OpenVINO, QNN, Vitis AI, NV TensorRT RTX, MIGraphX) and
    plugin EPs are left out. The one exception is the CUDA EP shipped as a dependency of TensorRT.
    """
    def has(*names: str) -> bool:
        # the SDKs are only available for the build host platform; x86_64-apple-darwin is cross-compiled on arm64
        return job.triple != "x86_64-apple-darwin" and all(job.platform in sdk.dl.component(n).platforms for n in names)

    flags = ["--use_xnnpack", "--use_webgpu"]  # static_lib: WebGPU as an internal EP
    if backends := dawn_backends(job):
        flags += ["--cmake_extra_defines", *backends]
    if job.os == "windows":
        flags += ["--use_dml"]
    if job.os == "macos":
        flags += macos_flags(job)
    if job.triple.startswith("x86_64") and job.os in ("linux", "windows"):
        flags += ["--use_dnnl"]
    if job.os in ("linux", "windows") and has("cuda", "cudnn", "tensorrt"):
        flags += tensorrt_flags(job)
    if job.os == "linux" and has("cann"):
        job.need("cann")
        flags += ["--use_cann", "--cann_home", job.sdks["cann"]]
        if "--use_cuda" in flags and not job.dry_run:
            allow_cann_with_cuda(ROOT / "onnxruntime")
    return flags


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
    """Return (extra build.py arguments, custom staging function). A None staging function means cmake --install."""
    v = job.variant
    if v in ("default", "arm64x"):
        flags = ["--use_xnnpack"] if v == "default" else []
        if job.triple.startswith("x86_64") and job.os in ("linux", "windows"):
            flags += ["--use_dnnl", "--use_vitisai"]
        if job.os == "macos":
            flags += macos_flags(job)
        return flags, None
    if v == "all":
        return all_builtin_flags(job), None
    if v == "qnn":
        job.need("qairt")
        return ["--use_qnn", "--qnn_home", job.sdks["qairt"]], None
    if v == "cuda":
        return tensorrt_flags(job), None
    if v == "cuda-plugin":
        flags = cuda_flags(job) + ["--cmake_extra_defines", "onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON"]
        if job.os == "windows":
            # The plugin target passes /bigobj only to the CUDA host compiler, but cudnn_flash_attention.cc needs it
            # too (C1128); ORT's build.py sets CMAKE_CXX_FLAGS itself, so add it to the Release flags (MSVC defaults)
            flags += ["--cmake_extra_defines", "CMAKE_CXX_FLAGS_RELEASE=/O2 /Ob2 /DNDEBUG /bigobj"]
        return flags, \
            lambda j, b: stage_shared_libs(j, b, ["onnxruntime_providers_cuda"])
    if v == "nv-tensorrt-rtx":
        job.need("cuda", "tensorrt-rtx")
        # onnxruntime_providers_nv.cmake runs find_package(CUDAToolkit), which does not see --cuda_home
        return ["--use_nv_tensorrt_rtx", "--tensorrt_rtx_home", job.sdks["tensorrt-rtx"], "--cuda_home", job.sdks["cuda"],
                "--cmake_extra_defines", f"CUDAToolkit_ROOT={job.sdks['cuda']}"], None
    if v == "webgpu-plugin":
        names = ["onnxruntime_providers_webgpu"] + (["dxcompiler"] if job.os == "windows" else [])  # same as the official package
        backends = dawn_backends(job)
        return ["--use_webgpu", "shared_lib", "--wgsl_template", "static",
                *(["--cmake_extra_defines", *backends] if backends else [])], \
            lambda j, b: stage_shared_libs(j, b, names)
    if v == "openvino":  # OpenVINO_DIR / INTEL_OPENVINO_DIR are set by ci/sdk.py
        job.need("openvino")
        return ["--use_openvino", "CPU"], None
    if v == "migraphx":  # built inside the rocm/dev-ubuntu container; ROCm and MIGraphX live in /opt/rocm
        return ["--use_migraphx", "--migraphx_home", os.environ.get("ROCM_PATH", "/opt/rocm")], None
    if v == "vsinpu":
        tim_vx = build_tim_vx(job)
        if not job.dry_run:
            os.environ["TIM_VX_INSTALL"] = str(tim_vx)
        return ["--use_vsinpu"], \
            lambda j, b: (install_ort(j, b), copy_files(j, ["lib/libtim-vx.so*"], tim_vx, j.stage / "lib"),
                          stage_license(j, "TIM-VX", j.sdks["tim-vx"], ["LICENSE"]))
    if v == "android":  # same EP set as the official Android AAR
        job.need("android-ndk")
        sdk_path = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT") or ""
        return ["--android", "--android_sdk_path", sdk_path, "--android_ndk_path", job.sdks["android-ndk"],
                "--android_abi", ANDROID_ABI[job.triple], "--use_nnapi", "--use_xnnpack", "--use_webgpu"], None
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
                 "wasm-webgpu-jspi": ["--use_webgpu", "--use_webnn", "--enable_wasm_jspi"],
                 "wasm-webgpu-static-plugin": ["--use_webgpu", "static_plugin", "--use_webnn", *catch]}[v]
        return ["--build_wasm", "--enable_wasm_simd", "--enable_wasm_threads", "--target", "onnxruntime_webassembly",
                "--disable_rtti", *extra], \
            lambda j, b: copy_files(j, ["ort-wasm-*.wasm", "ort-wasm-*.mjs"], b, j.stage / "lib")
    raise SystemExit(f"unknown ORT variant: {v}")


def build_ort(job: Job) -> None:
    src = ORT_SOURCES[job.recipe]
    flags, stager = ort_variant(job, src)
    build_dir = job.work / "build"
    base = [python(), src / "tools/ci_build/build.py", "--config", "Release", "--update", "--build", "--parallel",
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
        # Dawn loads the DXC shader compiler it builds (dxcompiler.dll) at run time, as in the WebGPU plugin package
        copy_files(job, ["dxcompiler.dll"], config_dir, job.stage / "bin")
    if job.os == "windows" and ("--use_webgpu" in flags or job.variant == "webgpu-plugin"):
        stage_license(job, "DirectXShaderCompiler", build_dir, ["**/third_party/dxc/LICENSE.TXT"])
    if "--use_dml" in flags:
        bundle_directml(job, build_dir)
    if "--use_openvino" in flags:
        bundle_openvino(job)
    if "--use_qnn" in flags:
        bundle_vendor_runtime(job, ["qnn"])
    if "--use_nv_tensorrt_rtx" in flags:
        bundle_vendor_runtime(job, ["nv-tensorrt-rtx"])


def install_ort(job: Job, config_dir: Path) -> None:
    run(job, ["cmake", "--install", config_dir, "--prefix", job.stage, "--config", "Release"])
    if job.os == "windows" and not job.dry_run:
        # The providers are MODULE libraries, which install() puts into lib/; they are loaded from next to onnxruntime.dll
        for dll in (job.stage / "lib").glob("*.dll"):
            shutil.move(dll, job.stage / "bin" / dll.name)


def build_ort_apple(job: Job) -> None:
    """Apple mobile platforms: one sysroot/arch dynamic framework per job, built the same way as the official pods."""
    src = ROOT / "onnxruntime"
    sysroot, arch, settings = APPLE_FRAMEWORKS[job.triple]
    settings = src / "tools/ci_build/github/apple" / settings
    build_dir = job.work / "build"
    if sysroot == "macabi" and not job.dry_run:
        # ORT's build.py turns warnings into errors for Catalyst (unlike iOS / tvOS / visionOS), and the CoreML EP uses
        # APIs newer than the 14.0 deployment target without availability guards
        data = json.loads(settings.read_text(encoding="utf-8"))
        data["build_params"]["macabi"].append("--compile_no_warning_as_error")
        settings = job.work / settings.name
        settings.parent.mkdir(parents=True, exist_ok=True)
        settings.write_text(json.dumps(data, indent=2), encoding="utf-8")
    run(job, [python(), src / "tools/ci_build/github/apple/build_apple_framework.py",
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


def migraphx_gpu_targets() -> str:
    return os.environ.get("MIGRAPHX_GPU_TARGETS") or DEFAULT_MIGRAPHX_GPU_TARGETS


def build_migraphx_linux(job: Job) -> tuple[Path, Path]:
    """Build MIGraphX (develop, which the AMD EPs follow) with its GPU backend, the same way AMDMIGraphX's ci.yaml does:
    inside its rocm/migraphx-ci-ubuntu image, which has ROCm in /opt/rocm and the dependencies in /usr/local.
    Returns (MIGraphX install prefix, ROCm root)."""
    job.need("amdmigraphx")
    src, b, install = job.sdks["amdmigraphx"], job.work / "migraphx-build", job.work / "migraphx-install"
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    run(job, ["cmake", "-S", src, "-B", b, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
              f"-DCMAKE_C_COMPILER={rocm / 'llvm/bin/clang'}", f"-DCMAKE_CXX_COMPILER={rocm / 'llvm/bin/clang++'}",
              "-DMIGRAPHX_ENABLE_GPU=On", "-DMIGRAPHX_ENABLE_PYTHON=Off", "-DBUILD_TESTING=Off",
              f"-DGPU_TARGETS={migraphx_gpu_targets()}", f"-DCMAKE_INSTALL_PREFIX={install}"])
    run(job, ["cmake", "--build", b, "--config", "Release"])
    run(job, ["cmake", "--install", b])
    return install, rocm


def install_rocm_wheels(job: Job) -> tuple[Path, Path]:
    """Install ROCm from TheRock's Python wheels, as AMDMIGraphX's windows.yaml does: unlike the multi-arch SDK tarball,
    the device-<gfx> packages also bring the device libraries HIP code needs. Returns (ROCm root, CMake prefix)."""
    targets = migraphx_gpu_targets().split(";")
    extras = ",".join(["libraries", "devel", *(f"device-{t}" for t in targets)])
    run(job, [python(), "-m", "pip", "install", "--index-url", "https://repo.amd.com/rocm/whl-multi-arch/",
              f"rocm[{extras}]=={WINDOWS_ROCM_VERSION}"])
    run(job, ["rocm-sdk", "init"])
    if job.dry_run:
        return job.sdk_root / "rocm", job.sdk_root / "rocm" / "lib" / "cmake"
    path = lambda kind: Path(subprocess.run(["rocm-sdk", "path", kind], capture_output=True, text=True,
                                            check=True).stdout.strip())
    return path("--root"), path("--cmake")


def build_migraphx_windows(job: Job) -> tuple[Path, Path]:
    """Build MIGraphX with its GPU backend on Windows, following AMDMIGraphX's .github/workflows/windows.yaml.

    ROCm comes from TheRock's Python wheels; MIGraphX's own dependencies (including rocMLIR) are built by rbuild.
    Returns (MIGraphX install prefix, ROCm root).
    """
    job.need("amdmigraphx")
    src = job.sdks["amdmigraphx"]
    rocm, rocm_cmake = install_rocm_wheels(job)
    install, cget = job.work / "migraphx-install", job.work / "cget"
    path = os.pathsep.join([str(rocm / "lib" / "llvm" / "bin"), str(rocm / "bin"), os.environ.get("PATH", "")])
    # rocMLIR has long paths; build the dependencies in a short directory like the upstream CI does
    short = Path(job.work.anchor) / "b"
    run(job, [python(), "-m", "pip", "install", "https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz"])
    run(job, ["rbuild", "prepare", "-d", cget, "-s", "gh-win-gpu", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
              "-DLLVM_DISABLE_ASSEMBLY_FILES=On", "-DLLVM_ENABLE_DIA_SDK=Off"], cwd=src,
        env={"PATH": path, "CC": "clang-cl", "CXX": "clang-cl", "CXXFLAGS": "/std:c++17", "CGET_BUILD_PATH": str(short)})
    run(job, ["rbuild", "build", "-d", cget, "-s", "gh-win-gpu", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
              f"-DGPU_TARGETS={migraphx_gpu_targets()}", f"-DCMAKE_INSTALL_PREFIX={install}"], cwd=src,
        env={"PATH": path, "CC": "clang", "CXX": "clang++", "CMAKE_PREFIX_PATH": f"{rocm_cmake};{cget}"})
    run(job, ["cmake", "--install", src / "build", "--prefix", install])
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
    return home


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
              f"-DCMAKE_CUDA_ARCHITECTURES={job.cuda_archs}", f"-DCMAKE_CUDA_COMPILER={nvcc}",
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
    args = ["--cuda_home", job.sdks["cuda"], "--onnxruntime_home", ort_home(job),
            "--trt_rtx_home", job.sdks["tensorrt-rtx"], "--build_dir", b, "--config", "Release"]
    if job.os == "windows":
        # build.bat hard-codes the VS 2022 generator; run a copy (next to it, since it uses %~dp0) with the
        # generator of the installed Visual Studio
        script = src / "build.ci.bat"
        if not job.dry_run:
            text = (src / "build.bat").read_text(encoding="utf-8")
            script.write_text(text.replace('"Visual Studio 17 2022"', f'"{vs_generator()}"'), encoding="utf-8")
        # Windows ARM64 is cross-compiled on an x64 host, as documented in docs/build-guide.md
        # --skip_tests: the unit test target runs its executable after linking (gtest_discover_tests), which fails
        # on a runner without an NVIDIA GPU
        arch = ["--arm64"] if job.triple.startswith("aarch64") else []
        run(job, ["cmd", "/c", script, *args, "--skip_tests", *arch], cwd=src)
    else:
        # build.sh compiles with a bare `cmake --build --parallel`, i.e. `make -j` without a limit, which runs the
        # runner out of memory; configure with it and build here, where CMAKE_BUILD_PARALLEL_LEVEL applies
        run(job, ["bash", src / "build.sh", *args, "--update"], cwd=src)
        run(job, ["cmake", "-DBUILD_TESTS=OFF", b])  # build.sh has no --skip_tests; see the Windows branch
        run(job, ["cmake", "--build", b, "--config", "Release"])
    stage_shared_libs(job, b, ["onnxruntime_providers_nv_tensorrt_rtx"])
    bundle_vendor_runtime(job, ["nv-tensorrt-rtx"])


def build_ep_amdgpu(job: Job) -> None:
    src, b = ROOT / "eps/official/onnxruntime-ep-amdgpu", job.work / "build"
    cmd = [python(), src / "tools/ci_build/build.py", "--build_dir", b, "--config", "Release", "--update", "--build",
           "--parallel", "--cmake_generator", "Ninja", "--onnxrt_home", ort_home(job)]
    if job.variant == "dml":
        cmd += ["--use_dml"]
        names = ["directx-ep"]
    else:
        migraphx, rocm = build_migraphx_windows(job) if job.os == "windows" else build_migraphx_linux(job)
        cmd += ["--migraphx_home", migraphx, "--hip_path", rocm]
        if job.variant == "amdgpu":  # AMDGPU EP; implies MIGraphX (and DirectML on Windows), renamed to *-backend
            cmd += ["--use_amdgpu"]
            names = ["amdgpu-ep", "migraphx-backend", "hip-backend", "directx-backend"]
        else:
            cmd += ["--use_migraphx"]
            names = ["migraphx-ep"]
    if job.os == "windows":
        # With VSINSTALLDIR set, the script's _build_environment() replaces the whole environment with an empty one
        # (no PATH, no compiler); without it, the script loads vcvars64.bat of the latest Visual Studio itself.
        os.environ.pop("VSINSTALLDIR", None)
    run(job, cmd, cwd=src)
    stage_shared_libs(job, b / "Release", names)
    if job.os == "windows" and job.variant in ("dml", "amdgpu"):  # the DirectML EP / backend
        bundle_directml(job, b)
    if job.variant != "dml":  # MIGraphX (MIT) is built from its develop branch here, so bundle it with its license
        if job.os == "windows":
            copy_files(job, ["bin/migraphx*.dll"], migraphx, job.stage / "bin")
        else:
            copy_files(job, ["lib/libmigraphx*.so*", "lib/migraphx/lib/*.so*"], migraphx, job.stage / "lib")
            set_origin_rpath(job, job.stage / "lib")
        stage_license(job, "MIGraphX", job.sdks["amdmigraphx"], ["LICENSE*"])


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
    cmd = [python(), "qcom/build_and_test.py", "--config", "Release", "--qairt-sdk", job.sdks["qairt"]]
    if job.os != "android":  # Android needs ORT as an AAR, so the repository script builds it from source
        # Build against the ORT release the repository's own CI uses: its code follows that release's experimental
        # API, which later releases changed. (Not known in the plan job, which has no submodules checked out.)
        packages = src / "qcom/packages.yml"
        if packages.exists():
            ort = sdk.dl.yaml_block(packages.read_text(encoding="utf-8"), "ort_prebuilt_windows_x64")["version"]
            job.need("onnxruntime-sdk", pins={"onnxruntime-sdk": ort})
        else:
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
    bundle_vendor_runtime(job, ["qnn"])


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
    # libmlx statically links fmt, nlohmann/json and gguflib and is compiled with the metal-cpp headers
    for name, sub in (("mlx", "mlx-source"), ("mlx-c", "source"), ("fmt", "fmt-source"), ("nlohmann-json", "json-source"),
                      ("gguflib", "gguflib-source"), ("metal-cpp", "metal-cpp-source/metal-cpp")):
        stage_license(job, name, mlx_build / sub, ["LICENSE*"])


def build_ep_iree(job: Job) -> None:
    src = ROOT / "eps/community/onnxruntime-ep-iree"
    if job.os == "windows":
        # The build tree lives in <source>/build/cmake/default and some IREE object paths exceed MAX_PATH (cl.exe
        # fails with C1083), so build from a copy in a short directory
        short = Path(job.work.anchor) / "iree"
        if not job.dry_run:
            top_level = lambda d, names: [n for n in names if n in (".git", "build")] if Path(d) == src else []
            shutil.copytree(src, short, ignore=top_level, dirs_exist_ok=True)
        src = short
    run(job, [python(), "-m", "pip", "install", "-r", "requirements-dev.txt"], cwd=src)
    run(job, [python(), "-m", "pip", "wheel", "./python", "--no-deps", "-w", job.stage / "python"], cwd=src)
    install = src / "build" / "cmake" / "default" / "python-install"
    stage_shared_libs(job, install, ["onnxruntime_ep_iree"])


def build_ep_telum(job: Job) -> None:
    src, b = ROOT / "eps/community/onnxruntime-ep-telum", job.work / "build"
    toolchain = job.work / "s390x-toolchain.cmake"
    if not job.dry_run:
        job.work.mkdir(parents=True, exist_ok=True)
        toolchain.write_text("set(CMAKE_SYSTEM_NAME Linux)\nset(CMAKE_SYSTEM_PROCESSOR s390x)\n"
                             "set(CMAKE_C_COMPILER s390x-linux-gnu-gcc)\nset(CMAKE_CXX_COMPILER s390x-linux-gnu-g++)\n"
                             "set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)\n", encoding="utf-8")
    headers = ROOT / "onnxruntime/include/onnxruntime/core/session"  # only the ORT headers are needed
    run(job, ["cmake", "-G", "Ninja", "-S", src, "-B", b, "-DCMAKE_BUILD_TYPE=Release",
              f"-DCMAKE_TOOLCHAIN_FILE={toolchain}", f"-DONNXRUNTIME_INCLUDE_DIR={headers}", "-DTELUM_EP_ENABLE_ZDNN=ON"])
    run(job, ["cmake", "--build", b, "--parallel"])
    copy_files(job, ["libtelum_plugin_ep.so"], b, job.stage / "lib")


def build_ep_vulkan(job: Job) -> None:
    job.need("vulkan-sdk")
    src = ROOT / "eps/community/onnxruntime-ep-vulkan/rust"
    env = {"VULKAN_SDK": str(job.sdks["vulkan-sdk"]), "CARGO_TARGET_DIR": str(job.work / "target")}
    run(job, ["cargo", "build", "--release"], cwd=src, env=env)
    stage_shared_libs(job, job.work / "target" / "release", ["onnxruntime_vulkan_ep"])


RECIPES: dict[str, Callable[[Job], None]] = {
    "ort": build_ort, "ort-intel": build_ort, "ort-apple": build_ort_apple,
    "ep-tensorrt": build_ep_tensorrt, "ep-nv-tensorrt-rtx": build_ep_nv_tensorrt_rtx, "ep-amdgpu": build_ep_amdgpu,
    "ep-qnn": build_ep_qnn, "ep-mlx": build_ep_mlx, "ep-iree": build_ep_iree, "ep-telum": build_ep_telum,
    "ep-vulkan": build_ep_vulkan,
}


def required_sdks(recipe: str, variant: str, triple: str, cuda_major: str = "13") -> set[tuple[str, str, str]]:
    """Walk through a recipe in dry-run mode and return the required (component, platform, pinned version) tuples."""
    import contextlib
    import io
    job = Job(recipe, variant, triple, Path("work"), Path("stage"), Path("sdk"), cuda_major, DEFAULT_CUDA_ARCHS, True)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            RECIPES[recipe](job)
        except SystemExit:
            pass  # missing outputs are expected in dry-run mode
    return job.needed


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


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Build ORT / EPs from a recipe and stage the outputs")
    ap.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    ap.add_argument("--variant", default="default")
    ap.add_argument("--triple", required=True, choices=sorted(TRIPLES))
    ap.add_argument("--work", type=Path, required=True, help="build working directory")
    ap.add_argument("--stage", type=Path, required=True, help="staging directory (input of ci/package.py)")
    ap.add_argument("--sdk-root", type=Path, required=True, help="directory where third-party SDKs are extracted")
    ap.add_argument("--cuda-major", default="13", choices=("12", "13"))
    ap.add_argument("--cuda-archs", default=os.environ.get("CUDA_ARCHITECTURES", DEFAULT_CUDA_ARCHS),
                    help="CMAKE_CUDA_ARCHITECTURES (defaults to the CUDA_ARCHITECTURES environment variable)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    job = Job(args.recipe, args.variant, args.triple, args.work.resolve(), args.stage.resolve(),
              args.sdk_root.resolve(), args.cuda_major, args.cuda_archs, args.dry_run)
    if Path("/proc/meminfo").exists() and not args.dry_run:
        watch_memory()
    RECIPES[args.recipe](job)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
