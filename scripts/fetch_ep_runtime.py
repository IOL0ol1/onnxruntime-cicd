#!/usr/bin/env python3
"""Download the closed-source SDKs that some providers / EPs need, and extract only the files an application needs.

Three callers use it: ci/build.py bundles the small vendor runtimes whose licenses allow distributing them with an
application (QAIRT, TensorRT for RTX with the CUDA runtime) into the packages; ci/bundle.py adds the CUDA libraries,
cuDNN and TensorRT (too large for the packages) to a bundle; and the fetch-restricted script of a bundle fetches what
may not be redistributed at all (headers and import libraries, the CANN toolkit, the Vitis AI runtime) on the user's
machine. The files come from the same no-account official sources as scripts/download_ep_packages.py.

On a bundle, the EPs, target triple, CUDA major version and exact SDK versions are read from its BUNDLE_INFO.json:

    python tools/fetch_ep_runtime.py --package-dir <extracted bundle>

Or select EPs and a target explicitly:

    python scripts/fetch_ep_runtime.py --ep cuda tensorrt --target x86_64-unknown-linux-gnu --dest ./runtime
    python scripts/fetch_ep_runtime.py --ep qnn --target aarch64-pc-windows-msvc --dry-run
    python scripts/fetch_ep_runtime.py --list

An EP can be suffixed with the parts to fetch: `cuda:runtime` (the default), `cuda:dev` (the backend's headers into
include/<backend>/ and its import libraries into lib/, which the vendors' licenses do not let us redistribute) or
`cuda:runtime+dev`.

Only the Python standard library is required. Python 3.14+ reads .tar.zst archives itself; older versions use zstd, a
libarchive tar (Windows, macOS) or 7-Zip when one is installed, and otherwise download 7-Zip to a temporary directory.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import functools
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

# explicit, because a bundle runs this script with python -I, which leaves the script's directory out of sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_ep_packages as dl  # noqa: E402

# target triple -> (download platform, runtime OS, QAIRT library directory)
TARGETS = {
    "x86_64-unknown-linux-gnu": ("linux-x86_64", "linux", "x86_64-linux-clang"),
    "aarch64-unknown-linux-gnu": ("linux-aarch64", "linux", "aarch64-oe-linux-gcc11.2"),
    "x86_64-pc-windows-msvc": ("windows-x86_64", "windows", "x86_64-windows-msvc"),
    "aarch64-pc-windows-msvc": ("windows-arm64", "windows", "aarch64-windows-msvc"),
    "arm64ec-pc-windows-msvc": ("windows-arm64x", "windows", "arm64x-windows-msvc"),
    "arm64x-pc-windows-msvc": ("windows-arm64x", "windows", "arm64x-windows-msvc"),
    "aarch64-linux-android": ("linux-x86_64", "android", "aarch64-android"),
    "aarch64-apple-darwin": ("macos-arm64", "macos", ""),
}

# Runtime library name patterns (matched against the file name inside the archive)
CUDA_LIBS = {
    "cudart": (r"libcudart\.so[.\d]*", r"cudart64_\d+\.dll"),
    "cublas": (r"libcublas(Lt)?\.so[.\d]*", r"cublas(Lt)?64_\d+\.dll"),
    "cufft": (r"libcufft\.so[.\d]*", r"cufft64_\d+\.dll"),
    "curand": (r"libcurand\.so[.\d]*", r"curand64_\d+\.dll"),
    "nvrtc": (r"libnvrtc(-builtins)?\.so[.\d]*", r"nvrtc(-builtins)?64_\d+(_0)?\.dll"),
    "nvjitlink": (r"libnvJitLink\.so[.\d]*", r"nvJitLink_\d+_0\.dll"),
}
CUDNN_LIBS = (r"libcudnn[\w]*\.so[.\d]*", r"cudnn[\w]*64_\d+\.dll")
TENSORRT_LIBS = (r"lib(nvinfer[\w]*|nvonnxparser)\.so[.\d]*", r"(nvinfer[\w]*|nvonnxparser)_\d+\.dll")
TENSORRT_RTX_LIBS = (r"libtensorrt(_onnxparser)?_rtx[\w]*\.so[.\d]*", r"tensorrt(_onnxparser)?_rtx_[\d_]+\.dll")
QAIRT_LIBS = r"(libQnn\w*\.so|Qnn\w*\.dll|libGenie\w*\.so|Genie\w*\.dll|libHtpPrepare\.so|HtpPrepare\.dll)"
# Hexagon DSP libraries loaded by the HTP backend (same lists as onnxruntime-qnn's cmake/CMakeLists.txt)
QAIRT_SKELS = {
    "aarch64-windows-msvc": ("68", "73", "81"),
    "arm64x-windows-msvc": ("68", "73", "81"),
    "aarch64-oe-linux-gcc11.2": ("68", "69", "73", "75", "79", "81"),
    "aarch64-android": ("68", "69", "73", "75", "79", "81"),
}
LICENSE_FILE = re.compile(r"(?i)^[^/]+(/[^/]+)?/(license|eula|notice|acknowledg)[^/]*$|\.dist-info/licenses/[^/]+$")
SKIP_DIRS = re.compile(r"(?i)/(samples?|python|doc|docs|include|targets/[^/]+/lib/stubs|stubs)/")
SKIP_DEV_DIRS = re.compile(r"(?i)/(samples?|python|doc|docs)/")
PARTS = ("runtime", "dev")

# 7-Zip (24.01+ unpacks zstd) for hosts without another zstd decompressor: host platform -> files to download
SEVEN_ZIP_URL = "https://github.com/ip7z/7zip/releases/download/25.01/"
SEVEN_ZIP = {
    "windows-x86_64": [("7zr.exe", "27cbe3d5804ad09e90bbcaa916da0d5c3b0be9462d0e0fb6cb54be5ed9030875"),
                       ("7z2501-x64.exe", "78afa2a1c773caf3cf7edf62f857d2a8a5da55fb0fff5da416074c0d28b2b55f")],
    "windows-arm64": [("7zr.exe", "27cbe3d5804ad09e90bbcaa916da0d5c3b0be9462d0e0fb6cb54be5ed9030875"),
                      ("7z2501-arm64.exe", "6365c7c44e217b9c1009e065daf9f9aa37454e64315b4aaa263f7f8f060755dc")],
    "linux-x86_64": [("7z2501-linux-x64.tar.xz", "4ca3b7c6f2f67866b92622818b58233dc70367be2f36b498eb0bdeaaa44b53f4")],
    "linux-aarch64": [("7z2501-linux-arm64.tar.xz", "39c5140f02ce4436599303c59a149f654cb1bbc47cdc105a942120d747ae040d")],
    "macos-arm64": [("7z2501-mac.tar.xz", "26aa75bc262bb10bf0805617b95569c3035c2c590a99f7db55c7e9607b2685e0")],
}


@dataclass
class DevRule:
    """A development file rule: archive paths matching `pattern` go to <root>/<dest>/<the pattern's "rel" group>."""
    pattern: str
    dest: str


@dataclass
class Selection:
    """Which runtime files to take from one SDK component."""
    component: str
    patterns: list[str] = field(default_factory=list)       # regexes on the path inside the archive
    archives: list[str] = field(default_factory=list)       # only download archives whose name starts with one of these
    install: bool = False                                    # run the vendor installer instead of extracting
    dev: list[DevRule] = field(default_factory=list)        # headers / import libraries, keeping their directory tree
    licenses: list[str] = field(default_factory=list)       # license files besides the ones LICENSE_FILE finds
    tree: list[DevRule] = field(default_factory=list)       # runtime files that keep their directory tree
    beside: list[DevRule] = field(default_factory=list)     # the same, relative to the runtime files rather than <root>


def _libs(os_name: str, *pairs: tuple[str, str]) -> list[str]:
    idx = 1 if os_name == "windows" else 0
    return [rf"(^|/){p[idx]}$" for p in pairs]


def _dml_arch(platform: str) -> str:
    return {"windows-arm64": "arm64", "windows-arm64x": "arm64ec"}.get(platform, "x64")


DML_LICENSES = [r"^(LICENSE|LICENSE-CODE|ThirdPartyNotices)\.txt$"]  # at the root of the NuGet package
OPENVINO_LICENSES = [r"^[^/]+/docs/licensing/[^/]+$", r"/runtime/3rdparty/tbb/TBB-LICENSE$"]

# CUDA library -> redist archive that contains it, and the import libraries of that archive
CUDA_ARCHIVES = {"cudart": "cuda_cudart", "cublas": "libcublas", "cufft": "libcufft", "curand": "libcurand",
                 "nvrtc": "cuda_nvrtc", "nvjitlink": "libnvjitlink"}
CUDA_IMPORT_LIBS = {"cudart": "cudart|cuda", "cublas": "cublas|cublasLt", "cufft": "cufft", "curand": "curand",
                    "nvrtc": "nvrtc", "nvjitlink": "nvJitLink"}
# The CUDA libraries the runtime of an EP needs (selections_for); its development files come from the same archives
CUDA_RUNTIME = {"cuda": ("cudart", "cublas", "cufft", "curand", "nvrtc", "nvjitlink"),
                "tensorrt": ("cudart", "cublas", "nvjitlink"), "nv-tensorrt-rtx": ("cudart",)}


def selections_for(ep: str, platform: str, os_name: str, qnn_abi: str,
                   amd_targets: tuple[str, ...] = ()) -> list[Selection]:
    cuda = lambda *names: Selection("cuda", _libs(os_name, *(CUDA_LIBS[n] for n in names)),
                                    [CUDA_ARCHIVES[n] for n in names])
    if ep == "cuda":
        return [cuda(*CUDA_RUNTIME[ep]), Selection("cudnn", _libs(os_name, CUDNN_LIBS))]
    if ep == "tensorrt":
        return [cuda(*CUDA_RUNTIME[ep]), Selection("tensorrt", _libs(os_name, TENSORRT_LIBS))]
    if ep == "nv-tensorrt-rtx":
        return [cuda(*CUDA_RUNTIME[ep]), Selection("tensorrt-rtx", _libs(os_name, TENSORRT_RTX_LIBS))]
    if ep == "qnn":
        patterns = [rf"/lib/{re.escape(qnn_abi)}/{QAIRT_LIBS}$"]
        skels = QAIRT_SKELS.get(qnn_abi, ())
        if skels:
            v = "|".join(skels)
            patterns.append(rf"/lib/hexagon-v({v})/unsigned/(libQnnHtpV({v})Skel\.so|libqnnhtpv({v})\.cat)$")
        return [Selection("qairt", patterns)]
    if ep == "vitisai":
        # onnxruntime_providers_vitisai.dll is part of our package; the closed backend lives in the voe wheel
        return [Selection("vitisai", [r"onnxruntime/capi/[^/]+\.dll$"], ["voe-"])]
    if ep == "cann":
        return [Selection("cann", install=True)]
    if ep == "openvino":  # the files its docs/licensing/redist.txt lists; on Linux oneTBB comes from the system
        if os_name == "windows":
            patterns = [r"/runtime/bin/intel64/Release/[^/]+$",
                        r"/runtime/3rdparty/tbb/bin/(tbb12|tbbmalloc|tbbmalloc_proxy|tbbbind_2_5)\.dll$"]
        else:
            patterns = [r"/runtime/lib/intel64/[^/]+$"]
        return [Selection("openvino", patterns, licenses=OPENVINO_LICENSES)]
    if ep == "dml":
        return [Selection("directml", [rf"^bin/{_dml_arch(platform)}-win/DirectML\.dll$"], licenses=DML_LICENSES)]
    if ep in ("migraphx", "amdgpu"):
        # TheRock's runtime wheels, unpacked into rocm/ with their layout (as pip installs them): rocBLAS and hipBLASLt
        # find their kernels, comgr its device libraries relative to their own files. On Linux ROCm is a system install.
        if os_name != "windows":
            raise SystemExit(f"{ep}: install ROCm on {os_name} yourself")
        wheels = ["rocm_sdk_core-", "rocm_sdk_libraries-", *(f"rocm_sdk_device_{t}-" for t in amd_targets)]
        return [Selection("rocm-wheels", archives=wheels,
                          tree=[DevRule(r"^(?P<rel>_rocm_sdk_(?:core|libraries)/(?:bin|lib|\.kpack)/.+)$", "rocm")])]
    if ep == "hip":
        # hip-ep's HIP EP, which the AMDGPU EP's hip-backend loads by name from its own directory. On Windows with the
        # ROCm runtime it is built with: next to amdgpu-ep.dll, so the process uses that one rather than rocm/; on Linux
        # the system ROCm. Kernel libraries and the clang resources keep their place relative to the libraries.
        if os_name == "windows":
            dlls = (r"hipgpu|amdhip64_7|amd_comgr\d+|hiprtc\d+|hiprtc-builtins\d+|hipblaslt|tensilelite-host|amdmlss|"
                    r"custom_kernels_[\w-]+")
            return [Selection("hip-ep", [rf"^bin/(?:{dlls})\.dll$"],
                              beside=[DevRule(r"^bin/(?P<rel>hipblaslt/library/.+)$", ".")])]
        return [Selection("hip-ep", [r"^lib/lib(?:hipgpu|custom_kernels_[\w-]+)\.so$"],
                          beside=[DevRule(r"^lib/(?P<rel>clang/.+)$", ".")])]
    if ep == "neuropilot":  # the EP loads it by name from the loader search path (next to it, or LD_LIBRARY_PATH)
        target = "android-arm64-v8a" if os_name == "android" else "linux-aarch64"
        return [Selection("neuron-op-capability", [rf"-{target}/libneuron_op_capability\.so$"])]
    raise SystemExit(f"unknown EP: {ep}")


def _headers(subdir: str, dest: str, depth: int = 1) -> DevRule:
    """Headers below <top-level dirs>/<subdir>/ (archives have `depth` top-level directories) into dest/."""
    # extension-less files are C++ standard-library style headers (CCCL's cuda/std/version, ...)
    return DevRule(rf"^(?:[^/]+/){{{depth}}}{subdir}/(?P<rel>.+\.(?:h|hpp|hxx|inl|inc|cuh|cu)|(?:.+/)?[^/.]+)$", dest)


def _import_libs(platform: str, os_name: str, names: str, subdir: str = "lib", depth: int = 1) -> list[DevRule]:
    """Windows import libraries below <top-level dirs>/<subdir>/[x64|arm64/] into lib/ (the bundled .so files serve
    as link files elsewhere)."""
    if os_name != "windows":
        return []
    arch = "arm64" if platform == "windows-arm64" else "x64"
    return [DevRule(rf"^(?:[^/]+/){{{depth}}}{subdir}/(?:{arch}/)?(?P<rel>(?:{names})\.lib)$", "lib")]


def dev_selections_for(ep: str, platform: str, os_name: str, cuda_major: str) -> list[Selection]:
    """The headers and import libraries of an EP's backend SDK, laid out like the bundles' own backend files. They come
    from the archives its runtime is downloaded from, plus the CUDA header-only archives (CCCL, CRT)."""
    libs = CUDA_RUNTIME.get(ep, ())
    cuda_archives = [CUDA_ARCHIVES[n] for n in libs] + ["cuda_crt", "cccl", "cuda_cccl"] + (
        ["cuda_nvcc"] if cuda_major == "12" else [])
    cuda = Selection("cuda", archives=cuda_archives,
                     dev=[_headers("include", "include/cuda"),
                          *_import_libs(platform, os_name, "|".join(CUDA_IMPORT_LIBS[n] for n in libs))])
    if ep == "cuda":
        return [cuda, Selection("cudnn", dev=[_headers("include", "include/cudnn"),
                                              *_import_libs(platform, os_name, r"cudnn[\w]*")])]
    if ep == "tensorrt":
        return [cuda, Selection("tensorrt", dev=[_headers("include", "include/tensorrt"),
                                                 *_import_libs(platform, os_name, r"nvinfer[\w]*|nvonnxparser[\w]*")])]
    if ep == "nv-tensorrt-rtx":
        return [cuda, Selection("tensorrt-rtx", dev=[_headers("include", "include/tensorrt_rtx"),
                                                     *_import_libs(platform, os_name, r"tensorrt[\w]*")])]
    if ep == "qnn":  # the zip holds qairt/<version>/include/QNN/; QNN backends are loaded at run time, not linked
        return [Selection("qairt", dev=[_headers("include/QNN", "include/qnn", depth=2)])]
    if ep == "openvino":
        return [Selection("openvino", dev=[_headers("runtime/include", "include"),
                                           *_import_libs(platform, os_name, r"openvino[\w]*",
                                                         "runtime/lib/intel64/Release")], licenses=OPENVINO_LICENSES)]
    if ep == "hip":  # the import libraries of hipgpu and its runtime (Windows; the archives have no headers)
        if os_name != "windows":
            return []
        return [Selection("hip-ep", dev=[DevRule(r"^lib/(?P<rel>(?:hipgpu|HipCInterface|hipblaslt|custom_kernels_[\w-]+)\.lib)$",
                                                 "lib")])]
    if ep == "dml":
        return [Selection("directml", dev=[DevRule(r"^include/(?P<rel>[^/]+\.h)$", "include"),
                                           DevRule(rf"^bin/{_dml_arch(platform)}-win/(?P<rel>DirectML\.lib)$", "lib")],
                          licenses=DML_LICENSES)]
    return []  # Vitis AI, CANN and NeuroPilot have no separate development files


EPS = {
    "cuda": "CUDA EP (in-tree or plugin): cudart, cuBLAS, cuFFT, cuRAND, NVRTC, nvJitLink, cuDNN",
    "tensorrt": "TensorRT EP (in-tree or plugin): TensorRT runtime plus cudart / cuBLAS",
    "nv-tensorrt-rtx": "NVIDIA TensorRT RTX EP (in-tree or plugin): TensorRT-RTX runtime plus cudart",
    "qnn": "Qualcomm QNN EP (in-tree or plugin): QAIRT backends and Hexagon skel libraries",
    "vitisai": "AMD Vitis AI EP: closed-source backend DLLs from the voe wheel (Windows x64)",
    "cann": "Huawei CANN EP: installs the CANN toolkit (its libraries need the full install and set_env.sh)",
    "openvino": "Intel OpenVINO EP: OpenVINO runtime and device plugins (the files its redist.txt lists)",
    "dml": "DirectML EP (in-tree or the AMD plugin): DirectML.dll from the Microsoft.AI.DirectML NuGet package",
    "neuropilot": "MediaTek NeuroPilot plugin EP: the prebuilt op capability library it loads at run time",
    "hip": "HIP backend of the AMDGPU plugin EP: hipgpu of ROCm/hip-ep, with its ROCm runtime on Windows",
    "migraphx": "AMD MIGraphX plugin EP on Windows: the ROCm runtime (TheRock wheels, the device libraries of the GPU "
                "targets), in rocm/",
    "amdgpu": "AMD AMDGPU plugin EP on Windows: the ROCm runtime (TheRock wheels, the device libraries of the GPU "
              "targets), in rocm/",
}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _wanted(path: str, patterns: list[re.Pattern]) -> bool:
    return not SKIP_DIRS.search("/" + path) and any(p.search(path) for p in patterns)


def _dev_target(path: str, dev: list[tuple[re.Pattern, Path]]) -> Path | None:
    """Where a development file goes (None: not one)."""
    if SKIP_DEV_DIRS.search("/" + path):
        return None
    for pattern, dest in dev:
        if m := pattern.search(path):
            return dest / m.group("rel")
    return None


def extract_selected(archive: Path, patterns: list[re.Pattern], dest: Path, licenses: Path,
                     dev: list[tuple[re.Pattern, Path]] = (), license_patterns: list[re.Pattern] = ()) -> list[str]:
    """Copy matching files from archive: runtime files into dest (flattened), development files to the targets of
    their dev rules (keeping their directory tree), license files into licenses/. Returns the copied names."""
    copied: list[str] = []
    name = archive.name.lower()
    is_license = lambda m: LICENSE_FILE.search(m) or any(p.search(m) for p in license_patterns)
    selected = lambda m: is_license(m) or _wanted(m, patterns) or _dev_target(m, dev)

    def store(member: str, data: bytes) -> None:
        if is_license(member):
            target = licenses / member.replace("/", "__")
        elif _wanted(member, patterns):
            target = dest / Path(member).name
            copied.append(target.name)
        elif target := _dev_target(member, dev):
            copied.append(target.name)
        else:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    links: dict[str, str] = {}  # runtime library symlinks (name -> target name), made once their targets exist
    if name.endswith((".zip", ".whl", ".nupkg")):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if not info.is_dir() and selected(info.filename):
                    store(info.filename, zf.read(info))
    else:
        with _open_tar(archive) as tf:
            for member in tf:
                if not selected(member.name):
                    continue
                if member.issym() and _wanted(member.name, patterns):
                    links[Path(member.name).name] = Path(member.linkname).name
                elif member.isfile():
                    store(member.name, tf.extractfile(member).read())
    for link in links:
        _link(dest, link, links, copied)
    return copied


def _link(dest: Path, name: str, links: dict[str, str], copied: list[str]) -> None:
    """Recreate a library symlink (e.g. libcudart.so.13 -> libcudart.so.13.4.92). Where symlinks cannot be created
    (Windows without Developer Mode), hard-link or copy the file it resolves to, so that a Linux bundle prepared there
    still has its SONAME files."""
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / name
    target.unlink(missing_ok=True)
    try:
        target.symlink_to(links[name])
    except OSError:
        real, seen = links[name], {name}
        while real in links and real not in seen:
            seen.add(real)
            real = links[real]
        if not (dest / real).is_file():
            return
        try:
            os.link(dest / real, target)
        except OSError:
            shutil.copy2(dest / real, target)
    copied.append(name)


@contextlib.contextmanager
def _open_tar(archive: Path):
    """Open a tar archive for streaming (no random access into multi-GB tarballs)."""
    if not archive.name.lower().endswith(".zst"):
        with tarfile.open(archive, "r|*") as tf:
            yield tf
        return
    if hasattr(tarfile.TarFile, "zstopen"):  # Python 3.14+
        with tarfile.open(archive, "r|zst") as tf:
            yield tf
        return
    cmd, prefix = _zstd_decompressor()
    proc = subprocess.Popen([*cmd, prefix + str(archive)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
            yield tf
        while proc.stdout.read(1 << 20):  # let the decompressor write the end of the stream before checking its status
            pass
    finally:
        proc.stdout.close()
        code = proc.wait()
    if code:
        raise subprocess.CalledProcessError(code, proc.args)


@functools.lru_cache(maxsize=None)
def _zstd_decompressor() -> tuple[list[str], str]:
    """A command that writes the tar stream of a .tar.zst file to stdout: (arguments, prefix of the archive path)."""
    if zstd := shutil.which("zstd"):
        return [zstd, "-dcq"], ""
    if tar := shutil.which("tar"):  # libarchive's tar copies the entries of an @archive into a new uncompressed one
        version = subprocess.run([tar, "--version"], capture_output=True, text=True, errors="replace").stdout
        if "bsdtar" in version and "zstd" in version:
            return [tar, "-cf", "-", "--format", "pax"], "@"
    for name in ("7zz", "7z", "7za"):  # p7zip (7z / 7za 16.02) cannot unpack zstd
        if exe := shutil.which(name):
            formats = subprocess.run([exe, "i"], capture_output=True, text=True, errors="replace").stdout
            if re.search(r"(?i)\bzstd\b", formats):
                return [exe, "x", "-so", "-tzstd"], ""
    return [str(_download_7zip()), "x", "-so", "-tzstd"], ""


def _download_7zip() -> Path:
    host = dl.detect_platform()
    if host not in SEVEN_ZIP:
        raise SystemExit("cannot read .tar.zst archives: install zstd or 7-Zip, or use Python 3.14+")
    tmp = Path(tempfile.mkdtemp(prefix="7zip-"))
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)
    print(f"    no zstd decompressor found, downloading 7-Zip to {tmp}")
    for filename, sha256 in SEVEN_ZIP[host]:
        dl.download_file(dl.FileItem(SEVEN_ZIP_URL + filename, filename, sha256), tmp)
    package = tmp / SEVEN_ZIP[host][-1][0]
    if package.suffix == ".exe":  # the installer is a 7z archive, which 7zr (7z format only) unpacks
        subprocess.run([str(tmp / "7zr.exe"), "x", "-y", "-bso0", f"-o{tmp}", str(package), "7z.exe", "7z.dll"],
                       check=True)
        return tmp / "7z.exe"
    exe = tmp / "7zz"
    with tarfile.open(package) as tf:
        exe.write_bytes(tf.extractfile("7zz").read())
    exe.chmod(0o755)
    return exe


def install_cann(run_file: Path, dest: Path) -> Path:
    prefix = dest / "cann"
    prefix.mkdir(parents=True, exist_ok=True)
    run_file.chmod(0o755)
    subprocess.run([str(run_file), "--install", "--quiet", f"--install-path={prefix}"], check=True)
    return prefix


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def pins_from_build_info(sdks: dict[str, str]) -> dict[str, str]:
    """Translate the resolved versions recorded in BUILD_INFO.json / BUNDLE_INFO.json (sdk-versions.json) into
    download-script version selectors."""
    pins = {}
    for comp, version in sdks.items():
        if comp == "tensorrt":  # resolved as 10.15.1.29; the resolver selects a NVIDIA/TensorRT release branch
            pins[comp] = "release/" + ".".join(version.split(".")[:2])
        elif comp == "tensorrt-rtx":  # resolved as 1.6.1.120; the resolver selects a release series
            pins[comp] = ".".join(version.split(".")[:2])
        elif comp == "vitisai":  # onnxruntime-vitisai-1.23.2_voe-1.7.0_ryzenai-dynamic-dispatch-1.7.0
            for part in version.split("_"):
                pkg, _, ver = part.rpartition("-")
                pins[pkg] = ver
        elif comp in ("cuda", "cudnn", "qairt", "cann", "rocm", "openvino", "directml", "neuron-op-capability", "hip-ep"):
            pins[comp] = version
        if comp == "rocm":  # the Windows AMD plugin EPs record the version of TheRock's wheels they are built with
            pins["rocm-wheels"] = version
    return pins


def parse_eps(values: list[str]) -> dict[str, set[str]]:
    """["cuda", "qnn:dev", "tensorrt:runtime+dev"] -> {EP: parts}; a bare EP means its runtime files."""
    eps: dict[str, set[str]] = {}
    for value in values:
        ep, _, parts = value.partition(":")
        parts = set((parts or "runtime").split("+"))
        if ep not in EPS or not parts <= set(PARTS):
            raise SystemExit(f"invalid --ep {value!r}: use <ep>[:runtime|dev|runtime+dev] with an EP from {', '.join(EPS)}")
        eps.setdefault(ep, set()).update(parts)
    return eps


def merge_selections(sels: list[Selection]) -> dict[str, Selection]:
    """Merge the selections of several EPs per component (an empty archive list means all archives)."""
    merged: dict[str, Selection] = {}
    for sel in sels:
        cur = merged.get(sel.component)
        if cur is None:
            merged[sel.component] = Selection(sel.component, list(sel.patterns), list(sel.archives), sel.install,
                                              list(sel.dev), list(sel.licenses), list(sel.tree), list(sel.beside))
            continue
        cur.install = cur.install or sel.install
        cur.patterns += [p for p in sel.patterns if p not in cur.patterns]
        cur.dev += [r for r in sel.dev if r not in cur.dev]
        cur.licenses += [p for p in sel.licenses if p not in cur.licenses]
        cur.tree += [r for r in sel.tree if r not in cur.tree]
        cur.beside += [r for r in sel.beside if r not in cur.beside]
        cur.archives = cur.archives + [a for a in sel.archives if a not in cur.archives] \
            if cur.archives and sel.archives else []
    return merged


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Download closed-source EP runtimes and extract only the runtime files")
    ap.add_argument("--list", action="store_true", help="list the supported EPs")
    ap.add_argument("--package-dir", type=Path, help="extracted bundle; EPs, target and versions come from its "
                                                     "BUNDLE_INFO.json")
    ap.add_argument("--ep", nargs="+", metavar="EP[:PARTS]",
                    help=f"EPs to fetch for ({', '.join(EPS)}), optionally with the parts: runtime (default), dev, "
                         "runtime+dev")
    ap.add_argument("--target", choices=sorted(TARGETS), help="target triple (default: from BUNDLE_INFO.json or this host)")
    ap.add_argument("--cuda-major", choices=("12", "13"), help="CUDA major version (default: from BUNDLE_INFO.json, else 13)")
    ap.add_argument("--amd-gpu-targets", nargs="+", metavar="GFX",
                    help="GPU targets of the ROCm device libraries (default: from BUNDLE_INFO.json)")
    ap.add_argument("--pin", nargs="+", default=[], metavar="NAME=VER", help="override SDK versions, same syntax as download_ep_packages.py")
    ap.add_argument("--dest", type=Path, help="output directory of the runtime files (default: the bundle's bin/ on "
                                              "Windows or lib/ elsewhere, otherwise ./ep-runtime/<target>)")
    ap.add_argument("--cache", type=Path, default=dl.DEFAULT_DEST, help=f"download cache (default: {dl.DEFAULT_DEST})")
    ap.add_argument("--licenses-dir", type=Path, help="where the license texts go (default: <package-dir or dest>/"
                                                      "third_party_licenses); THIRD_PARTY_RUNTIME.json is written next to it")
    ap.add_argument("--dry-run", action="store_true", help="resolve the SDK files and show the extraction rules only")
    ap.add_argument("--yes", action="store_true", help="accept the vendor licenses a bundle lists without asking")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    if args.list:
        for name, desc in EPS.items():
            print(f"{name:16} {desc}")
        print(f"\ntargets: {', '.join(TARGETS)}")
        return 0

    info: dict = {}
    if args.package_dir:  # ci/bundle.py records what the bundle still needs
        bundle_file = args.package_dir / "BUNDLE_INFO.json"
        if not bundle_file.exists():
            raise SystemExit(f"{bundle_file} not found")
        info = json.loads(bundle_file.read_text(encoding="utf-8"))["fetch_restricted"]
    target = args.target or info.get("target_triple") or next(
        (t for t, v in TARGETS.items() if v[0] == dl.detect_platform()), None)
    if target not in TARGETS:
        raise SystemExit(f"target {target} has no closed-source runtime to fetch (supported: {', '.join(TARGETS)})")
    platform, os_name, qnn_abi = TARGETS[target]

    eps = parse_eps(args.ep or info.get("eps", []))
    if not eps:
        print("nothing to do: no closed-source SDK files are needed")
        return 0
    if info.get("licenses") and not args.dry_run:
        print("The files are downloaded from their vendors and are subject to the vendors' licenses:")
        for lic in info["licenses"]:
            print(f"  {lic['name']}: {lic['license']}")
        if not args.yes:
            if not sys.stdin.isatty():
                raise SystemExit("run again with --yes to accept these licenses")
            if input("Accept these licenses and continue? [y/N] ").strip().lower() not in ("y", "yes"):
                return 1
    cuda_major = args.cuda_major or info.get("cuda_major") or "13"
    amd_targets = tuple(args.amd_gpu_targets or info.get("amd_gpu_targets") or ())
    pins = pins_from_build_info(info.get("third_party_sdks", {}))
    pins.update(dict(p.split("=", 1) for p in args.pin))

    if args.dest:
        dest = args.dest
    elif args.package_dir:
        dest = args.package_dir / ("bin" if os_name == "windows" else "lib")
    else:
        dest = Path("ep-runtime") / target
    dest = dest.resolve()
    root = (args.package_dir or dest).resolve()  # development files go to its include/ and lib/
    licenses = args.licenses_dir.resolve() if args.licenses_dir else root / "third_party_licenses"

    merged = merge_selections([sel for ep, parts in eps.items() for sel in
                               (selections_for(ep, platform, os_name, qnn_abi, amd_targets) if "runtime" in parts else []) +
                               (dev_selections_for(ep, platform, os_name, cuda_major) if "dev" in parts else [])])
    labels = ", ".join(f"{ep}:{'+'.join(sorted(parts))}" for ep, parts in eps.items())
    print(f"target {target}, EPs: {labels}, CUDA {cuda_major}.x, runtime files: {dest}, development files: {root}")
    ctx = dl.Context(platform, cuda_major, "ubuntu24", pins)
    report_file = licenses.parent / "THIRD_PARTY_RUNTIME.json"
    report = json.loads(report_file.read_text(encoding="utf-8")) if report_file.exists() else {}
    report["target_triple"] = target
    report["eps"] = sorted(set(report.get("eps", [])) | {f"{ep}:{p}" for ep, parts in eps.items() for p in parts})
    components = report.setdefault("components", {})
    # a component's resolver may not filter its files by platform (the Vitis AI wheels are win_amd64 only)
    if unsupported := [c for c in merged if platform not in dl.component(c).platforms]:
        raise SystemExit(f"no files for {target}: {', '.join(f'{c} ({' / '.join(dl.component(c).platforms)})' for c in unsupported)}")
    for comp_name, sel in merged.items():
        comp = dl.component(comp_name)
        print(f"==> {comp_name}")
        file_filter = "^(" + "|".join(re.escape(a) for a in sel.archives) + ")" if sel.archives else None
        out, resolved = dl.fetch(comp, ctx, args.cache.resolve(), dry_run=args.dry_run, file_filter=file_filter)
        files = resolved.files
        if args.dry_run:
            for p in sel.patterns or ([] if sel.dev or sel.tree or sel.beside else ["(vendor installer)"]):
                print(f"    extract: {p}")
            for r in sel.tree:
                print(f"    extract with its tree: {r.pattern} -> {r.dest}/")
            for r in sel.beside:
                print(f"    extract with its tree, next to the runtime files: {r.pattern} -> {r.dest}/")
            for r in sel.dev:
                print(f"    develop: {r.pattern} -> {r.dest}/")
            continue
        entry = components.setdefault(comp_name, {"files": []})
        entry.update({"version": resolved.version, "source": comp.source})
        if sel.install:
            prefix = install_cann(out / files[0].filename, dest)
            entry["install_prefix"] = str(prefix)
            print(f"    installed to {prefix}; run `source {prefix}/cann/set_env.sh` (8.x: ascend-toolkit/set_env.sh) before use")
        else:
            patterns = [re.compile(p) for p in sel.patterns]
            dev = [(re.compile(r.pattern), root / r.dest) for r in sel.dev + sel.tree] +                 [(re.compile(r.pattern), dest / r.dest) for r in sel.beside]
            license_patterns = [re.compile(p) for p in sel.licenses]
            new: list[str] = []
            for f in files:
                if f.filename.lower().endswith((".txt", "license")):  # a license text next to the archives
                    (licenses / comp_name).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(out / f.filename, licenses / comp_name / f.filename)
                    continue
                new += extract_selected(out / f.filename, patterns, dest, licenses / comp_name, dev, license_patterns)
            if not new:
                raise SystemExit(f"no matching files in {comp_name} {resolved.version}")
            entry["files"] = sorted(set(entry.get("files", [])) | set(new))
            shown = sorted(set(new))
            print(f"    {len(shown)} files: {', '.join(shown[:40])}{' ...' if len(shown) > 40 else ''}")

    if not args.dry_run:
        report["note"] = ("These files are third-party software under their vendors' licenses (see third_party_licenses/), "
                          "not under the license of the ONNX Runtime binaries next to them.")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"done: runtime files in {dest}, development files in {root}, license texts in {licenses}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
