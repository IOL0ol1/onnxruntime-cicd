#!/usr/bin/env python3
"""Download the closed-source SDKs that some providers / EPs need at runtime, and extract only the runtime files.

The release packages of this repository never contain third-party SDK files, because CUDA, cuDNN, TensorRT,
TensorRT-RTX, QAIRT (QNN), CANN and the Vitis AI runtime cannot be redistributed. This script fetches them from
the same no-account official sources as scripts/download_ep_packages.py and copies just the runtime libraries
next to the ONNX Runtime binaries.

Typical use: point it at an extracted release package. The EPs, target triple, CUDA major version and the exact
SDK versions used for the build are read from the package's BUILD_INFO.json:

    python scripts/fetch_ep_runtime.py --package-dir onnxruntime-all-1.31.0-x86_64-pc-windows-msvc

Or select EPs and a target explicitly:

    python scripts/fetch_ep_runtime.py --ep cuda tensorrt --target x86_64-unknown-linux-gnu --dest ./runtime
    python scripts/fetch_ep_runtime.py --ep qnn --target aarch64-pc-windows-msvc --dry-run
    python scripts/fetch_ep_runtime.py --list

Only the Python standard library is required (.tar.zst archives also need a `tar` with zstd support).
"""

from __future__ import annotations

import argparse
import importlib.util
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

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("dl", Path(__file__).with_name("download_ep_packages.py"))
dl = importlib.util.module_from_spec(_spec)
sys.modules["dl"] = dl
_spec.loader.exec_module(dl)

# target triple -> (download platform, runtime OS, QAIRT library directory)
TARGETS = {
    "x86_64-unknown-linux-gnu": ("linux-x86_64", "linux", "x86_64-linux-clang"),
    "aarch64-unknown-linux-gnu": ("linux-aarch64", "linux", "aarch64-oe-linux-gcc11.2"),
    "x86_64-pc-windows-msvc": ("windows-x86_64", "windows", "x86_64-windows-msvc"),
    "aarch64-pc-windows-msvc": ("windows-arm64", "windows", "aarch64-windows-msvc"),
    "arm64ec-pc-windows-msvc": ("windows-arm64x", "windows", "arm64x-windows-msvc"),
    "arm64x-pc-windows-msvc": ("windows-arm64x", "windows", "arm64x-windows-msvc"),
    "aarch64-linux-android": ("linux-x86_64", "android", "aarch64-android"),
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


@dataclass
class Selection:
    """Which runtime files to take from one SDK component."""
    component: str
    patterns: list[str] = field(default_factory=list)       # regexes on the path inside the archive
    archives: list[str] = field(default_factory=list)       # only download archives whose name starts with one of these
    install: bool = False                                    # run the vendor installer instead of extracting


def _libs(os_name: str, *pairs: tuple[str, str]) -> list[str]:
    idx = 1 if os_name == "windows" else 0
    return [rf"(^|/){p[idx]}$" for p in pairs]


# CUDA library -> redist archive that contains it
CUDA_ARCHIVES = {"cudart": "cuda_cudart", "cublas": "libcublas", "cufft": "libcufft", "curand": "libcurand",
                 "nvrtc": "cuda_nvrtc", "nvjitlink": "libnvjitlink"}


def selections_for(ep: str, os_name: str, qnn_abi: str) -> list[Selection]:
    cuda = lambda *names: Selection("cuda", _libs(os_name, *(CUDA_LIBS[n] for n in names)),
                                    [CUDA_ARCHIVES[n] for n in names])
    if ep == "cuda":
        return [cuda("cudart", "cublas", "cufft", "curand", "nvrtc", "nvjitlink"), Selection("cudnn", _libs(os_name, CUDNN_LIBS))]
    if ep == "tensorrt":
        return [cuda("cudart", "cublas", "nvjitlink"), Selection("tensorrt", _libs(os_name, TENSORRT_LIBS))]
    if ep == "nv-tensorrt-rtx":
        return [cuda("cudart"), Selection("tensorrt-rtx", _libs(os_name, TENSORRT_RTX_LIBS))]
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
    raise SystemExit(f"unknown EP: {ep}")


EPS = {
    "cuda": "CUDA EP (in-tree or plugin): cudart, cuBLAS, cuFFT, cuRAND, NVRTC, nvJitLink, cuDNN",
    "tensorrt": "TensorRT EP (in-tree or plugin): TensorRT runtime plus cudart / cuBLAS",
    "nv-tensorrt-rtx": "NVIDIA TensorRT RTX EP (in-tree or plugin): TensorRT-RTX runtime plus cudart",
    "qnn": "Qualcomm QNN EP (in-tree or plugin): QAIRT backends and Hexagon skel libraries",
    "vitisai": "AMD Vitis AI EP: closed-source backend DLLs from the voe wheel (Windows x64)",
    "cann": "Huawei CANN EP: installs the CANN toolkit (its libraries need the full install and set_env.sh)",
}
# (recipe, variant) in BUILD_INFO.json -> EPs that need closed-source runtime files
PACKAGE_EPS = {
    ("ort", "cuda"): ["cuda", "tensorrt"], ("ort", "cuda-plugin"): ["cuda"],
    ("ort", "nv-tensorrt-rtx"): ["nv-tensorrt-rtx"], ("ort", "qnn"): ["qnn"],
    ("ep-tensorrt", "default"): ["tensorrt"], ("ep-nv-tensorrt-rtx", "default"): ["nv-tensorrt-rtx"],
    ("ep-qnn", "default"): ["qnn"],
}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _wanted(path: str, patterns: list[re.Pattern]) -> bool:
    return not SKIP_DIRS.search("/" + path) and any(p.search(path) for p in patterns)


def extract_selected(archive: Path, patterns: list[re.Pattern], dest: Path, licenses: Path) -> list[str]:
    """Copy matching files from archive into dest (flattened); license files go to licenses/. Returns copied names."""
    copied: list[str] = []
    name = archive.name.lower()

    def store(member: str, data: bytes) -> None:
        if LICENSE_FILE.search(member):
            target = licenses / member.replace("/", "__")
        elif _wanted(member, patterns):
            target = dest / Path(member).name
            copied.append(target.name)
        else:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    if name.endswith((".zip", ".whl")):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if not info.is_dir() and (LICENSE_FILE.search(info.filename) or _wanted(info.filename, patterns)):
                    store(info.filename, zf.read(info))
    elif name.endswith(".zst"):  # Python's tarfile cannot read zstd before 3.14; use the system tar
        listing = subprocess.run(["tar", "--zstd", "-tf", str(archive)], check=True, capture_output=True, text=True).stdout
        members = [m for m in listing.splitlines() if not m.endswith("/") and (LICENSE_FILE.search(m) or _wanted(m, patterns))]
        with tempfile.TemporaryDirectory() as tmp:
            if members:
                subprocess.run(["tar", "--zstd", "-xf", str(archive), "-C", tmp, *members], check=True)
            for m in members:
                src = Path(tmp) / m
                if src.is_symlink() and os.name != "nt":
                    _symlink(dest, src.name, os.readlink(src), copied)
                elif src.is_file():
                    store(m, src.read_bytes())
    else:
        with tarfile.open(archive, "r|*") as tf:  # streaming: no random access into multi-GB tarballs
            for member in tf:
                if not (LICENSE_FILE.search(member.name) or _wanted(member.name, patterns)):
                    continue
                if member.issym() and os.name != "nt":
                    _symlink(dest, Path(member.name).name, member.linkname, copied)
                elif member.isfile():
                    store(member.name, tf.extractfile(member).read())
    return copied


def _symlink(dest: Path, name: str, link: str, copied: list[str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / name
    target.unlink(missing_ok=True)
    target.symlink_to(Path(link).name)  # e.g. libcudart.so.13 -> libcudart.so.13.4.92
    copied.append(name)


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
    """Translate the resolved versions recorded in BUILD_INFO.json into download-script version selectors."""
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
        elif comp in ("cuda", "cudnn", "qairt", "cann", "rocm", "openvino"):
            pins[comp] = version
    return pins


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Download closed-source EP runtimes and extract only the runtime files")
    ap.add_argument("--list", action="store_true", help="list the supported EPs")
    ap.add_argument("--package-dir", type=Path, help="extracted release package; EPs, target and versions come from BUILD_INFO.json")
    ap.add_argument("--ep", nargs="+", choices=sorted(EPS), help="EPs to fetch runtimes for")
    ap.add_argument("--target", choices=sorted(TARGETS), help="target triple (default: from BUILD_INFO.json or this host)")
    ap.add_argument("--cuda-major", choices=("12", "13"), help="CUDA major version (default: from BUILD_INFO.json, else 13)")
    ap.add_argument("--pin", nargs="+", default=[], metavar="NAME=VER", help="override SDK versions, same syntax as download_ep_packages.py")
    ap.add_argument("--dest", type=Path, help="output directory (default: the package's bin/ on Windows or lib/ elsewhere, "
                                              "otherwise ./ep-runtime/<target>)")
    ap.add_argument("--cache", type=Path, default=dl.DEFAULT_DEST, help=f"download cache (default: {dl.DEFAULT_DEST})")
    ap.add_argument("--dry-run", action="store_true", help="resolve the SDK files and show the extraction rules only")
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
    if args.package_dir:
        info_file = args.package_dir / "BUILD_INFO.json"
        if not info_file.exists():
            raise SystemExit(f"{info_file} not found")
        info = json.loads(info_file.read_text(encoding="utf-8"))
    target = args.target or info.get("target_triple") or next(
        (t for t, v in TARGETS.items() if v[0] == dl.detect_platform()), None)
    if target not in TARGETS:
        raise SystemExit(f"target {target} has no closed-source runtime to fetch (supported: {', '.join(TARGETS)})")
    platform, os_name, qnn_abi = TARGETS[target]

    eps = args.ep
    if not eps:
        recipe, _, variant = info.get("recipe", "/").partition("/")
        eps = PACKAGE_EPS.get((recipe, variant or "default"), [])
        if (recipe, variant) == ("ort", "default") and target == "x86_64-pc-windows-msvc":
            eps = ["vitisai"]  # the default Windows x64 package includes the Vitis AI bridge
        if (recipe, variant) == ("ort", "all"):  # the EP set depends on the target; follow the SDKs it was built with
            eps = [ep for ep in ("cuda", "tensorrt", "cann") if ep in info.get("third_party_sdks", {})]
        if not eps:
            print("nothing to do: this package does not use a closed-source SDK at runtime")
            return 0
    cuda_major = args.cuda_major or info.get("cuda_major") or "13"
    pins = pins_from_build_info(info.get("third_party_sdks", {}))
    pins.update(dict(p.split("=", 1) for p in args.pin))

    if args.dest:
        dest = args.dest
    elif args.package_dir:
        dest = args.package_dir / ("bin" if os_name == "windows" else "lib")
    else:
        dest = Path("ep-runtime") / target
    dest = dest.resolve()
    licenses = dest / "third_party_licenses"

    # Merge the selections of all requested EPs per component
    merged: dict[str, Selection] = {}
    for ep in eps:
        for sel in selections_for(ep, os_name, qnn_abi):
            cur = merged.setdefault(sel.component, Selection(sel.component, [], [], sel.install))
            cur.patterns += [p for p in sel.patterns if p not in cur.patterns]
            cur.archives += [a for a in sel.archives if a not in cur.archives]

    print(f"target {target}, EPs: {', '.join(eps)}, CUDA {cuda_major}.x, output: {dest}")
    ctx = dl.Context(platform, cuda_major, "ubuntu24", pins)
    report = {"target_triple": target, "eps": eps, "components": {}}
    for comp_name, sel in merged.items():
        comp = dl.component(comp_name)
        print(f"==> {comp_name}")
        file_filter = "^(" + "|".join(re.escape(a) for a in sel.archives) + ")" if sel.archives else None
        out, resolved = dl.fetch(comp, ctx, args.cache.resolve(), dry_run=args.dry_run, file_filter=file_filter)
        files = resolved.files
        if args.dry_run:
            for p in sel.patterns or ["(vendor installer)"]:
                print(f"    extract: {p}")
            continue
        entry = {"version": resolved.version, "source": comp.source, "files": []}
        if sel.install:
            prefix = install_cann(out / files[0].filename, dest)
            entry["install_prefix"] = str(prefix)
            print(f"    installed to {prefix}; run `source {prefix}/cann/set_env.sh` (8.x: ascend-toolkit/set_env.sh) before use")
        else:
            patterns = [re.compile(p) for p in sel.patterns]
            for f in files:
                entry["files"] += extract_selected(out / f.filename, patterns, dest, licenses / comp_name)
            if not entry["files"]:
                raise SystemExit(f"no runtime files matched in {comp_name} {resolved.version}")
            print(f"    {len(entry['files'])} files: {', '.join(sorted(set(entry['files'])))}")
        report["components"][comp_name] = entry

    if not args.dry_run:
        report["note"] = ("These files are third-party software under their vendors' licenses (see third_party_licenses/). "
                          "Do not redistribute them.")
        (dest / "THIRD_PARTY_RUNTIME.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"done: runtime files in {dest}, license texts in {licenses}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
