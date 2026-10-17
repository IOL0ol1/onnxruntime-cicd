#!/usr/bin/env python3
"""Download the closed-source SDKs that some providers / EPs need at runtime, and extract only the runtime files.

The packages bundle the small vendor runtimes whose licenses allow distributing them with an application (QAIRT,
TensorRT for RTX with the CUDA runtime; ci/build.py uses this script for that). The CUDA libraries, cuDNN and TensorRT
are left out of the packages for their size (bundles add them, see ci/bundle.py), the CANN toolkit and the Vitis AI
runtime for their licenses. This script fetches those from the same no-account official sources as
scripts/download_ep_packages.py and copies just the runtime libraries next to the ONNX Runtime binaries.

Typical use: point it at an extracted package. The EPs, target triple, CUDA major version and the exact
SDK versions used for the build are read from the package's BUILD_INFO.json:

    python scripts/fetch_ep_runtime.py --package-dir onnxruntime-all-1.31.0-x86_64-pc-windows-msvc

Or select EPs and a target explicitly:

    python scripts/fetch_ep_runtime.py --ep cuda tensorrt --target x86_64-unknown-linux-gnu --dest ./runtime
    python scripts/fetch_ep_runtime.py --ep qnn --target aarch64-pc-windows-msvc --dry-run
    python scripts/fetch_ep_runtime.py --list

An EP can be suffixed with the parts to fetch: `cuda:runtime` (the default), `cuda:dev` (the backend's headers into
include/<backend>/ and its import libraries into lib/, which the vendors' licenses do not let us redistribute) or
`cuda:runtime+dev`. Bundles built from a bundle request (ci/bundle.py) record what they still need in
BUNDLE_INFO.json; their fetch-restricted script runs this script with --package-dir on the bundle.

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
SKIP_DEV_DIRS = re.compile(r"(?i)/(samples?|python|doc|docs)/")
PARTS = ("runtime", "dev")


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
    if ep == "openvino":
        return []  # the packages bundle the OpenVINO runtime
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
    """The headers and import libraries of an EP's backend SDK, laid out like the bundles' own backend files."""
    cuda_archives = [*CUDA_ARCHIVES.values(), "cuda_crt", "cccl", "cuda_cccl"] + (["cuda_nvcc"] if cuda_major == "12" else [])
    cuda = Selection("cuda", archives=cuda_archives,
                     dev=[_headers("include", "include/cuda"),
                          *_import_libs(platform, os_name, "cudart|cublas|cublasLt|cufft|curand|nvrtc|nvJitLink|cuda")])
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
                                                         "runtime/lib/intel64/Release")])]
    return []  # Vitis AI and CANN have no separate development files


EPS = {
    "cuda": "CUDA EP (in-tree or plugin): cudart, cuBLAS, cuFFT, cuRAND, NVRTC, nvJitLink, cuDNN",
    "tensorrt": "TensorRT EP (in-tree or plugin): TensorRT runtime plus cudart / cuBLAS",
    "nv-tensorrt-rtx": "NVIDIA TensorRT RTX EP (in-tree or plugin): TensorRT-RTX runtime plus cudart",
    "qnn": "Qualcomm QNN EP (in-tree or plugin): QAIRT backends and Hexagon skel libraries",
    "vitisai": "AMD Vitis AI EP: closed-source backend DLLs from the voe wheel (Windows x64)",
    "cann": "Huawei CANN EP: installs the CANN toolkit (its libraries need the full install and set_env.sh)",
    "openvino": "Intel OpenVINO EP: development files only (OpenVINO headers and import libraries)",
}
# (recipe, variant) in BUILD_INFO.json -> EPs whose closed-source runtime files the package does not bundle
# (the QNN and TensorRT RTX packages already contain theirs)
PACKAGE_EPS = {("ort", "cuda-plugin"): ["cuda"], ("ep-tensorrt", "default"): ["tensorrt"]}


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
                     dev: list[tuple[re.Pattern, Path]] = ()) -> list[str]:
    """Copy matching files from archive: runtime files into dest (flattened), development files to the targets of
    their dev rules (keeping their directory tree), license files into licenses/. Returns the copied names."""
    copied: list[str] = []
    name = archive.name.lower()
    selected = lambda m: LICENSE_FILE.search(m) or _wanted(m, patterns) or _dev_target(m, dev)

    def store(member: str, data: bytes) -> None:
        if LICENSE_FILE.search(member):
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

    if name.endswith((".zip", ".whl")):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if not info.is_dir() and selected(info.filename):
                    store(info.filename, zf.read(info))
    elif name.endswith(".zst"):  # Python's tarfile cannot read zstd before 3.14; use the system tar
        listing = subprocess.run(["tar", "--zstd", "-tf", str(archive)], check=True, capture_output=True, text=True).stdout
        members = [m for m in listing.splitlines() if not m.endswith("/") and selected(m)]
        with tempfile.TemporaryDirectory() as tmp:
            if members:
                subprocess.run(["tar", "--zstd", "-xf", str(archive), "-C", tmp, *members], check=True)
            for m in members:
                src = Path(tmp) / m
                if src.is_symlink() and os.name != "nt" and _wanted(m, patterns):
                    _symlink(dest, src.name, os.readlink(src), copied)
                elif src.is_file():
                    store(m, src.read_bytes())
    else:
        with tarfile.open(archive, "r|*") as tf:  # streaming: no random access into multi-GB tarballs
            for member in tf:
                if not selected(member.name):
                    continue
                if member.issym() and os.name != "nt" and _wanted(member.name, patterns):
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
                                              list(sel.dev))
            continue
        cur.install = cur.install or sel.install
        cur.patterns += [p for p in sel.patterns if p not in cur.patterns]
        cur.dev += [r for r in sel.dev if r not in cur.dev]
        cur.archives = cur.archives + [a for a in sel.archives if a not in cur.archives] \
            if cur.archives and sel.archives else []
    return merged


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Download closed-source EP runtimes and extract only the runtime files")
    ap.add_argument("--list", action="store_true", help="list the supported EPs")
    ap.add_argument("--package-dir", type=Path, help="extracted package or bundle; EPs, target and versions come "
                                                     "from its BUILD_INFO.json / BUNDLE_INFO.json")
    ap.add_argument("--ep", nargs="+", metavar="EP[:PARTS]",
                    help=f"EPs to fetch for ({', '.join(EPS)}), optionally with the parts: runtime (default), dev, "
                         "runtime+dev")
    ap.add_argument("--target", choices=sorted(TARGETS), help="target triple (default: from BUILD_INFO.json or this host)")
    ap.add_argument("--cuda-major", choices=("12", "13"), help="CUDA major version (default: from BUILD_INFO.json, else 13)")
    ap.add_argument("--pin", nargs="+", default=[], metavar="NAME=VER", help="override SDK versions, same syntax as download_ep_packages.py")
    ap.add_argument("--dest", type=Path, help="output directory of the runtime files (default: the package's bin/ on "
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
    if args.package_dir:
        bundle_file, info_file = args.package_dir / "BUNDLE_INFO.json", args.package_dir / "BUILD_INFO.json"
        if bundle_file.exists():  # a bundle: ci/bundle.py records what it still needs
            info = json.loads(bundle_file.read_text(encoding="utf-8"))["fetch_restricted"]
        elif info_file.exists():
            info = json.loads(info_file.read_text(encoding="utf-8"))
        else:
            raise SystemExit(f"neither {bundle_file} nor {info_file} found")
    target = args.target or info.get("target_triple") or next(
        (t for t, v in TARGETS.items() if v[0] == dl.detect_platform()), None)
    if target not in TARGETS:
        raise SystemExit(f"target {target} has no closed-source runtime to fetch (supported: {', '.join(TARGETS)})")
    platform, os_name, qnn_abi = TARGETS[target]

    if args.ep or "eps" in info:
        eps = parse_eps(args.ep or info["eps"])
    else:
        recipe, _, variant = info.get("recipe", "/").partition("/")
        names = PACKAGE_EPS.get((recipe, variant or "default"), [])
        if recipe == "ort" and info.get("eps"):  # a core: the in-tree EPs it was built with (BUILD_INFO "eps")
            names = [ep for ep in ("cuda", "tensorrt", "vitisai", "cann") if ep in info["eps"].split(",")]
        eps = parse_eps(names)
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
                               (selections_for(ep, os_name, qnn_abi) if "runtime" in parts else []) +
                               (dev_selections_for(ep, platform, os_name, cuda_major) if "dev" in parts else [])])
    labels = ", ".join(f"{ep}:{'+'.join(sorted(parts))}" for ep, parts in eps.items())
    print(f"target {target}, EPs: {labels}, CUDA {cuda_major}.x, runtime files: {dest}, development files: {root}")
    ctx = dl.Context(platform, cuda_major, "ubuntu24", pins)
    report_file = licenses.parent / "THIRD_PARTY_RUNTIME.json"
    report = json.loads(report_file.read_text(encoding="utf-8")) if report_file.exists() else {}
    report["target_triple"] = target
    report["eps"] = sorted(set(report.get("eps", [])) | {f"{ep}:{p}" for ep, parts in eps.items() for p in parts})
    components = report.setdefault("components", {})
    for comp_name, sel in merged.items():
        comp = dl.component(comp_name)
        print(f"==> {comp_name}")
        file_filter = "^(" + "|".join(re.escape(a) for a in sel.archives) + ")" if sel.archives else None
        out, resolved = dl.fetch(comp, ctx, args.cache.resolve(), dry_run=args.dry_run, file_filter=file_filter)
        files = resolved.files
        if args.dry_run:
            for p in sel.patterns or ([] if sel.dev else ["(vendor installer)"]):
                print(f"    extract: {p}")
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
            dev = [(re.compile(r.pattern), root / r.dest) for r in sel.dev]
            new: list[str] = []
            for f in files:
                new += extract_selected(out / f.filename, patterns, dest, licenses / comp_name, dev)
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
