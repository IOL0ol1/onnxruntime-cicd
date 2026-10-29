#!/usr/bin/env python3
"""Download external SDKs and dependencies that EP builds need but cannot fetch automatically at build time.

Scope (obsolete / deprecated EPs are ignored):
  - Closed-source SDKs: CUDA, cuDNN, TensorRT, TensorRT-RTX, QAIRT (QNN), DirectML, CANN, Vitis AI runtime
  - Open-source SDKs that must be installed beforehand: OpenVINO, ROCm, Android NDK,
    ONNX Runtime prebuilt packages (used by plugin EP builds)
  - Open-source dependencies whose sources must be prepared before building (shallow git clone):
    TIM-VX

Every source is an official direct link or official mirror that requires no account (NVIDIA redist /
developer.download.nvidia.com, Qualcomm softwarecenter public API, api.nuget.org, Huawei Cloud Ascend artifactory,
repo.amd.com, pypi.amd.com, storage.openvinotoolkit.org, dl.google.com, sdk.lunarg.com, GitHub Releases).

Only the Python standard library is required (git components also need git). Examples:

    python scripts/download_ep_packages.py --list
    python scripts/download_ep_packages.py --dry-run                         # all components for this platform
    python scripts/download_ep_packages.py --only cuda cudnn tensorrt --platform linux-x86_64
    python scripts/download_ep_packages.py --only openvino --pin openvino=latest
    python scripts/download_ep_packages.py --only vitisai --pin voe=1.6.1 ryzenai-dynamic-dispatch=1.6.1

Output: packages/<component>/<version>/ with a manifest.json; git components are cloned to
packages/<component>/<ref>/src.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import platform as host_platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = ROOT / "packages"
USER_AGENT = "onnxruntime-cicd-downloader/1.0"
PLATFORMS = ("linux-x86_64", "linux-aarch64", "windows-x86_64", "windows-arm64", "windows-arm64x", "macos-arm64")
LATEST = "latest"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FileItem:
    url: str
    filename: str
    sha256: str | None = None
    sha1: str | None = None


@dataclass
class Resolved:
    version: str
    files: list[FileItem] = field(default_factory=list)
    git: tuple[str, str] | None = None  # (repository URL, ref)
    note: str = ""


@dataclass
class Context:
    platform: str
    cuda_major: str
    openvino_distro: str
    pins: dict[str, str]


@dataclass(frozen=True)
class Component:
    name: str
    eps: str
    platforms: tuple[str, ...]
    default: str
    resolve: Callable[[Context, str], Resolved]
    source: str
    optional: bool = False


class UnsupportedPlatform(Exception):
    pass


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _request(url: str, headers: dict[str, str] | None = None, method: str = "GET") -> urllib.request.Request:
    h = {"User-Agent": USER_AGENT}
    if url.startswith("https://api.github.com/") and os.environ.get("GITHUB_TOKEN"):
        h["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    h.update(headers or {})
    return urllib.request.Request(url, headers=h, method=method)


def http_text(url: str, retries: int = 3) -> str:
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(_request(url), timeout=120) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError:
            raise
        except OSError:
            if attempt == retries:
                raise
            print(f"    request timed out or was interrupted, retry {attempt}/{retries - 1}: {url}", flush=True)


def http_json(url: str):
    return json.loads(http_text(url))


def url_exists(url: str) -> bool:
    try:
        with urllib.request.urlopen(_request(url, {"Range": "bytes=0-0"}), timeout=60) as resp:
            return resp.status in (200, 206)
    except OSError:
        return False


def http_header(url: str, name: str) -> str | None:
    with urllib.request.urlopen(_request(url, method="HEAD"), timeout=120) as resp:
        return resp.headers.get(name)


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a" and (href := dict(attrs).get("href")):
            self.links.append(href)


def html_links(url: str) -> list[str]:
    parser = _LinkParser()
    parser.feed(http_text(url))
    return parser.links


def version_key(version: str) -> tuple:
    nums = tuple(int(x) for x in re.findall(r"\d+", version.split("+")[0]))
    return nums, not is_prerelease(version)


def is_prerelease(version: str) -> bool:
    return bool(re.search(r"(dev|alpha|beta|rc|a\d|b\d)", version.split("+")[0], re.I))


def latest(versions: list[str]) -> str:
    stable = [v for v in versions if not is_prerelease(v)]
    if not (stable or versions):
        raise RuntimeError("no usable version found")
    return max(stable or versions, key=version_key)


def github_tags(repo: str) -> list[str]:
    return [t["name"] for t in http_json(f"https://api.github.com/repos/{repo}/tags?per_page=100")]


def read_submodule(rel: str) -> str:
    """Read a file of a submodule; when the submodule is not checked out (e.g. the CI resolve job), fetch the file from
    GitHub at the commit the superproject records, so the result is the same. A copy of this script outside the
    repository (the tools/ of a bundle, see ci/bundle.py) reads the snapshots in submodule-files/ next to it."""
    for path in (ROOT / rel, Path(__file__).with_name("submodule-files") / rel):
        if path.exists():
            return path.read_text(encoding="utf-8")
    path = ROOT / rel
    try:
        names = subprocess.run(["git", "config", "--file", str(ROOT / ".gitmodules"), "--get-regexp", r"\.path$"],
                               capture_output=True, text=True, cwd=ROOT, check=True).stdout.split()
        sub = next(p for p in names[1::2] if rel.startswith(p + "/"))
        url = subprocess.run(["git", "config", "--file", str(ROOT / ".gitmodules"), f"submodule.{sub}.url"],
                             capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()
        sha = subprocess.run(["git", "rev-parse", f"HEAD:{sub}"], capture_output=True, text=True, cwd=ROOT,
                             check=True).stdout.strip()
        repo = re.match(r"https://github\.com/(.+?)(\.git)?$", url).group(1)
        raw = f"https://raw.githubusercontent.com/{repo}/{sha}/{rel[len(sub) + 1:]}"
        return urllib.request.urlopen(raw, timeout=60).read().decode("utf-8")
    except (subprocess.CalledProcessError, StopIteration, AttributeError, OSError) as e:
        raise RuntimeError(f"{rel} is missing (run git submodule update --init) and could not be fetched: {e}")


def yaml_block(text: str, key: str) -> dict[str, str]:
    """Read the scalar fields under `key` from a simple two-level YAML file (avoids depending on PyYAML)."""
    m = re.search(rf"^{re.escape(key)}:\s*(?:&\S+)?\s*\n((?:[ \t]+.*\n?)*)", text, re.M)
    if not m:
        raise RuntimeError(f"{key} not found in packages.yml")
    fields = dict(re.findall(r"^\s+(\w+):\s*(.+?)\s*$", m.group(1), re.M))
    return {k: v.replace("{version}", fields.get("version", "")) for k, v in fields.items()}


# ---------------------------------------------------------------------------
# Component resolvers
# ---------------------------------------------------------------------------

NV_REDIST = "https://developer.download.nvidia.com/compute"
NV_PLATFORM = {"linux-x86_64": "linux-x86_64", "linux-aarch64": "linux-sbsa",
               "windows-x86_64": "windows-x86_64", "windows-arm64": "windows-arm64"}
# CUDA components needed by the ORT CUDA / TensorRT EP builds (cuda_cccl on 12.x, cccl on 13.x; missing ones are skipped)
CUDA_COMPONENTS = ("cccl", "cuda_cccl", "cuda_crt", "cuda_cudart", "cuda_cupti", "cuda_nvcc", "cuda_nvml_dev",
                   "cuda_nvrtc", "cuda_nvtx", "cuda_profiler_api", "libcublas", "libcufft", "libcurand", "libnvjitlink",
                   "libnvvm")


def _nv_redist_version(product: str, version: str, prefix: str) -> str:
    if version != LATEST:
        return version
    names = set(re.findall(r"redistrib_([\d.]+)\.json", http_text(f"{NV_REDIST}/{product}/redist/")))
    return latest([n for n in names if n.startswith(prefix)])


def resolve_cuda(ctx: Context, version: str) -> Resolved:
    plat = NV_PLATFORM.get(ctx.platform) or _unsupported()
    version = _nv_redist_version("cuda", version, f"{ctx.cuda_major}.")
    manifest = http_json(f"{NV_REDIST}/cuda/redist/redistrib_{version}.json")
    files = []
    for comp in CUDA_COMPONENTS:
        entry = manifest.get(comp, {}).get(plat)
        if entry:
            files.append(FileItem(f"{NV_REDIST}/cuda/redist/{entry['relative_path']}",
                                  entry["relative_path"].rsplit("/", 1)[-1], entry.get("sha256")))
    return Resolved(version, files, note="extract all components into one directory and use it as CUDA_HOME")


def resolve_cudnn(ctx: Context, version: str) -> Resolved:
    plat = NV_PLATFORM.get(ctx.platform) or _unsupported()
    version = _nv_redist_version("cudnn", version, "9.")
    entry = http_json(f"{NV_REDIST}/cudnn/redist/redistrib_{version}.json")["cudnn"].get(plat) or _unsupported()
    entry = entry.get(f"cuda{ctx.cuda_major}", entry)
    if "relative_path" not in entry:
        raise RuntimeError(f"cuDNN {version} has no CUDA {ctx.cuda_major} build")
    path = entry["relative_path"]
    return Resolved(version, [FileItem(f"{NV_REDIST}/cudnn/redist/{path}", path.rsplit("/", 1)[-1], entry["sha256"])])


def _match_platform(name: str, plat: str) -> bool:
    n = name.lower()
    if plat.startswith("linux"):
        arch = {"linux-x86_64": "x86_64", "linux-aarch64": "aarch64"}.get(plat)
        return "linux" in n and arch is not None and arch in n
    if plat.startswith("windows"):
        return "win" in n and (("arm64" in n) == (plat == "windows-arm64"))
    return False


def _pick_cuda(urls: list[str], cuda_major: str) -> list[str]:
    picked = [u for u in urls if re.search(rf"cuda-{cuda_major}\.", u)]
    if not picked:
        _unsupported(f"no CUDA {cuda_major} package for this platform; try a different --cuda-major")
    return [max(picked, key=lambda u: version_key(re.search(r"cuda-([\d.]+)", u).group(1)))]


def resolve_tensorrt(ctx: Context, version: str) -> Resolved:
    # The README of each NVIDIA/TensorRT release branch lists direct links to that version's official GA packages
    ref = "main" if version == LATEST else version
    readme = http_text(f"https://raw.githubusercontent.com/NVIDIA/TensorRT/{ref}/README.md")
    urls = sorted(set(re.findall(r"https://developer\.nvidia\.com/downloads/compute/machine-learning/tensorrt/[^\s)\"']+", readme)))
    matched = [u for u in urls if _match_platform(u.rsplit("/", 1)[-1], ctx.platform)]
    note = f"from the NVIDIA/TensorRT@{ref} README"
    if not matched and ctx.platform == "linux-aarch64":
        # The README only lists x86_64; aarch64 packages live next to them with matching names (ORT CI relies on this)
        derived = [re.sub(r"(Linux[.-])x86_64", r"\g<1>aarch64", u) for u in urls if _match_platform(u.rsplit("/", 1)[-1], "linux-x86_64")]
        matched = [u for u in derived if url_exists(u)]
        note += " (aarch64 link derived from the official naming scheme and verified to exist)"
    url = _pick_cuda(matched or _unsupported(f"TensorRT@{ref} has no package for {ctx.platform}"), ctx.cuda_major)[0]
    if not url_exists(url):
        # Some README links are stale: Windows zips were renamed from ".Windows.win10." to ".Windows.amd64."
        alternatives = [url.replace(".Windows.win10.", ".Windows.amd64.")]
        url = next((u for u in alternatives if u != url and url_exists(u)), None) or _raise(f"TensorRT link is dead: {url}")
        note += " (README link was stale; using the official .Windows.amd64. file name)"
    name = url.rsplit("/", 1)[-1]
    return Resolved(re.search(r"TensorRT-(?:Enterprise-)?([\d.]+?)[.-](?:Linux|Windows)", name).group(1),
                    [FileItem(url, name)], note=note)


def resolve_tensorrt_rtx(ctx: Context, version: str) -> Resolved:
    # The official TensorRT-RTX page links every SDK version directly (the path contains "secure" but needs no login)
    page = http_text("https://developer.nvidia.com/tensorrt-rtx")
    urls = sorted(set(re.findall(r"https://developer\.nvidia\.com/downloads/trt/rtx_sdk/secure/[^\s\"'<>]+", page)))
    series = sorted({u.split("/secure/")[1].split("/")[0] for u in urls}, key=version_key)
    chosen = series[-1] if version == LATEST else version
    if chosen not in series:
        raise RuntimeError(f"{chosen} is not listed on the official page; available: {series}")
    supported = [v for v in series if any(f"/secure/{v}/" in u and _match_platform(u.rsplit("/", 1)[-1], ctx.platform) for u in urls)]
    urls = [u for u in urls if f"/secure/{chosen}/" in u and _match_platform(u.rsplit("/", 1)[-1], ctx.platform)]
    if not urls:
        _unsupported(f"TensorRT-RTX {chosen} has no package for {ctx.platform}; versions supporting it: "
                     f"{supported or 'none'}; use --pin tensorrt-rtx=<version>")
    url = _pick_cuda(urls, ctx.cuda_major)[0]
    name = url.rsplit("/", 1)[-1]
    return Resolved(re.search(r"TensorRT-RTX-([\d.]+)", name, re.I).group(1), [FileItem(url, name)])


def _qnn_package(key: str, version: str) -> Resolved:
    pkg = yaml_block(read_submodule("eps/official/onnxruntime-qnn/qcom/packages.yml"), key)
    if version in ("default", pkg["version"]):
        version, url, sha = pkg["version"], pkg["url"], pkg.get("sha256")
    else:  # other versions reuse the same URL template, but no checksum is known
        url, sha = pkg["url"].replace(pkg["version"], version), None
    return Resolved(version, [FileItem(url, url.rsplit("/", 1)[-1], sha)],
                    note="default version and checksum come from onnxruntime-qnn/qcom/packages.yml")


def resolve_qairt(ctx: Context, version: str) -> Resolved:
    return _qnn_package("qairt", version)


def resolve_hexagon(ctx: Context, version: str) -> Resolved:
    return _qnn_package("hexagon_linux_x86_64", version)


OV_BASE = "https://storage.openvinotoolkit.org"


def resolve_openvino(ctx: Context, version: str) -> Resolved:
    tree = http_json(f"{OV_BASE}/filetree.json")

    def child(node: dict, name: str) -> dict:
        for c in node.get("children", []):
            if c["name"] == name:
                return c
        raise RuntimeError(f"{name} not found in OpenVINO storage")

    packages = child(child(child(tree, "repositories"), "openvino"), "packages")
    if version == LATEST:
        version = latest([c["name"] for c in packages["children"] if re.fullmatch(r"\d{4}\.\d+(\.\d+)?", c["name"])])
    plat_dir, pattern = {
        "linux-x86_64": ("linux", r"openvino_toolkit_(\w+?)_[\d.]+\.\w+_x86_64\.tgz$"),
        "linux-aarch64": ("linux", r"openvino_toolkit_(\w+?)_[\d.]+\.\w+_arm64\.tgz$"),
        "windows-x86_64": ("windows", r"openvino_toolkit_(windows)_.*_x86_64\.zip$"),
        "macos-arm64": ("macos", r"openvino_toolkit_(macos)_.*_arm64\.tgz$"),
    }.get(ctx.platform) or _unsupported()
    names = [c["name"] for c in child(child(packages, version), plat_dir).get("children", [])]
    archives = {m.group(1): n for n in sorted(names) if (m := re.match(pattern, n))}
    if not archives:
        _unsupported(f"OpenVINO {version} has no package for {ctx.platform}")
    distro = ctx.openvino_distro if ctx.openvino_distro in archives else sorted(archives)[-1]
    if plat_dir == "linux" and distro != ctx.openvino_distro:
        print(f"    note: no {ctx.openvino_distro} package, using {distro} instead (available: {', '.join(sorted(archives))})")
    name = archives[distro]
    url = f"{OV_BASE}/repositories/openvino/packages/{version}/{plat_dir}/{name}"
    sha = http_text(url + ".sha256").split()[0] if f"{name}.sha256" in names else None
    return Resolved(version, [FileItem(url, name, sha)])


CANN_BASE = "https://ascend.devcloud.huaweicloud.com/artifactory/cann-run/software"


def resolve_cann(ctx: Context, version: str) -> Resolved:
    arch = {"linux-x86_64": "x86_64", "linux-aarch64": "aarch64"}.get(ctx.platform) or _unsupported()
    if version == LATEST:
        version = latest([h.strip("/") for h in html_links(f"{CANN_BASE}/") if re.match(r"\d+\.\d+", h)])
    name = f"Ascend-cann-toolkit_{version}_linux-{arch}.run"
    url = f"{CANN_BASE}/{version}/{arch}/{name}"
    return Resolved(version, [FileItem(url, name, http_header(url, "X-Checksum-Sha256"))],
                    note="toolkit only; per-chip operator packages (*-ops_*.run) are in the same directory if needed")


ROCM_BASE = "https://repo.amd.com/rocm/tarball-multi-arch"


def resolve_rocm(ctx: Context, version: str) -> Resolved:
    os_name = {"linux-x86_64": "linux", "windows-x86_64": "windows"}.get(ctx.platform) or _unsupported()
    names = re.findall(r'"name":\s*"([^"]+)"', http_text(f"{ROCM_BASE}/"))
    versions = [m.group(1) for n in names if (m := re.fullmatch(rf"therock-dist-{os_name}-multiarch-([\d.]+\w*)\.tar\.gz", n))]
    if version == LATEST:
        version = latest(versions)
    name = f"therock-dist-{os_name}-multiarch-{version}.tar.gz"
    if name not in names:
        raise RuntimeError(f"{name} not found; available versions: {sorted(set(versions), key=version_key)}")
    return Resolved(version, [FileItem(f"{ROCM_BASE}/{name}", name)],
                    note="TheRock multi-arch ROCm SDK; extract and use it as ROCM_PATH")


ROCM_WHEELS = "https://repo.amd.com/rocm/whl-multi-arch/"


def resolve_rocm_wheels(ctx: Context, version: str) -> Resolved:
    """TheRock's ROCm runtime as Python wheels (Windows): rocm-sdk-core, rocm-sdk-libraries and one
    rocm-sdk-device-<gfx> per GPU target with the device libraries of that target; the caller picks the device wheels
    it needs (fetch-restricted: the GPU targets of the bundle)."""
    if ctx.platform != "windows-x86_64":
        _unsupported()
    devices = sorted(set(re.findall(r'href="(rocm-sdk-device-gfx[0-9a-z]+)/"', http_text(ROCM_WHEELS))))
    files = []
    for project in ["rocm-sdk-core", "rocm-sdk-libraries", *devices]:
        page = urllib.parse.urljoin(ROCM_WHEELS, f"{project}/")
        for href in html_links(page):
            url, _, frag = urllib.parse.urljoin(page, href).partition("#")
            name = urllib.parse.unquote(url.rsplit("/", 1)[-1])
            if name.endswith("-py3-none-win_amd64.whl"):
                files.append(FileItem(url, name, frag[7:] if frag.startswith("sha256=") else None))
    if version == LATEST:
        version = latest([f.filename.split("-")[1] for f in files if f.filename.startswith("rocm_sdk_core-")])
    files = [f for f in files if f.filename.split("-")[1] == version]
    if not any(f.filename.startswith("rocm_sdk_core-") for f in files):
        _unsupported(f"no ROCm {version} wheels for Windows")
    return Resolved(version, files, note="unpack the wheels into one directory, keeping their layout (as pip does)")


def resolve_android_ndk(ctx: Context, version: str) -> Resolved:
    host = {"linux-x86_64": "linux", "windows-x86_64": "windows", "macos-arm64": "macosx"}.get(ctx.platform) or _unsupported()
    xml = http_text("https://dl.google.com/android/repository/repository2-3.xml")
    channels = dict(re.findall(r'<channel id="([^"]+)">([^<]+)</channel>', xml))
    packages = {}
    for m in re.finditer(r'<remotePackage path="ndk;([^"]+)">(.*?)</remotePackage>', xml, re.S):
        ref = re.search(r'channelRef ref="([^"]+)"', m.group(2))
        if ref and channels.get(ref.group(1)) == "stable":
            packages[m.group(1)] = m.group(2)
    if version == LATEST:
        version = max(packages, key=version_key)
    body = packages.get(version) or _raise(f"no stable NDK {version}")
    for archive in re.findall(r"<archive>(.*?)</archive>", body, re.S):
        if re.search(rf"<host-os>{host}</host-os>", archive) or "<host-os>" not in archive:
            name = re.search(r"<url>([^<]+)</url>", archive).group(1)
            sha1 = re.search(r"<checksum[^>]*>([0-9a-f]+)</checksum>", archive).group(1)
            return Resolved(version, [FileItem(f"https://dl.google.com/android/repository/{name}", name, sha1=sha1)])
    _unsupported()


def resolve_onnxruntime_sdk(ctx: Context, version: str) -> Resolved:
    tag = "latest" if version == LATEST else f"tags/v{version}"
    release = http_json(f"https://api.github.com/repos/microsoft/onnxruntime/releases/{tag}")
    version = release["tag_name"].lstrip("v")
    plat = {"linux-x86_64": "linux-x64", "linux-aarch64": "linux-aarch64", "windows-x86_64": "win-x64",
            "windows-arm64": "win-arm64", "windows-arm64x": "win-arm64x",
            "macos-arm64": "osx-arm64"}.get(ctx.platform) or _unsupported()
    for asset in release["assets"]:
        if re.fullmatch(rf"onnxruntime-{plat}-{re.escape(version)}\.(tgz|zip)", asset["name"]):
            digest = asset.get("digest") or ""
            sha = digest.split(":", 1)[1] if digest.startswith("sha256:") else None
            return Resolved(version, [FileItem(asset["browser_download_url"], asset["name"], sha)],
                            note="ORT headers and libraries (ORT_HOME) for plugin EP builds")
    _unsupported()


NEUROPILOT_REPO = "onnxruntime/onnxruntime-ep-neuropilot"


def resolve_neuron_op_capability(ctx: Context, version: str) -> Resolved:
    """The prebuilt libneuron_op_capability.so of the NeuroPilot plugin EP: the release assets of its repository, one
    archive (under 1 MB) per target, aarch64 Linux and Android arm64-v8a; the build and fetch-restricted pick theirs."""
    if version == "default":  # the release of the plugin EP version the submodule builds (its op capability ABI)
        cmake = read_submodule("eps/official/onnxruntime-ep-neuropilot/CMakeLists.txt")
        version = re.search(r"project\(ORTNeuronEp VERSION ([\d.]+)", cmake).group(1)
    try:
        release = http_json(f"https://api.github.com/repos/{NEUROPILOT_REPO}/releases/"
                            + ("latest" if version == LATEST else f"tags/v{version}"))
    except urllib.error.HTTPError:  # the submodule is ahead of the newest release
        release = http_json(f"https://api.github.com/repos/{NEUROPILOT_REPO}/releases/latest")
    files = [FileItem(a["browser_download_url"], a["name"], (a.get("digest") or "").removeprefix("sha256:") or None)
             for a in release["assets"] if a["name"].startswith("libneuron-op-capability-")]
    if not files:
        _unsupported(f"{NEUROPILOT_REPO} {release['tag_name']} has no libneuron-op-capability archives")
    return Resolved(release["tag_name"].lstrip("v"), files,
                    note="closed source; the archives carry no license terms")


HIP_EP_REPO = "ROCm/hip-ep"


def resolve_hip_ep(ctx: Context, version: str) -> Resolved:
    """The prebuilt HIP EP of ROCm/hip-ep (hipgpu, the HIP backend of the AMDGPU plugin EP) with the ROCm runtime it is
    built with: the release archive of the platform, and the repository's license (the archive has none)."""
    os_name = {"windows-x86_64": "windows", "linux-x86_64": "linux"}.get(ctx.platform) or _unsupported()
    release = http_json(f"https://api.github.com/repos/{HIP_EP_REPO}/releases/"
                        + ("latest" if version == LATEST else f"tags/v{version}"))
    version = release["tag_name"].lstrip("v")
    asset = next((a for a in release["assets"] if a["name"] == f"hipep_{version}_{os_name}.zip"), None) \
        or _unsupported(f"{HIP_EP_REPO} {release['tag_name']} has no {os_name} package")
    digest = (asset.get("digest") or "").removeprefix("sha256:") or None
    license_url = f"https://raw.githubusercontent.com/{HIP_EP_REPO}/{release['tag_name']}/LICENSE"
    return Resolved(version, [FileItem(asset["browser_download_url"], asset["name"], digest),
                              FileItem(license_url, "LICENSE")],
                    note="hipgpu with its ROCm runtime (Windows); Linux uses the system ROCm")


AMD_INDEX = "https://pypi.amd.com/simple/"
VITISAI_PACKAGES = ("onnxruntime-vitisai", "voe", "ryzenai-dynamic-dispatch")


def resolve_vitisai(ctx: Context, version: str) -> Resolved:
    files, versions = [], []
    for package in VITISAI_PACKAGES:
        page = urllib.parse.urljoin(AMD_INDEX, f"{package}/")
        wheels = []
        for href in html_links(page):
            url, _, frag = urllib.parse.urljoin(page, href).partition("#")
            name = urllib.parse.unquote(url.rsplit("/", 1)[-1])
            if name.endswith(".whl"):
                wheels.append(FileItem(url, name, frag[7:] if frag.startswith("sha256=") else None))
        pinned = ctx.pins.get(package)
        wanted = pinned or latest([w.filename.split("-")[1] for w in wheels])
        picked = [w for w in wheels if w.filename.split("-")[1] == wanted] or _raise(f"{package} has no version {wanted}")
        files += picked
        versions.append(f"{package}-{wanted}")
    return Resolved("_".join(versions), files,
                    note="onnxruntime-vitisai replaces the onnxruntime package from PyPI; install it last")


NUGET = "https://api.nuget.org/v3-flatcontainer"


def resolve_directml(ctx: Context, version: str) -> Resolved:
    if not ctx.platform.startswith("windows"):
        _unsupported()
    if version == LATEST:
        version = latest(http_json(f"{NUGET}/microsoft.ai.directml/index.json")["versions"])
    name = f"microsoft.ai.directml.{version}.nupkg"
    return Resolved(version, [FileItem(f"{NUGET}/microsoft.ai.directml/{version}/{name}", name)],
                    note="NuGet package (a zip): DirectML.dll in bin/<arch>-win/, headers in include/")


def _git_resolver(repo: str, default_ref: Callable[[], str]) -> Callable[[Context, str], Resolved]:
    def resolve(ctx: Context, version: str) -> Resolved:
        ref = default_ref() if version == LATEST else version
        return Resolved(ref, git=(f"https://github.com/{repo}.git", ref))
    return resolve


def _latest_tag(repo: str) -> Callable[[], str]:
    return lambda: latest(github_tags(repo))


def _unsupported(reason: str = ""):
    raise UnsupportedPlatform(reason)


def _raise(msg: str):
    raise RuntimeError(msg)


COMPONENTS: tuple[Component, ...] = (
    Component("cuda", "CUDA, TensorRT, TensorRT plugin, TensorRT RTX", tuple(NV_PLATFORM), LATEST, resolve_cuda,
              "developer.download.nvidia.com/compute/cuda/redist"),
    Component("cudnn", "CUDA", tuple(NV_PLATFORM), LATEST, resolve_cudnn,
              "developer.download.nvidia.com/compute/cudnn/redist"),
    # The default branch matches TensorRT 10.15 used by ORT CI; latest = main branch (TensorRT 11.x)
    Component("tensorrt", "TensorRT (in-tree), onnxruntime-ep-tensorrt", ("linux-x86_64", "linux-aarch64", "windows-x86_64"),
              "release/10.15", resolve_tensorrt, "developer.nvidia.com/downloads (links taken from the NVIDIA/TensorRT README)"),
    # Default 1.7: cmake/tensorrt_rtx.cmake in TensorRT-RTX-EP-ABI requires >= 1.7 at configure time
    Component("tensorrt-rtx", "NVIDIA TensorRT RTX (in-tree + plugin)",
              ("linux-x86_64", "linux-aarch64", "windows-x86_64", "windows-arm64"), "1.7", resolve_tensorrt_rtx,
              "direct links on developer.nvidia.com/tensorrt-rtx"),
    Component("qairt", "Qualcomm QNN (in-tree + onnxruntime-qnn)",
              ("linux-x86_64", "linux-aarch64", "windows-x86_64", "windows-arm64", "windows-arm64x"), "default", resolve_qairt,
              "softwarecenter.qualcomm.com public download API"),
    Component("hexagon-sdk", "Qualcomm QNN (only needed for custom HTP ops)", ("linux-x86_64",), "default", resolve_hexagon,
              "softwarecenter.qualcomm.com public download API", optional=True),
    # The default version matches the CI of the intel/onnxruntime ovep-develop branch
    Component("openvino", "Intel OpenVINO",
              ("linux-x86_64", "linux-aarch64", "windows-x86_64", "macos-arm64"), "2026.2.1", resolve_openvino,
              "storage.openvinotoolkit.org"),
    # The default version matches onnxruntime/packages.config and the AMD plugin EPs
    Component("directml", "DirectML (in-tree + onnxruntime-ep-amdgpu)", ("windows-x86_64", "windows-arm64", "windows-arm64x"),
              "1.15.4", resolve_directml, "api.nuget.org (Microsoft.AI.DirectML)"),
    Component("cann", "Huawei CANN", ("linux-x86_64", "linux-aarch64"), LATEST, resolve_cann,
              "ascend.devcloud.huaweicloud.com artifactory (official Huawei Cloud mirror)"),
    Component("rocm", "AMD MIGraphX / AMDGPU (in-tree + onnxruntime-ep-amdgpu)", ("linux-x86_64", "windows-x86_64"), LATEST,
              resolve_rocm, "repo.amd.com/rocm/tarball-multi-arch (TheRock)"),
    # The version the Windows AMD plugin EPs are built with is recorded as "rocm" in their BUILD_INFO.json
    Component("rocm-wheels", "AMD MIGraphX / AMDGPU plugins on Windows (the ROCm runtime)", ("windows-x86_64",), LATEST,
              resolve_rocm_wheels, "repo.amd.com/rocm/whl-multi-arch (TheRock Python wheels)"),
    # The AMDGPU package with the HIP backend records the hip-ep release it is paired with (BUILD_INFO.json)
    Component("hip-ep", "AMDGPU plugin EP with the HIP backend (hipgpu and its runtime)", ("windows-x86_64", "linux-x86_64"),
              LATEST, resolve_hip_ep, f"github.com/{HIP_EP_REPO}/releases"),
    Component("vitisai", "AMD Vitis AI (Ryzen AI NPU)", ("windows-x86_64",), LATEST, resolve_vitisai,
              "pypi.amd.com/simple (public AMD index)"),
    # The default version matches ORT's android.yml CI
    Component("android-ndk", "NNAPI, QNN (Android), XNNPACK (Android)",
              ("linux-x86_64", "windows-x86_64", "macos-arm64"), "28.0.13004108", resolve_android_ndk,
              "dl.google.com/android/repository"),
    Component("onnxruntime-sdk", "all plugin EPs (ORT headers and libraries)",
              ("linux-x86_64", "linux-aarch64", "windows-x86_64", "windows-arm64", "windows-arm64x", "macos-arm64"), LATEST,
              resolve_onnxruntime_sdk, "github.com/microsoft/onnxruntime/releases"),
    # linux-x86_64: the host platform of the Android builds
    Component("neuron-op-capability", "MediaTek NeuroPilot plugin EP (the op capability library it loads at run time)",
              ("linux-aarch64", "linux-x86_64"), "default", resolve_neuron_op_capability,
              f"github.com/{NEUROPILOT_REPO}/releases"),
    Component("tim-vx", "VeriSilicon VSINPU", ("linux-x86_64", "linux-aarch64"), LATEST,
              _git_resolver("VeriSilicon/TIM-VX", _latest_tag("VeriSilicon/TIM-VX")), "github.com/VeriSilicon/TIM-VX"),
)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _check(path: Path, item: FileItem) -> bool:
    if item.sha256:
        return _digest(path, "sha256") == item.sha256.lower()
    if item.sha1:
        return _digest(path, "sha1") == item.sha1.lower()
    return False


def _digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def _download_part(item: FileItem, part: Path, name: str) -> bool:
    """Download into part, resuming what it already holds; True when the server's whole content has arrived."""
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with urllib.request.urlopen(_request(item.url, headers), timeout=600) as resp:
        if offset and resp.status != 206:
            offset = 0  # the server does not support resuming; start over
        length = int(resp.headers.get("Content-Length") or 0)
        total = length + offset if length else 0
        done, last = offset, -1
        with open(part, "ab" if offset else "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                pct = done * 100 // total if total else -1
                if pct != last and pct % 10 == 0:
                    print(f"    {name}: {pct}% ({done >> 20} MiB)", flush=True)
                    last = pct
    if total and done < total:
        print(f"    {name}: connection closed at {done >> 20} of {total >> 20} MiB", flush=True)
        return False
    return True


def download_file(item: FileItem, out: Path) -> str:
    dest = out / item.filename
    if dest.exists() and (_check(dest, item) or not (item.sha256 or item.sha1)):
        print(f"    already present, skipping: {dest.name}")
        return _digest(dest, "sha256")
    part = dest.with_name(dest.name + ".part")
    # Large downloads (QAIRT is ~3 GB) sometimes end early without an error; resume them, and start over once if the
    # completed file does not match its checksum
    for attempt in range(1, 7):
        try:
            if _download_part(item, part, dest.name):
                if not (item.sha256 or item.sha1) or _check(part, item):
                    break
                part.unlink()
                print(f"    {dest.name}: checksum mismatch, downloading again", flush=True)
        except (OSError, http.client.HTTPException) as e:  # URLError, timeouts, IncompleteRead, reset connections
            print(f"    {dest.name}: {e.__class__.__name__}: {e}; resuming", flush=True)
        if attempt == 6:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"download failed or checksum mismatch: {dest.name}")
        time.sleep(10 * attempt)
    part.replace(dest)
    print(f"    downloaded: {dest.name}{' (verified)' if item.sha256 or item.sha1 else ' (no upstream checksum)'}")
    return _digest(dest, "sha256")


def git_fetch(repo: str, ref: str, out: Path) -> None:
    src = out / "src"
    if (src / ".git").exists():
        print(f"    already present, skipping: {src}")
        return
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", "--branch", ref, repo, str(src)], check=True)


def fetch(comp: Component, ctx: Context, dest: Path, dry_run: bool = False,
          file_filter: str | None = None) -> tuple[Path, Resolved]:
    """Resolve and download one component; returns (download directory, resolution).
    Reused by ci/sdk.py and scripts/fetch_ep_runtime.py; file_filter (regex) limits which files are downloaded."""
    version = ctx.pins.get(comp.name, comp.default)
    resolved = comp.resolve(ctx, version)
    if file_filter:
        resolved.files = [f for f in resolved.files if re.search(file_filter, f.filename)]
    out = dest / comp.name / resolved.version.replace("/", "_")
    if resolved.git:
        repo, ref = resolved.git
        print(f"  git {repo} @ {ref} -> {out / 'src'}")
        if not dry_run:
            git_fetch(repo, ref, out)
        return out, resolved
    for item in resolved.files:
        check = "sha256" if item.sha256 else "sha1" if item.sha1 else "no checksum"
        print(f"  - {item.filename}  [{check}]")
    if resolved.note:
        print(f"    note: {resolved.note}")
    if dry_run:
        return out, resolved
    out.mkdir(parents=True, exist_ok=True)
    manifest = [{"file": i.filename, "url": i.url, "sha256": download_file(i, out),
                 "upstream_checksum": i.sha256 or i.sha1} for i in resolved.files]
    info = {"component": comp.name, "eps": comp.eps, "platform": ctx.platform, "version": resolved.version,
            "source": comp.source, "note": resolved.note, "files": manifest}
    (out / "manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return out, resolved


def component(name: str) -> Component:
    return next(c for c in COMPONENTS if c.name == name)


def detect_platform() -> str:
    machine = host_platform.machine().lower()
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}.get(machine, machine)
    if sys.platform.startswith("win"):
        return "windows-arm64" if arch == "aarch64" else "windows-x86_64"
    if sys.platform == "darwin":
        return "macos-arm64"
    return f"linux-{arch}"


def parse_args(argv: list[str]) -> argparse.Namespace:
    names = [c.name for c in COMPONENTS]
    ap = argparse.ArgumentParser(description="Download external SDKs and dependencies that EP builds cannot fetch automatically")
    ap.add_argument("--list", action="store_true", help="list all components")
    ap.add_argument("--only", nargs="+", choices=names, metavar="NAME", help="only process the given components")
    ap.add_argument("--with-optional", action="store_true", help="also process optional components (e.g. hexagon-sdk)")
    ap.add_argument("--platform", choices=PLATFORMS, default=detect_platform(), help="target platform (default: this host)")
    ap.add_argument("--cuda-major", default="13", choices=("12", "13"),
                    help="CUDA major version; affects cuda/cudnn/tensorrt/tensorrt-rtx (default: 13)")
    ap.add_argument("--openvino-distro", default="ubuntu24", help="distro tag of the Linux OpenVINO package (default: ubuntu24)")
    ap.add_argument("--pin", nargs="+", default=[], metavar="NAME=VER",
                    help="override versions: component=version or latest; vitisai packages can be pinned by package name, e.g. voe=1.6.1")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST, help=f"download directory (default: {DEFAULT_DEST})")
    ap.add_argument("--dry-run", action="store_true", help="only resolve and show what would be downloaded")
    args = ap.parse_args(argv)
    try:
        args.pin = dict(p.split("=", 1) for p in args.pin)
    except ValueError:
        ap.error("--pin expects NAME=VER")
    return args


def main(argv: list[str]) -> int:
    # Windows uses the local code page when output is piped; force UTF-8 so non-ASCII paths print correctly
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    if args.list:
        for c in COMPONENTS:
            flag = " (optional)" if c.optional else ""
            print(f"{c.name:20} {c.eps}{flag}")
            print(f"{'':20}   default version: {c.default}    platforms: {', '.join(c.platforms)}")
            print(f"{'':20}   source         : {c.source}")
        return 0

    ctx = Context(args.platform, args.cuda_major, args.openvino_distro, args.pin)
    selected = [c for c in COMPONENTS if (c.name in args.only if args.only else (args.with_optional or not c.optional))]
    print(f"target platform: {ctx.platform}, CUDA {ctx.cuda_major}.x")
    failures = 0
    for comp in selected:
        print(f"==> {comp.name}: {comp.eps}")
        if ctx.platform not in comp.platforms:
            print(f"    skipped: {ctx.platform} is not supported")
            continue
        try:
            fetch(comp, ctx, args.dest, args.dry_run)
        except UnsupportedPlatform as e:
            print(f"    skipped: {e or f'upstream has no package for {ctx.platform}'}")
        except (OSError, RuntimeError, subprocess.CalledProcessError, KeyError, AttributeError) as e:
            failures += 1
            print(f"  !! {comp.name} failed: {e}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
