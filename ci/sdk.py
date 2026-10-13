#!/usr/bin/env python3
"""CI helper: download and extract the third-party SDKs a build needs, and export environment variables.

Reuses the download logic of scripts/download_ep_packages.py. Each component is extracted into
<root>/<component>/ (without the archive's top-level directory), then variables such as CUDA_HOME and
TENSORRT_HOME are written to $GITHUB_ENV (and printed when running locally).

    python ci/sdk.py --root $RUNNER_TEMP/sdk --platform linux-x86_64 --cuda-major 12 cuda cudnn tensorrt

Resolved versions are written to <root>/sdk-versions.json, which ci/package.py records in BUILD_INFO.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("dl", ROOT / "scripts" / "download_ep_packages.py")
dl = importlib.util.module_from_spec(_spec)
sys.modules["dl"] = dl
_spec.loader.exec_module(dl)

# component -> exported environment variables (values are paths relative to the component directory)
ENV_VARS: dict[str, dict[str, str]] = {
    "cuda": {"CUDA_HOME": "", "CUDA_PATH": ""},
    "cudnn": {"CUDNN_HOME": ""},
    "tensorrt": {"TENSORRT_HOME": ""},
    "tensorrt-rtx": {"TENSORRT_RTX_HOME": ""},
    "qairt": {"QNN_HOME": "", "QNN_SDK_ROOT": ""},
    "hexagon-sdk": {"HEXAGON_SDK_ROOT": ""},
    "openvino": {"INTEL_OPENVINO_DIR": "", "OpenVINO_DIR": "runtime/cmake"},
    "cann": {"CANN_HOME": ""},
    "rocm": {"ROCM_PATH": "", "HIP_PATH": ""},
    "amdmigraphx": {"AMDMIGRAPHX_SRC": ""},
    "android-ndk": {"ANDROID_NDK_HOME": "", "ANDROID_NDK_ROOT": ""},
    "onnxruntime-sdk": {"ORT_HOME": ""},
    "vitisai": {"VITISAI_WHEELS": ""},
    "vulkan-sdk": {"VULKAN_SDK": ""},
    "tim-vx": {"TIM_VX_SRC": ""},
    "zdnn": {"ZDNN_SRC": ""},
    "compass-npu-driver": {"COMPASS_NPU_DRIVER_SRC": ""},
}
# directories to add to PATH
PATH_DIRS = {"cuda": "bin", "rocm": "bin", "vulkan-sdk": "bin"}


def _strip_single_top(tmp: Path, target: Path) -> None:
    entries = list(tmp.iterdir())
    src = entries[0] if len(entries) == 1 and entries[0].is_dir() else tmp
    target.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        dest = target / item.name
        if dest.is_dir() and item.is_dir():  # CUDA components must be merged into one directory tree
            shutil.copytree(item, dest, dirs_exist_ok=True, symlinks=True)
            shutil.rmtree(item)
        else:
            shutil.move(str(item), str(dest))


def extract(archive: Path, target: Path) -> None:
    """Extract into target (dropping a single top-level directory). Linux/macOS use system tools to keep symlinks and exec bits."""
    name = archive.name.lower()
    with tempfile.TemporaryDirectory(dir=target.parent) as t:
        tmp = Path(t)
        if name.endswith((".zip", ".whl")):
            if os.name == "nt":
                zipfile.ZipFile(archive).extractall(tmp)
            else:
                subprocess.run(["unzip", "-q", str(archive), "-d", str(tmp)], check=True)
        elif ".tar" in name or name.endswith(".tgz"):
            if os.name == "nt" and not name.endswith(".zst"):
                with tarfile.open(archive) as tf:
                    tf.extractall(tmp, filter="tar")
            else:
                flags = ["--zstd"] if name.endswith(".zst") else []
                subprocess.run(["tar", *flags, "-xf", str(archive), "-C", str(tmp)], check=True)
        else:
            raise RuntimeError(f"unknown archive format: {archive.name}")
        _strip_single_top(tmp, target)


def install_cann(run_file: Path, target: Path) -> None:
    run_file.chmod(0o755)
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(run_file), "--install", "--quiet", f"--install-path={target}"], check=True)


def install_vulkan_sdk(installer: Path, target: Path) -> None:
    # silent-install flags of the LunarG installer (Qt Installer Framework), same as the onnxruntime-ep-vulkan CI
    subprocess.run([str(installer), "--root", str(target), "--accept-licenses", "--default-answer",
                    "--confirm-command", "install"], check=True)


def add_cuda_target_layout(target: Path, platform: str) -> None:
    """Give the merged redist components the layout of a full toolkit install.

    nvcc.profile and FindCUDAToolkit look for headers and libraries under targets/<arch>-linux/, which the
    redist archives do not have (they ship plain include/ and lib/).
    """
    links = {target / "lib64": "lib"}
    arch_dir = target / "targets" / ("sbsa-linux" if platform == "linux-aarch64" else "x86_64-linux")
    links.update({arch_dir / "include": "../../include", arch_dir / "lib": "../../lib"})
    arch_dir.mkdir(parents=True, exist_ok=True)
    for link, dest in links.items():
        if not link.exists() and not link.is_symlink():
            link.symlink_to(dest, target_is_directory=True)


def setup(name: str, ctx, root: Path, download_dir: Path) -> dict:
    comp = dl.component(name)
    if ctx.platform not in comp.platforms:
        raise SystemExit(f"{name} does not support {ctx.platform}")
    out, resolved = dl.fetch(comp, ctx, download_dir)
    target = root / name
    marker = target / ".sdk-version"
    if marker.exists() and marker.read_text() == resolved.version:
        print(f"  {name} {resolved.version} already set up: {target}")
    elif resolved.git:
        target = out / "src"
    elif name == "vitisai":
        target = out  # wheels are pip-installed directly, not extracted
    else:
        if target.exists():
            shutil.rmtree(target)
        for item in resolved.files:
            path = out / item.filename
            if path.suffix == ".run":
                install_cann(path, target)
            elif path.suffix == ".exe":
                install_vulkan_sdk(path, target)
            else:
                extract(path, target)
        marker.write_text(resolved.version)
    if name == "cuda" and ctx.platform.startswith("linux"):
        add_cuda_target_layout(target, ctx.platform)
    if name == "cann":  # CANN 9.x installs into <path>/cann, 8.x into <path>/ascend-toolkit/latest
        target = next((c for c in (target / "cann", target / "ascend-toolkit" / "latest") if c.is_dir()), target)
    if name == "vulkan-sdk" and (target / "x86_64").is_dir():
        target = target / "x86_64"  # actual SDK root inside the Linux tarball
    return {"component": name, "version": resolved.version, "path": str(target)}


def export(results: list[dict]) -> None:
    env_lines, path_lines = [], []
    for r in results:
        base = Path(r["path"])
        for var, sub in ENV_VARS.get(r["component"], {}).items():
            env_lines.append(f"{var}={base / sub if sub else base}")
        if r["component"] in PATH_DIRS:
            path_lines.append(str(base / PATH_DIRS[r["component"]]))
    if gh_env := os.environ.get("GITHUB_ENV"):
        with open(gh_env, "a", encoding="utf-8") as f:
            f.write("\n".join(env_lines) + "\n")
    if (gh_path := os.environ.get("GITHUB_PATH")) and path_lines:
        with open(gh_path, "a", encoding="utf-8") as f:
            f.write("\n".join(path_lines) + "\n")
    for line in env_lines:
        print(f"  export {line}")


def prepare(names: list[str], platform: str, root: Path, cuda_major: str = "13", openvino_distro: str = "ubuntu24",
            pins: dict[str, str] | None = None, export_env: bool = True) -> dict[str, Path]:
    """Called by ci/build.py: set up a list of SDKs, return {component: path} and update sdk-versions.json."""
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    ctx = dl.Context(platform, cuda_major, openvino_distro, pins or {})
    results = []
    for name in names:
        print(f"==> SDK {name}", flush=True)
        results.append(setup(name, ctx, root, root / "_downloads"))
    versions_file = root / "sdk-versions.json"
    versions = json.loads(versions_file.read_text(encoding="utf-8")) if versions_file.exists() else {}
    versions.update({r["component"]: r["version"] for r in results})
    versions_file.write_text(json.dumps(versions, indent=2), encoding="utf-8")
    if export_env:
        export(results)
    return {r["component"]: Path(r["path"]) for r in results}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Download and extract the third-party SDKs needed by CI")
    ap.add_argument("components", nargs="*", help="component names (see scripts/download_ep_packages.py --list)")
    ap.add_argument("--root", type=Path, required=True, help="extraction directory")
    ap.add_argument("--platform", default=dl.detect_platform(), choices=dl.PLATFORMS)
    ap.add_argument("--cuda-major", default="13", choices=("12", "13"))
    ap.add_argument("--openvino-distro", default="ubuntu24")
    ap.add_argument("--pin", nargs="*", default=[], metavar="NAME=VER")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    prepare(args.components, args.platform, args.root, args.cuda_major, args.openvino_distro,
            dict(p.split("=", 1) for p in args.pin))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
