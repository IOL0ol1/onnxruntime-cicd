#!/usr/bin/env python3
"""CI helper: generate the build matrix, the third-party SDK archive matrix and the version; write them to $GITHUB_OUTPUT.

Environment variables (all optional):
  FILTER            keep only entries whose package name or "package/triple" matches this regex (workflow_dispatch input)
  RUNNER_MAP        JSON mapping default runner labels to custom labels, e.g. {"ubuntu-24.04": "my-large-linux"}
  HEAVY_RUNNER_MAP  same, but only applied to heavy=true entries (long builds such as CUDA and QNN)
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_bspec = importlib.util.spec_from_file_location("ci_build", ROOT / "ci" / "build.py")
build = importlib.util.module_from_spec(_bspec)
sys.modules["ci_build"] = build
_bspec.loader.exec_module(build)

DEFAULTS = {"variant": "default", "cuda_major": "13", "experimental": False, "heavy": False, "docker": "", "apt": "",
            "keep_android_sdk": False}


def version() -> tuple[str, str]:
    """Return (version, release tag). Tag pushes use the tag itself; otherwise ORT version + date + commit."""
    ref = os.environ.get("GITHUB_REF", "")
    if ref.startswith("refs/tags/v"):
        tag = ref.removeprefix("refs/tags/")
        return tag[1:], tag
    sha = os.environ.get("GITHUB_SHA") or subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                                          cwd=ROOT).stdout.strip()
    ort = "0.0.0"
    try:  # the submodule may not be checked out, so read VERSION_NUMBER upstream at the gitlink commit
        line = subprocess.run(["git", "ls-tree", "HEAD", "onnxruntime"], capture_output=True, text=True, cwd=ROOT).stdout
        ort_sha = line.split()[2]
        url = f"https://raw.githubusercontent.com/microsoft/onnxruntime/{ort_sha}/VERSION_NUMBER"
        ort = urllib.request.urlopen(url, timeout=60).read().decode().strip()
    except (IndexError, OSError):
        pass
    date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    v = f"{ort}-dev.{date}.{sha[:8] or 'local'}"
    return v, f"build-{v}"


def msvc_arch(triple: str, runner: str) -> str:
    """MSVC environment for Windows jobs: native on ARM64 runners, cross amd64_arm64 for ARM64 on x64 runners.
    ARM64EC / ARM64X builds use the Visual Studio generator, which selects the platform itself."""
    if "arm" in runner:
        return "arm64"
    if triple.startswith("aarch64") and "windows" in triple:
        return "amd64_arm64"
    return "amd64"


def build_matrix() -> list[dict]:
    entries = json.loads((ROOT / "ci" / "matrix.json").read_text(encoding="utf-8"))["include"]
    pattern = re.compile(os.environ.get("FILTER") or ".*")
    runner_map = json.loads(os.environ.get("RUNNER_MAP") or "{}")
    heavy_map = json.loads(os.environ.get("HEAVY_RUNNER_MAP") or "{}")
    out = []
    for e in entries:
        e = {**DEFAULTS, **e}
        if not (pattern.search(e["package"]) or pattern.search(f"{e['package']}/{e['triple']}")):
            continue
        runner = e["runner"]
        e["runner"] = (heavy_map.get(runner) if e["heavy"] else None) or runner_map.get(runner) or runner
        e.setdefault("msvc_arch", msvc_arch(e["triple"], runner))
        e["id"] = f"{e['package']}-{e['triple']}"
        out.append(e)
    ids = [e["id"] for e in out]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise SystemExit(f"duplicate package-triple entries in the matrix: {dup}")
    return out


# components whose resolved version depends on the CUDA major version
CUDA_DEPENDENT = {"cuda", "cudnn", "tensorrt", "tensorrt-rtx"}


def sdk_matrix(builds: list[dict]) -> list[dict]:
    """Collect the SDKs the builds actually use; one archive job per (component, platform, CUDA major, pin)."""
    entries: set[tuple[str, str, str, str]] = set()
    for b in builds:
        for comp, plat, pin in build.required_sdks(b["recipe"], b["variant"], b["triple"], b["cuda_major"]):
            entries.add((comp, plat, b["cuda_major"] if comp in CUDA_DEPENDENT else "13", pin))
    out = []
    for comp, plat, cuda, pin in sorted(entries):
        suffix = (f"-cuda{cuda}" if comp in CUDA_DEPENDENT else "") + (f"-{pin}" if pin else "")
        out.append({"component": comp, "platform": plat, "cuda_major": cuda,
                    "pin": f"{comp}={pin}" if pin else "", "name": f"{plat}-{comp}{suffix}"})
    return out


def main() -> int:
    builds = build_matrix()
    v, tag = version()
    outputs = {
        "build_matrix": json.dumps({"include": builds}),
        "sdk_matrix": json.dumps({"include": sdk_matrix(builds)}),
        "version": v,
        "release_tag": tag,
        "count": str(len(builds)),
    }
    for k, val in outputs.items():
        print(f"{k}={val if k != 'build_matrix' else f'<{len(builds)} entries>'}")
    if gh_out := os.environ.get("GITHUB_OUTPUT"):
        with open(gh_out, "a", encoding="utf-8") as f:
            for k, val in outputs.items():
                f.write(f"{k}={val}\n")
    if not builds:
        raise SystemExit("FILTER matched no build entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
