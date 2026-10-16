#!/usr/bin/env python3
"""CI helper: generate the build matrix (with the inputs of each entry, see add_reuse) and the version; write them to $GITHUB_OUTPUT.

Environment variables (all optional):
  FILTER            keep only entries whose package name or "package/triple" matches this regex (workflow_dispatch input)
  RUNNER_MAP        JSON mapping default runner labels to custom labels, e.g. {"ubuntu-24.04": "my-large-linux"}
  HEAVY_RUNNER_MAP  same, but only applied to heavy=true entries (long builds such as CUDA and QNN)
  FORCE_REBUILD     "true" to build every entry even if an earlier run built the same inputs (see add_reuse)
  GITHUB_TOKEN      lets the plan find earlier runs with the same inputs; without it everything is built
"""

from __future__ import annotations

import datetime
import hashlib
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


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()


class Inputs:
    """Hash of everything a build entry depends on, used to reuse the package of an earlier run with the same inputs.

    Covered: the entry itself, the commits of its submodules, the CI scripts and workflow (any change there rebuilds
    everything), the resolved third-party SDK versions, the Docker image digest and the CUDA architectures. Anything
    that cannot be resolved makes the entry unique to this run, so it is rebuilt.
    """

    def __init__(self) -> None:
        self.ci = git("rev-parse", "HEAD:ci", "HEAD:scripts", "HEAD:.github/workflows/build.yml",
                      "HEAD:.github/actions").split()
        self.sdks: dict[tuple, str] = {}
        self.images: dict[str, str] = {}

    def sdk(self, comp: str, plat: str, cuda: str, pin: str) -> str:
        key = (comp, plat, cuda, pin)
        if key not in self.sdks:
            dl = build.sdk.dl
            try:
                ctx = dl.Context(plat, cuda, "ubuntu24", {comp: pin} if pin else {})
                resolved = dl.component(comp).resolve(ctx, ctx.pins.get(comp, dl.component(comp).default))
                if resolved.git:  # a branch: use the commit it points to
                    repo, ref = resolved.git
                    out = subprocess.run(["git", "ls-remote", repo, ref], capture_output=True, text=True, timeout=120)
                    value = (out.stdout.split() or [""])[0] or f"unresolved-{os.urandom(8).hex()}"
                else:
                    value = resolved.version + "|" + ",".join(f.filename for f in resolved.files)
            except Exception as e:  # e.g. a version read from a submodule that the plan job has not checked out
                value = f"unresolved-{os.urandom(8).hex()}"
                print(f"  {comp} ({plat}): not resolved here ({str(e).splitlines()[0][:100]})", file=sys.stderr)
            self.sdks[key] = value
        return self.sdks[key]

    def image(self, name: str) -> str:
        if name and name not in self.images:
            repo, _, tag = name.partition(":")
            try:
                url = f"https://hub.docker.com/v2/repositories/{repo}/tags/{tag or 'latest'}"
                self.images[name] = json.load(urllib.request.urlopen(url, timeout=60))["digest"]
            except (OSError, KeyError, ValueError):
                self.images[name] = f"unresolved-{os.urandom(8).hex()}"
        return self.images.get(name, "")

    def of(self, e: dict) -> str:
        subs = {s: git("rev-parse", f"HEAD:{s}") for s in e["submodules"].split()}
        sdks = sorted((c, p, pin, self.sdk(c, p, e["cuda_major"] if c in CUDA_DEPENDENT else "13", pin))
                      for c, p, pin in build.required_sdks(e["recipe"], e["variant"], e["triple"], e["cuda_major"]))
        data = {"entry": e, "submodules": subs, "ci": self.ci, "sdks": sdks, "image": self.image(e["docker"]),
                "cuda_archs": os.environ.get("CUDA_ARCHITECTURES", "")}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:32]


def previous_run(marker: str) -> str:
    """Run id of the newest unexpired build whose marker artifact records the same inputs ("" if none)."""
    repo, token = os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_TOKEN")
    if not (repo and token):
        return ""
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}/actions/artifacts?name={marker}",
                                 headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    try:
        arts = json.load(urllib.request.urlopen(req, timeout=60))["artifacts"]
    except (OSError, KeyError, ValueError):
        return ""
    arts = sorted((a for a in arts if not a["expired"]), key=lambda a: a["created_at"], reverse=True)
    return str(arts[0]["workflow_run"]["id"]) if arts else ""


def add_reuse(builds: list[dict]) -> None:
    """Give each entry its inputs hash and, unless FORCE_REBUILD is set, the run whose package it can reuse."""
    force = os.environ.get("FORCE_REBUILD", "").lower() == "true"
    inputs = Inputs()
    for e in builds:
        # marker artifact of a build, named like the package artifact pkg-<id>
        e["inputs_artifact"] = f"inputs-{e['id']}-{inputs.of(e)}"
        e["reuse_run"] = "" if force else previous_run(e["inputs_artifact"])
        if e["reuse_run"]:
            print(f"  reuse {e['id']} from run {e['reuse_run']}")


def main() -> int:
    builds = build_matrix()
    add_reuse(builds)
    v, tag = version()
    outputs = {
        "build_matrix": json.dumps({"include": builds}),
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
