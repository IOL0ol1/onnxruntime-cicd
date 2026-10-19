#!/usr/bin/env python3
"""CI helper: plan the package builds of a bundle request.

Every package is built from a build recipe entry of ci/matrix.json (or, for a custom core, a synthetic entry with
the selected in-tree EPs) by the build-entry action. Its package artifact is named after the hash of its build
inputs, so a later request with the same inputs reuses the package (while the artifact lasts) instead of building it
again.

    python ci/plan.py                    # print every matrix entry with its inputs hash and reusable artifact

Environment variables (all optional):
  RUNNER_MAP        JSON mapping default runner labels to custom labels, e.g. {"ubuntu-24.04": "my-large-linux"}
  HEAVY_RUNNER_MAP  same, but only applied to heavy=true entries (long builds such as CUDA and QNN)
  CUDA_ARCHITECTURES  part of the inputs of the CUDA builds
  GITHUB_TOKEN      lets the plan find the packages of earlier runs with the same inputs
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import build

ROOT = Path(__file__).resolve().parent.parent
REPO = os.environ.get("GITHUB_REPOSITORY") or "IOL0ol1/onnxruntime-cicd"
DEFAULTS = {"variant": "default", "cuda_major": "13", "experimental": False, "heavy": False, "docker": "", "apt": "",
            "keep_android_sdk": False, "eps": ""}
# The files a package build depends on besides its entry, submodules and SDKs. Changes elsewhere (the bundle and
# page code) keep the packages valid.
BUILD_FILES = ("ci/build.py", "ci/sdk.py", "ci/package.py", "scripts", ".github/actions/build-entry")
# components whose resolved version depends on the CUDA major version
CUDA_DEPENDENT = {"cuda", "cudnn", "tensorrt", "tensorrt-rtx"}


CORE = "onnxruntime"  # package name of the core libraries
# core platform -> (recipe, variant) of ci/build.py
CORE_RECIPES = {"desktop": ("ort", "custom"), "arm64x": ("ort", "arm64x"), "android": ("ort", "android"),
                "apple": ("ort-apple", "default"), "s390x": ("ort", "s390x")}


def matrix() -> dict:
    return json.loads((ROOT / "ci" / "matrix.json").read_text(encoding="utf-8"))


def entries() -> list[dict]:
    """The plugin EP package entries of ci/matrix.json, with defaults."""
    return [{**DEFAULTS, **e} for e in matrix()["plugins"]]


def cores() -> dict[str, dict]:
    """Core recipe per target triple (core_from resolved: the triple whose core is used)."""
    out = {c["triple"]: c for c in matrix()["cores"]}
    return {t: {**out[c["core_from"]], "triple": t, "core_from": c["core_from"]} if c.get("core_from") else c
            for t, c in out.items()}


def core_entry(triple: str, eps: list[str], cuda_major: str = "13", wasm_mode: str = "",
               link: dict[str, str] | None = None) -> dict:
    """Build recipe entry of the core of a bundle: the core recipe of the target with the selected in-tree EPs (the
    WebAssembly build mode on wasm32), the packages and container its EPs need, and heavy if one of them is. link holds
    the linkage choices other than the defaults, passed to ci/build.py as <ep>:<choice>."""
    core = cores()[triple]
    caps = core["eps"]
    if core["platform"] == "wasm":
        recipe, variant, eps = "ort", core["wasm_modes"][wasm_mode]["variant"], []
    else:
        recipe, variant = CORE_RECIPES[core["platform"]]
    apt, docker = set(core.get("apt", "").split()), ""
    heavy, experimental = False, bool(core.get("experimental"))
    for ep in eps:
        req = caps[ep]
        apt |= set(req.get("apt", "").split())
        docker = docker or req.get("docker", "")
        heavy = heavy or bool(req.get("heavy"))
        experimental = experimental or bool(req.get("experimental"))
    return {**DEFAULTS, "package": CORE, "recipe": recipe, "variant": variant,
            "triple": core.get("core_from") or triple, "runner": core["runner"], "submodules": "onnxruntime",
            "cuda_major": cuda_major or "13", "eps": ",".join(f"{ep}:{link[ep]}" if ep in (link or {}) else ep for ep in eps), "apt": " ".join(sorted(apt)), "docker": docker,
            "heavy": heavy, "experimental": experimental, "keep_android_sdk": bool(core.get("keep_android_sdk"))}


def msvc_arch(triple: str, runner: str) -> str:
    """MSVC environment for Windows jobs: native on ARM64 runners, cross amd64_arm64 for ARM64 on x64 runners.
    ARM64EC / ARM64X builds use the Visual Studio generator, which selects the platform itself."""
    if "arm" in runner:
        return "arm64"
    if triple.startswith("aarch64") and "windows" in triple:
        return "amd64_arm64"
    return "amd64"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()


def sdks_of(e: dict) -> set[str]:
    return {c for c, _, _ in build.required_sdks(e["recipe"], e["variant"], e["triple"], e["cuda_major"], e["eps"])}


class Inputs:
    """Hash of everything a build entry depends on: the entry itself, the commits of its submodules, the build scripts,
    the resolved third-party SDK versions, the Docker image digest and the CUDA architectures. Anything that cannot be
    resolved makes the hash unique, so the entry is built."""

    def __init__(self) -> None:
        self.files = git("rev-parse", *(f"HEAD:{p}" for p in BUILD_FILES)).split()
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
            except Exception as e:
                value = f"unresolved-{os.urandom(8).hex()}"
                print(f"  {comp} ({plat}): not resolved ({str(e).splitlines()[0][:100]})", file=sys.stderr)
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
        cuda = e["cuda_major"]
        sdks = sorted((c, p, pin, self.sdk(c, p, cuda if c in CUDA_DEPENDENT else "13", pin))
                      for c, p, pin in build.required_sdks(e["recipe"], e["variant"], e["triple"], cuda, e["eps"]))
        recipe = {k: e[k] for k in ("package", "recipe", "variant", "triple", "cuda_major", "eps", "docker", "apt")}
        data = {"entry": recipe, "submodules": subs, "files": self.files, "sdks": sdks, "image": self.image(e["docker"]),
                "cuda_archs": os.environ.get("CUDA_ARCHITECTURES", "")}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def build_entry(e: dict, inputs: Inputs) -> dict:
    """A build job of the bundle workflow for a recipe entry: runner, MSVC settings, inputs hash and artifact name."""
    runner_map = json.loads(os.environ.get("RUNNER_MAP") or "{}")
    heavy_map = json.loads(os.environ.get("HEAVY_RUNNER_MAP") or "{}")
    runner = e["runner"]
    sdks = sdks_of(e)
    digest = inputs.of(e)
    name = e["package"] + (f"-{hashlib.sha256(e['eps'].encode()).hexdigest()[:6]}" if e["eps"] else "")
    return {
        "id": f"{name}-{e['triple']}",
        "package": e["package"], "recipe": e["recipe"], "variant": e["variant"], "triple": e["triple"],
        "eps": e["eps"], "cuda_major": e["cuda_major"], "submodules": e["submodules"], "docker": e["docker"],
        "apt": e["apt"], "keep_android_sdk": bool(e["keep_android_sdk"]),
        "runner": (heavy_map.get(runner) if e["heavy"] else None) or runner_map.get(runner) or runner,
        "msvc_arch": msvc_arch(e["triple"], runner),
        # CUDA 12 does not accept the VS 2026 compiler (MSVC 14.5x), and the constexpr <cmath> comparisons of its STL
        # clash with the HIP device overloads when building MIGraphX: use the VS 2022 toolset the image also ships
        "msvc_toolset": "14.44" if (e["cuda_major"] == "12" and "cuda" in sdks) or e["recipe"] == "ep-amdgpu" else "",
        # compiler caching (ccache in the Actions cache, 10 GB per repository) for the long builds only: CUDA and
        # MIGraphX; container builds are left out
        "ccache": not e["docker"] and ("cuda" in sdks or (e["recipe"] == "ep-amdgpu" and e["variant"] != "dml")),
        # cores of different EP sets share their ccache directory (at least the third-party code is the same)
        "ccache_key": f"{CORE}-{e['triple']}-cuda{e['cuda_major']}" if e["package"] == CORE else f"{name}-{e['triple']}",
        "hash": digest,
        "artifact": f"pkg-{name}-{e['triple']}-{digest}",
    }


def find_artifact(name: str) -> dict | None:
    """{"run": run id, "id": artifact id} of the newest unexpired artifact with this name, or None."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return None
    req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/actions/artifacts?name={name}&per_page=20",
                                 headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    try:
        arts = json.load(urllib.request.urlopen(req, timeout=60))["artifacts"]
    except (OSError, KeyError, ValueError):
        return None
    arts = sorted((a for a in arts if not a["expired"]), key=lambda a: a["created_at"], reverse=True)
    return {"run": str(arts[0]["workflow_run"]["id"]), "id": str(arts[0]["id"])} if arts else None


def main() -> int:
    inputs = Inputs()
    default_cores = [core_entry(t, [ep for ep in ("xnnpack",) if ep in c["eps"]], wasm_mode="cpu")
                     for t, c in cores().items() if not c.get("core_from")]
    for e in default_cores + entries():
        b = build_entry(e, inputs)
        found = find_artifact(b["artifact"])
        print(f"{b['id']:60} {b['runner']:18} ccache={str(b['ccache']):5} {b['hash']} "
              f"{'reuse run ' + found['run'] if found else 'build'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
