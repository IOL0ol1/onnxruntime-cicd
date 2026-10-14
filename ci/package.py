#!/usr/bin/env python3
"""CI helper: pack a staging directory into <name>-<version>-<target-triple>.(tar.gz|zip) plus a .sha256 file.

The staging directory holds bin/, lib/ and include/ (filled by ci/build.py). Packaging also writes
BUILD_INFO.json with the target triple, the commits of this repository and the relevant submodules, and the
third-party SDK versions used. The third-party SDKs themselves are never packaged (their licenses generally
do not allow redistribution).

    python ci/package.py --stage stage/onnxruntime-cuda --name onnxruntime-cuda \\
        --triple x86_64-unknown-linux-gnu --version 1.31.0-20261009.abc1234 \\
        --source onnxruntime --sdk-versions $RUNNER_TEMP/sdk/sdk-versions.json --out dist
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def git_head(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_archive(stage: Path, archive: Path, top: str) -> None:
    files = sorted(p for p in stage.rglob("*") if p.is_file() or p.is_symlink())
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for f in files:
                zf.write(f, f"{top}/{f.relative_to(stage).as_posix()}")
    else:
        with tarfile.open(archive, "w:gz", compresslevel=9) as tf:
            tf.add(stage, arcname=top)  # keeps symlinks (e.g. libonnxruntime.so -> libonnxruntime.so.1)


def unpack(src_dir: Path, stage: Path) -> dict:
    """Extract the package archive in src_dir into stage (without its top-level directory); return its BUILD_INFO."""
    archives = [p for p in src_dir.iterdir() if p.name.endswith((".tar.gz", ".zip"))]
    if len(archives) != 1:
        raise SystemExit(f"expected one package archive in {src_dir}, found {[p.name for p in archives]}")
    archive = archives[0]
    with tempfile.TemporaryDirectory(dir=stage.parent) as tmp:
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(tmp)
        else:
            with tarfile.open(archive) as tf:
                tf.extractall(tmp, filter="tar")  # keeps the library symlinks
        top = next(Path(tmp).iterdir())
        stage.mkdir(parents=True, exist_ok=True)
        for item in top.iterdir():
            shutil.move(str(item), str(stage / item.name))
    info_file = stage / "BUILD_INFO.json"
    previous = json.loads(info_file.read_text(encoding="utf-8")) if info_file.exists() else {}
    info_file.unlink(missing_ok=True)
    print(f"repackaging {archive.name} (built {previous.get('built_at')})")
    return previous


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Package build outputs")
    ap.add_argument("--stage", type=Path, required=True, help="staging directory (with bin/ lib/ include/)")
    ap.add_argument("--name", required=True, help="package name, e.g. onnxruntime-cuda")
    ap.add_argument("--triple", required=True, help="target triple, e.g. x86_64-unknown-linux-gnu")
    ap.add_argument("--version", required=True)
    ap.add_argument("--source", nargs="*", default=[], help="submodule paths whose commits are recorded (relative to the repository root)")
    ap.add_argument("--sdk-versions", type=Path, help="sdk-versions.json produced by ci/sdk.py")
    ap.add_argument("--extra", nargs="*", default=[], metavar="KEY=VALUE", help="extra fields to write into BUILD_INFO")
    ap.add_argument("--out", type=Path, default=ROOT / "dist")
    ap.add_argument("--from-archive", type=Path, metavar="DIR",
                    help="repackage the package found in DIR (from an earlier run with the same build inputs) under "
                         "this version instead of packaging a fresh build; its sources and SDK versions are kept")
    args = ap.parse_args(argv)

    stage = args.stage.resolve()
    previous = unpack(args.from_archive, stage) if args.from_archive else None
    if not any(stage.rglob("*")):
        raise SystemExit(f"staging directory is empty: {stage}")
    for sub in ("lib", "include"):
        if not (stage / sub).exists():
            print(f"note: staging directory has no {sub}/", file=sys.stderr)

    info = {
        "name": args.name,
        "version": args.version,
        "target_triple": args.triple,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repository": {"path": ".", "commit": git_head(ROOT)},
        "sources": {s: git_head(ROOT / s) for s in args.source},
        "third_party_sdks": (json.loads(args.sdk_versions.read_text(encoding="utf-8"))
                             if args.sdk_versions and args.sdk_versions.exists() else {}),
        "note": "Redistributable third-party runtimes are bundled (see third_party_licenses/); fetch the others (CUDA, "
                "cuDNN, TensorRT, CANN) with scripts/fetch_ep_runtime.py --package-dir <this directory>.",
    }
    info.update(dict(e.split("=", 1) for e in args.extra))
    if previous:  # the binaries were built by the earlier run: keep what it recorded about them
        info["sources"], info["third_party_sdks"] = previous.get("sources", {}), previous.get("third_party_sdks", {})
        info["reused_from"] = {k: previous.get(k) for k in ("version", "built_at")}
        info["reused_from"]["commit"] = previous.get("repository", {}).get("commit")
    if "cuda" not in info["third_party_sdks"]:  # the workflow passes the CUDA settings for every build
        info.pop("cuda_architectures", None)
        info.pop("cuda_major", None)
    (stage / "BUILD_INFO.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    # the license of the packaged project: the first source that has one (an EP repository, or onnxruntime)
    licenses = [ROOT / s / "LICENSE" for s in args.source] + [ROOT / "onnxruntime" / "LICENSE"]
    license_file = next((f for f in licenses if f.exists()), None)
    if license_file and not (stage / "LICENSE").exists():
        (stage / "LICENSE").write_bytes(license_file.read_bytes())
    # the notices of the open-source code linked into the binaries (ONNX Runtime and some EP repositories keep one)
    notices = next((f for f in (ROOT / s / "ThirdPartyNotices.txt" for s in args.source) if f.exists()), None)
    if notices and not (stage / notices.name).exists():
        (stage / notices.name).write_bytes(notices.read_bytes())

    top = f"{args.name}-{args.version}-{args.triple}"
    ext = ".zip" if "windows" in args.triple else ".tar.gz"
    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out / f"{top}{ext}"
    write_archive(stage, archive, top)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (args.out / f"{archive.name}.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"{archive}  {archive.stat().st_size >> 20} MiB  sha256={digest}")
    if archive.stat().st_size >= 2 << 30:  # GitHub rejects release assets of 2 GiB or more
        print(f"::warning::{archive.name} is {archive.stat().st_size >> 20} MiB, too large for a GitHub release asset")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
