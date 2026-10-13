# onnxruntime-cicd

This repository collects the ONNX Runtime main repository and the Execution Provider (EP) implementations that are currently maintained:

- Every project with public git sources is included as a **git submodule** (shallow clone).
- SDKs and dependencies that cannot be fetched automatically at build time (CUDA, TensorRT, QAIRT, OpenVINO, CANN, ROCm, the Vitis AI runtime, ...) are downloaded into `packages/` by [`scripts/download_ep_packages.py`](scripts/download_ep_packages.py) from official sources that need no account; see [External dependencies and the download script](#external-dependencies-and-the-download-script).
- [`.github/workflows/build.yml`](.github/workflows/build.yml) builds everything with a matrix and publishes packages per target triple; see [Continuous integration](#continuous-integration-github-actions).

> Surveyed on 2026-10-08, against the latest ONNX Runtime release **v1.30.0** (2026-09-10) and the `main` branch.

## Repository layout

```
onnxruntime-cicd/
├── onnxruntime/                         # microsoft/onnxruntime main repository (all in-tree EPs)
├── eps/
│   ├── official/                        # standalone plugin EPs under the onnxruntime GitHub organization
│   │   ├── onnxruntime-qnn/             # Qualcomm QNN plugin EP (maintained by Qualcomm)
│   │   ├── onnxruntime-ep-tensorrt/     # NVIDIA TensorRT plugin EP
│   │   ├── onnxruntime-ep-amdgpu/       # AMD plugin EPs: AMDGPU / MIGraphX / DirectML
│   │   └── onnxruntime-ep-mlx/          # Apple MLX plugin EP
│   ├── vendor/                          # EP sources or forks maintained by hardware vendors
│   │   ├── nvidia-tensorrt-rtx-ep/      # NVIDIA TensorRT RTX plugin EP
│   │   ├── intel-onnxruntime-openvino/  # Intel fork, ovep-develop branch (OpenVINO EP development line)
│   │   └── arm-china-compass-onnxruntime/  # Arm China fork, Zhouyi NPU EP
│   └── community/                       # third-party / community plugin EPs
│       ├── onnxruntime-ep-iree/         # IREE (MLIR compiler) plugin EP
│       ├── onnxruntime-ep-telum/        # IBM Telum (IBM Z AI accelerator) plugin EP
│       └── onnxruntime-ep-vulkan/       # Vulkan compute plugin EP (experimental)
├── .github/workflows/build.yml          # CI: matrix builds, packaging and releases for all platforms
├── ci/                                  # CI scripts: plan.py / build.py / sdk.py / package.py / matrix.json
├── scripts/
│   ├── download_ep_packages.py          # downloads the external SDKs and dependencies EPs need
│   └── fetch_ep_runtime.py              # extracts the runtime files of closed-source SDKs for the release packages
└── packages/                            # download output (in .gitignore)
```

## Quick start

```bash
# Clone the repository and all submodules as shallow clones
git clone --recurse-submodules --shallow-submodules <repository URL>

# In an existing clone, initialize or complete the submodules
git submodule update --init --depth 1

# Move every submodule to the latest commit of its tracked upstream branch
git submodule update --remote --depth 1

# Fetch a single submodule
git submodule update --init --depth 1 eps/official/onnxruntime-ep-tensorrt
```

> The main repository and the three forks (`onnxruntime-qnn`, `intel-onnxruntime-openvino`, `arm-china-compass-onnxruntime`) are full ONNX Runtime source trees and therefore large. `.gitmodules` sets `shallow = true`. On Windows, run `git config --global core.longpaths true` first.

## Submodules

| Path | Upstream | Tracked branch | EP / target hardware | Maintainer | Notes |
|---|---|---|---|---|---|
| `onnxruntime` | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | `main` | All in-tree EPs (see below), plus the CUDA / WebGPU plugin EPs | Microsoft | Main repository |
| `eps/official/onnxruntime-qnn` | [onnxruntime/onnxruntime-qnn](https://github.com/onnxruntime/onnxruntime-qnn) | `main` | Qualcomm Snapdragon (QAIRT / HTP / NPU) | Qualcomm | Shipped as a standalone plugin EP since 2.0.0 (`onnxruntime-qnn>=2.0`) |
| `eps/official/onnxruntime-ep-tensorrt` | [onnxruntime/onnxruntime-ep-tensorrt](https://github.com/onnxruntime/onnxruntime-ep-tensorrt) | `main` | NVIDIA GPU (TensorRT 10/11) | ONNX Runtime team | Migrated from the in-tree TensorRT EP; requires ORT ≥ 1.23 |
| `eps/official/onnxruntime-ep-amdgpu` | [onnxruntime/onnxruntime-ep-amdgpu](https://github.com/onnxruntime/onnxruntime-ep-amdgpu) | `main` | AMD GPU / Instinct (AMDGPU, MIGraphX), Windows DX12 (DirectML) | AMD | Requires ORT ≥ 1.24; plugin versions of MIGraphX / DirectML |
| `eps/official/onnxruntime-ep-mlx` | [onnxruntime/onnxruntime-ep-mlx](https://github.com/onnxruntime/onnxruntime-ep-mlx) | `main` | Apple Silicon (MLX) | Community (moved into the onnxruntime organization) | PyPI package `onnxruntime-ep-mlx`; requires ORT ≥ 1.29 |
| `eps/vendor/nvidia-tensorrt-rtx-ep` | [NVIDIA/TensorRT-RTX-EP-ABI](https://github.com/NVIDIA/TensorRT-RTX-EP-ABI) | `main` | NVIDIA RTX GPU (TensorRT-RTX) | NVIDIA | Plugin TensorRT RTX EP; PyPI package `onnxruntime-ep-nv-tensorrt-rtx` |
| `eps/vendor/intel-onnxruntime-openvino` | [intel/onnxruntime](https://github.com/intel/onnxruntime) | `ovep-develop` | Intel CPU / iGPU / dGPU / NPU (OpenVINO) | Intel | OpenVINO EP development line, regularly synced with upstream; the official plugin package `onnxruntime-ep-openvino` is binary-only |
| `eps/vendor/arm-china-compass-onnxruntime` | [Arm-China/Compass_Onnxruntime](https://github.com/Arm-China/Compass_Onnxruntime) | `rel-1.22.0` | Arm China Zhouyi NPU | Arm China | Fork of ORT 1.22 that adds the Zhouyi NPU EP; old baseline but still updated in 2026 |
| `eps/community/onnxruntime-ep-iree` | [iree-org/onnxruntime-ep-iree](https://github.com/iree-org/onnxruntime-ep-iree) | `main` | CPU / HIP / CUDA / Vulkan through IREE | IREE community | Plugin EP |
| `eps/community/onnxruntime-ep-telum` | [IBM/onnxruntime-ep-telum](https://github.com/IBM/onnxruntime-ep-telum) | `main` | IBM Telum (on-chip AI accelerator of IBM Z, zDNN) | IBM | Plugin EP; requires ORT ≥ 1.24.1 |
| `eps/community/onnxruntime-ep-vulkan` | [justinchuby/onnxruntime-ep-vulkan](https://github.com/justinchuby/onnxruntime-ep-vulkan) | `main` | Cross-platform Vulkan compute | Personal project | **Experimental**, written in Rust; its README states that CI is currently red |

## In-tree EPs of the main repository

Sources live in `onnxruntime/onnxruntime/core/providers/<directory>`.

| EP | Directory | Status | Notes / external implementation |
|---|---|---|---|
| CPU | `cpu` | Stable | Default EP, MLAS / KleidiAI kernels |
| CUDA | `cuda` | Stable | The plugin CUDA EP sources are in `cuda/plugin`, packaging in `plugin-ep-cuda/`, design docs in `docs/cuda_plugin_ep/` (plugin version 0.2 in v1.30) |
| TensorRT | `tensorrt` | Stable | Plugin version: `eps/official/onnxruntime-ep-tensorrt` |
| NVIDIA TensorRT RTX | `nv_tensorrt_rtx` | Stable | Plugin version: `eps/vendor/nvidia-tensorrt-rtx-ep` |
| DirectML | `dml` | Sustained engineering | New feature work moved to Windows ML; plugin version: `onnxruntime-ep-amdgpu/src/directx` |
| Intel OpenVINO | `openvino` | Stable | Development line: `eps/vendor/intel-onnxruntime-openvino`; plugin binary package `onnxruntime-ep-openvino` |
| Qualcomm QNN | `qnn` | In-tree version (provider bridge, 1.x) | New plugin EP: `eps/official/onnxruntime-qnn` |
| Apple CoreML | `coreml` | Preview | |
| Android NNAPI | `nnapi` | Stable | |
| XNNPACK | `xnnpack` | Stable | CPU acceleration for mobile / web |
| WebGPU | `webgpu` | Stable | Plugin packaging in `plugin-ep-webgpu/` (plugin version 0.4.0 in v1.30) |
| WebNN | `webnn` | Stable (web) | Used by onnxruntime-web |
| JSEP | `js` | Stable (web) | JavaScript EP of onnxruntime-web; front-end code in `js/` |
| AMD MIGraphX | `migraphx` | Stable | Plugin version: `eps/official/onnxruntime-ep-amdgpu/src/migraphx` |
| AMD Vitis AI | `vitisai` | Preview | The repository only contains a bridge layer; the closed-source backend is fetched by the download script (see below) |
| Huawei CANN | `cann` | Preview, community-maintained | |
| Intel oneDNN | `dnnl` | Stable, low activity | |
| Azure | `azure` | Preview | Cloud inference EP; no substantial code changes for a long time |
| VeriSilicon VSINPU | `vsinpu` | Community-maintained, low activity | VeriSilicon's own fork is archived |
| Rockchip RKNPU | `rknpu` | Obsolete | Targets early RK NPUs (depends on rknpu_ddk) and only receives security fixes; newer Rockchip platforms use the RKNN Toolkit |
| SNPE | `snpe` | Obsolete | Superseded by the QNN EP; last code change in 2025-03 |
| ACL | `acl` | **Deprecated** (2026-09, #32564) | Build flag marked `[DEPRECATED]`; will be removed in a future release |

> `winml`, `shared_library` and `shared` are infrastructure / API layers, not EPs.

## External dependencies and the download script

The submodules are not self-contained. External dependencies fall into three groups:

1. **Fetched automatically at build time; nothing to do**
   - ORT main repository: the 46 open-source dependencies in `cmake/deps.txt` (protobuf, abseil, XNNPACK, oneDNN, Dawn, ...) are downloaded with FetchContent; the nested submodules (onnx, emsdk, libprotobuf-mutator) are pulled by `build.py`, which runs `git submodule update --init --recursive`; DirectML / WIL / DirectX-Headers come from NuGet or FetchContent.
   - Plugin EPs: the TensorRT / TensorRT RTX / AMD / IREE / Telum plugins use FetchContent or vcpkg; the MLX plugin's `setup_mlx.sh` downloads mlx, mlx-c and metal-cpp; the Rust dependencies of the MLX and Vulkan plugins come from crates.io.
2. **SDKs and dependencies that must be prepared before building**: downloaded by [`scripts/download_ep_packages.py`](scripts/download_ep_packages.py); see the table below.
3. **Out of scope**
   - General toolchains: compilers, CMake, Rust, JDK, and Xcode (needed by CoreML).
   - Hardware drivers: NVIDIA, Intel NPU, AMD NPU, Qualcomm, and so on.
   - Obsolete or deprecated EPs: ACL, SNPE, RKNPU.

### Download script

The script only needs the Python standard library; git components also need `git`. Every source is an official direct link or official mirror that **requires no account**. Default versions follow what the relevant repositories' CI uses wherever possible and can be overridden with `--pin <component>=<version>` or `--pin <component>=latest`. Files with an upstream sha256 / sha1 are verified; large downloads resume after interruption, and completed files are skipped on later runs.

```bash
python scripts/download_ep_packages.py --list                                    # components, default versions and sources
python scripts/download_ep_packages.py --dry-run                                 # resolve every file for this platform without downloading
python scripts/download_ep_packages.py --platform linux-x86_64 --cuda-major 12  # choose the target platform and CUDA major version
python scripts/download_ep_packages.py --only cuda cudnn tensorrt                # only some components
python scripts/download_ep_packages.py --only tensorrt-rtx --pin tensorrt-rtx=latest
python scripts/download_ep_packages.py --only vitisai --pin voe=1.6.1 ryzenai-dynamic-dispatch=1.6.1 onnxruntime-vitisai=1.23.0
python scripts/download_ep_packages.py --with-optional                           # include optional components (hexagon-sdk)
```

- Downloads go to `packages/<component>/<version>/` together with a `manifest.json`; git components are cloned to `packages/<component>/<ref>/src`.
- Platforms: `linux-x86_64`, `linux-aarch64`, `linux-s390x`, `windows-x86_64`, `windows-arm64`, `windows-arm64x`, `macos-arm64`; the default is the current host. Components without an upstream package for the platform are skipped with an explanation.
- If the GitHub API rate limit is hit, set the `GITHUB_TOKEN` environment variable (optional).

| Component | Used by | Default version | Source (no account needed) | Checksum |
|---|---|---|---|---|
| `cuda` | CUDA, TensorRT, TensorRT plugin, TensorRT RTX | Latest 13.x (`--cuda-major 12` switches to 12.x) | NVIDIA redist: `developer.download.nvidia.com/compute/cuda/redist` | sha256 |
| `cudnn` | CUDA | Latest 9.x | NVIDIA redist: `developer.download.nvidia.com/compute/cudnn/redist` | sha256 |
| `tensorrt` | TensorRT (in-tree), `onnxruntime-ep-tensorrt` | 10.15 (same as ORT CI); `latest` is 11.x | `developer.nvidia.com/downloads/...`, links taken from the README of each [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) release branch; requests from mainland China are redirected to `developer.download.nvidia.cn` | — |
| `tensorrt-rtx` | TensorRT RTX (in-tree + plugin) | 1.6 (the plugin's CMake requires ≥ 1.6 at configure time) | `rtx_sdk/secure/...` direct links listed on the [official TensorRT-RTX page](https://developer.nvidia.com/tensorrt-rtx) (verified to work without login) | — |
| `qairt` | QNN (in-tree + `onnxruntime-qnn`) | 2.51.0.260929 (from `onnxruntime-qnn/qcom/packages.yml`) | Qualcomm public download API `softwarecenter.qualcomm.com/api/download/...` (instead of QPM, which needs a Qualcomm ID) | sha256 |
| `hexagon-sdk` (optional) | QNN (only needed for custom HTP ops) | 6.5.0.0 (same source) | Same as above | sha256 |
| `openvino` | OpenVINO (in-tree + Intel fork) | 2026.2.1 (same as the Intel fork CI) | `storage.openvinotoolkit.org` | sha256 |
| `cann` | Huawei CANN | Latest release (currently 9.1.1) | Huawei Cloud Ascend artifactory `ascend.devcloud.huaweicloud.com/artifactory/cann-run` (instead of the official site, which needs a Huawei account) | sha256 |
| `rocm` | MIGraphX / AMDGPU (in-tree + `onnxruntime-ep-amdgpu`) | Latest TheRock multi-arch package (currently 7.14.1); the Windows MIGraphX CI build pins 7.13.0 | `repo.amd.com/rocm/tarball-multi-arch` | — |
| `amdmigraphx` | Build script of the MIGraphX EP in `onnxruntime-ep-amdgpu` (needs MIGraphX sources) | `develop` (same as the official build script) | git: `github.com/ROCm/AMDMIGraphX` | — |
| `vitisai` | AMD Vitis AI (Ryzen AI NPU) | Latest version of each package | Public AMD PyPI index `pypi.amd.com/simple` | sha256 |
| `android-ndk` | NNAPI, QNN / XNNPACK on Android | 28.0.13004108 (same as ORT `android.yml`) | `dl.google.com/android/repository` | sha1 |
| `onnxruntime-sdk` | All plugin EPs (headers and libraries as `ORT_HOME`); `--platform windows-arm64x` selects the `win-arm64x` package | Latest release (currently 1.30.0) | GitHub Releases | sha256 |
| `vulkan-sdk` | Vulkan plugin EP (provides glslc) | 1.3.296.0 (same as the plugin's CI; glslc output differs between versions) | `sdk.lunarg.com` (official LunarG) | sha256 |
| `tim-vx` | VeriSilicon VSINPU | Latest tag | git: `github.com/VeriSilicon/TIM-VX` | — |
| `zdnn` | IBM Telum plugin (`libzdnn` is dlopen'ed at runtime, s390x only) | Latest tag | git: `github.com/IBM/zDNN` | — |
| `compass-npu-driver` | Arm China Zhouyi NPU runtime driver (UMD/KMD sources) | `main` | git: `github.com/Arm-China/Compass_NPU_Driver` | — |

Notes:

- **CUDA** is downloaded as per-component redist archives: nvcc, cudart, cublas, cufft, curand, nvrtc, nvjitlink, cccl, and so on. Extract them into one directory and use it as `CUDA_HOME`.
- **TensorRT**: the README only lists x86_64 packages. For linux-aarch64 the script derives the link from the official naming scheme and checks that the file exists before downloading (ORT CI downloads it the same way).
- **TensorRT-RTX**: 1.4 and earlier have no aarch64 or Windows ARM64 packages; the default 1.6 covers both.
- **OpenVINO** downloads the `ubuntu24` package on Linux by default; `--openvino-distro` selects another distro. If the requested distro has no package for the architecture, another distro is used automatically (for example, arm64 currently only has `ubuntu22`).
- **CANN**: only the toolkit is downloaded. The per-chip Ascend operator packages (`*-ops_*.run`) are in the same directory if needed.
- **Vitis AI**: the three packages are versioned independently (for example, Ryzen AI 1.7 is voe 1.7.0 / onnxruntime-vitisai 1.23.2), so pin each one with `--pin <package>=<version>`. `onnxruntime-vitisai` installs into the same directory as `onnxruntime` from PyPI, so install it last.
- **Licenses**: most of these SDKs may not be redistributed. `packages/` is in `.gitignore`; use the downloads locally or temporarily in CI only.

### Not obtainable

- **Windows ML EP catalog**: on Windows 11, vendor plugin EPs can be downloaded on demand by the OS through Windows ML's `ExecutionProviderCatalog` (for example, QNN EP 2.7.40 is only distributed through WinML). This channel does not offer direct file downloads.
- **Xcode / Apple SDKs** (CoreML, MLX): only available from the App Store or `xcode-select --install`; there is no legitimate mirror.
- **Arm China Zhouyi EP** (`arm-china-compass-onnxruntime`): building it needs closed-source Compass toolchain libraries under `AIPU_TOOLKIT_DIR`, such as `libaipu_runtime`, `libaiputoolchain` and `libaipu_buildtool`. They are not part of the open-source Compass_NPU_Driver and can only be requested from Arm China technical support (Support-AI@armchina.com) or taken from a board vendor's system image, so they are excluded from the download script and CI.

### Reference: official prebuilt packages of each EP (not downloaded by the script)

| EP | PyPI | NuGet |
|---|---|---|
| OpenVINO plugin EP (factory code in `providers/openvino/ov_factory.cc`) | `onnxruntime-ep-openvino` | `Intel.ML.OnnxRuntime.EP.OpenVINO` |
| WebGPU plugin EP | `onnxruntime-ep-webgpu` | `Microsoft.ML.OnnxRuntime.EP.WebGpu` |
| TensorRT RTX plugin EP | `onnxruntime-ep-nv-tensorrt-rtx` (`-cu13` / `-cu12`) | — |
| QNN plugin EP | `onnxruntime-qnn` (2.x) | `Qualcomm.ML.OnnxRuntime.QNN` |
| MLX plugin EP | `onnxruntime-ep-mlx` | — |
| CUDA / TensorRT (in-tree) | `onnxruntime-gpu` | `Microsoft.ML.OnnxRuntime.Gpu.Windows` / `.Linux` |
| MIGraphX (in-tree) | `onnxruntime-migraphx` | — |
| CANN (in-tree) | `onnxruntime-cann` | — |
| DirectML (in-tree, sustained engineering) | `onnxruntime-directml` | `Microsoft.ML.OnnxRuntime.DirectML` |

## Runtime files of closed-source SDKs

The release packages contain only what this project builds. Providers / EPs that depend on a closed-source SDK whose license does not allow redistribution need that SDK's runtime libraries next to the ONNX Runtime binaries. [`scripts/fetch_ep_runtime.py`](scripts/fetch_ep_runtime.py) downloads those SDKs from the same no-account sources as the download script, extracts **only the runtime files**, and copies them into place together with the vendors' license texts.

```bash
# Point it at an extracted release package: the EPs, target triple, CUDA major version and the exact SDK
# versions used by the build are read from BUILD_INFO.json; files go into the package's bin/ (Windows) or lib/
python scripts/fetch_ep_runtime.py --package-dir onnxruntime-cuda13-<version>-x86_64-pc-windows-msvc

# Or choose EPs and a target explicitly
python scripts/fetch_ep_runtime.py --ep cuda tensorrt --target x86_64-unknown-linux-gnu --dest ./runtime
python scripts/fetch_ep_runtime.py --ep qnn --target aarch64-pc-windows-msvc --dry-run   # show the extraction rules only
python scripts/fetch_ep_runtime.py --list
```

| EP | SDK | Runtime files extracted |
|---|---|---|
| `cuda` (in-tree CUDA EP, CUDA plugin EP) | CUDA, cuDNN | cudart, cuBLAS / cuBLASLt, cuFFT, cuRAND, NVRTC, nvJitLink, all cuDNN libraries |
| `tensorrt` (in-tree TensorRT EP, `onnxruntime-ep-tensorrt`) | TensorRT, CUDA | `nvinfer*` (including builder resources, lean / dispatch / plugin libraries), `nvonnxparser`, plus cudart / cuBLAS / nvJitLink |
| `nv-tensorrt-rtx` (in-tree and plugin) | TensorRT-RTX, CUDA | `tensorrt_rtx_*`, `tensorrt_onnxparser_rtx_*`, plus cudart |
| `qnn` (in-tree QNN EP, `onnxruntime-ep-qnn`) | QAIRT | `Qnn*`, `Genie*` and `HtpPrepare` from the QAIRT directory of the target, plus the Hexagon `libQnnHtpV*Skel.so` / `.cat` files, following the same rules as onnxruntime-qnn's CMake |
| `vitisai` (Vitis AI bridge in the Windows x64 `onnxruntime` package) | Vitis AI runtime (`voe` wheel) | `onnxruntime_vitisai_ep.dll`, `dyn_dispatch_core.dll`, `aiecompiler_client.dll`, `onnxruntime_vitis_ai_custom_ops.dll` |
| `cann` (`onnxruntime-cann`) | CANN toolkit | Cannot be reduced to a few files; the vendor installer is run into `<dest>/cann` (source its `set_env.sh` before use) |

Targets: `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc`, `arm64ec-pc-windows-msvc`, `arm64x-pc-windows-msvc`, `aarch64-linux-android`. The script writes `THIRD_PARTY_RUNTIME.json` (versions, sources and extracted files) and copies the license texts into `third_party_licenses/`. Archives are only streamed, never fully unpacked; downloads are cached in `packages/` and verified when the upstream provides a checksum. The extracted files remain subject to their vendors' licenses: use them, but do not redistribute them.

Not covered, because they are open source or may be redistributed: OpenVINO (Apache-2.0), ROCm / MIGraphX, DirectML (Microsoft's redistributable package), Vulkan / IREE / MLX dependencies. GPU / NPU drivers must be installed separately as before.

## Continuous integration (GitHub Actions)

[`.github/workflows/build.yml`](.github/workflows/build.yml) builds the whole project with a matrix and packages the outputs per **target triple**.

### Jobs

| Job | Purpose |
|---|---|
| `plan` | Runs [`ci/plan.py`](ci/plan.py): generates the build matrix from [`ci/matrix.json`](ci/matrix.json), walks every recipe in dry-run mode to collect the third-party SDKs it needs, and computes the version |
| `build` | One job per matrix entry: checks out only the submodules it needs (shallow), [`ci/build.py`](ci/build.py) downloads the SDKs, builds and stages `bin/ lib/ include/`, and [`ci/package.py`](ci/package.py) packs it as `<package>-<version>-<target-triple>.tar.gz/.zip` with a `.sha256` and `BUILD_INFO.json` |
| `sdk-archive` | **Private repositories only**: uploads one `third-party-sdk-<platform>-<component>[-cuda<N>][-<pinned version>]` artifact per SDK the builds use, visible to repository members only and kept for 7 days by default |
| `release` | On a `v*` tag push, or a manual run with `publish` checked, publishes all packages and `SHA256SUMS` as a GitHub Release (publicly downloadable when the repository is public) |

### Build matrix

85 entries in total. The target platforms follow what each repository's own CI/CD publishes (the ORT v1.30.0 release assets, PyPI wheel tags, and the plugin repositories' workflows).

| Package | Contents | Target triples |
|---|---|---|
| `onnxruntime` | CPU + XNNPACK; plus oneDNN and the Vitis AI bridge on x64; plus CoreML on macOS | `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc`, `aarch64-apple-darwin`, `x86_64-apple-darwin` |
| `onnxruntime` (ARM64X) | Same as the official `win-arm64x` package: an ARM64 pass plus an ARM64EC pass with `--buildasx` | `arm64x-pc-windows-msvc` |
| `onnxruntime` (Apple frameworks) | Dynamic `onnxruntime.framework` per sysroot/arch, built with the official `build_apple_framework.py` settings (CoreML, XNNPACK on iOS) | `aarch64-apple-ios`, `aarch64-apple-ios-sim`, `x86_64-apple-ios`, `aarch64-apple-ios-macabi`, `x86_64-apple-ios-macabi`, `aarch64-apple-tvos` ⚠, `aarch64-apple-tvos-sim` ⚠, `x86_64-apple-tvos` ⚠, `aarch64-apple-visionos` ⚠, `aarch64-apple-visionos-sim` ⚠ |
| `onnxruntime-all` | Every built-in EP the platform supports: EPs compiled into `onnxruntime` plus the shared providers loaded through `onnxruntime_providers_shared` (plugin EPs are not included). Linux x64: XNNPACK, WebGPU, oneDNN, Vitis AI, CUDA 13, TensorRT, TensorRT RTX, QNN, CANN, OpenVINO. Linux aarch64: XNNPACK, WebGPU, CUDA 13, TensorRT, TensorRT RTX, QNN, CANN, OpenVINO. Windows x64: XNNPACK, WebGPU, DirectML, oneDNN, Vitis AI, CUDA 13, TensorRT, TensorRT RTX, QNN, OpenVINO. Windows ARM64: XNNPACK, WebGPU, DirectML, CUDA 13, TensorRT RTX, QNN. macOS arm64: XNNPACK, WebGPU, CoreML, OpenVINO. macOS x86_64: XNNPACK, WebGPU, CoreML. MIGraphX stays in `onnxruntime-migraphx` (it needs the ROCm container); VSINPU / ACL / ArmNN / RKNPU are left out because they link vendor libraries into `onnxruntime` itself | `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc` ⚠, `aarch64-apple-darwin`, `x86_64-apple-darwin` |
| `onnxruntime-android` | NNAPI + XNNPACK + WebGPU (same EP set as the official AAR) | `aarch64-linux-android`, `armv7-linux-androideabi`, `i686-linux-android`, `x86_64-linux-android` |
| `onnxruntime-wasm` / `-wasm-jsep` / `-wasm-webgpu` / `-wasm-webgpu-jspi` / `-wasm-webgpu-static-plugin` | WebAssembly outputs of onnxruntime-web (`.wasm` / `.mjs`), same variants as ORT's `web.yml` | `wasm32-unknown-emscripten` |
| `onnxruntime-wasm-static` | WebAssembly static library `libonnxruntime_webassembly.a` plus the C/C++ API headers | `wasm32-unknown-emscripten` |
| `onnxruntime-directml` | DirectML | `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc` |
| `onnxruntime-qnn` | In-tree QNN EP (the official `win-arm64` package still ships it) | `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc` |
| `onnxruntime-cuda12` | CUDA 12 + TensorRT | `x86_64-unknown-linux-gnu`, `x86_64-pc-windows-msvc` |
| `onnxruntime-cuda13` | CUDA 13 + TensorRT | `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `x86_64-pc-windows-msvc` |
| `onnxruntime-ep-cuda12` | CUDA plugin EP, CUDA 12 | `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `x86_64-pc-windows-msvc` |
| `onnxruntime-ep-cuda13` | CUDA plugin EP, CUDA 13 | `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc` |
| `onnxruntime-nv-tensorrt-rtx` | TensorRT RTX (in-tree) | `x86_64-unknown-linux-gnu`, `x86_64-pc-windows-msvc` |
| `onnxruntime-ep-webgpu` | WebGPU plugin EP | `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc`, `aarch64-apple-darwin` |
| `onnxruntime-openvino` | OpenVINO (main repository) | `x86_64-unknown-linux-gnu`, `x86_64-pc-windows-msvc` |
| `onnxruntime-openvino-intel` | OpenVINO (Intel fork `ovep-develop`) | Same as above |
| `onnxruntime-migraphx` | MIGraphX (built in the `rocm/dev-ubuntu-24.04:7.2.4-complete` container) | `x86_64-unknown-linux-gnu` |
| `onnxruntime-cann` ⚠ | Huawei CANN | `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu` |
| `onnxruntime-vsinpu` ⚠ | VeriSilicon VSINPU (TIM-VX with the x86 simulator driver) | `x86_64-unknown-linux-gnu` |
| `onnxruntime-ep-tensorrt` | TensorRT plugin EP | `x86_64-unknown-linux-gnu`, `x86_64-pc-windows-msvc` |
| `onnxruntime-ep-nv-tensorrt-rtx-cuda13` | TensorRT RTX plugin EP (NVIDIA), CUDA 13 | `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc` ⚠ (cross-compiled on x64) |
| `onnxruntime-ep-nv-tensorrt-rtx-cuda12` | TensorRT RTX plugin EP (NVIDIA), CUDA 12 | `x86_64-unknown-linux-gnu`, `x86_64-pc-windows-msvc` |
| `onnxruntime-ep-directml-amd` | DirectML EP from the AMD plugin repository | `x86_64-pc-windows-msvc` |
| `onnxruntime-ep-migraphx-amd` | MIGraphX EP from the AMD plugin repository | `x86_64-unknown-linux-gnu` (ROCm container), `x86_64-pc-windows-msvc` ⚠ (MIGraphX built from source, see below) |
| `onnxruntime-ep-amdgpu` | AMDGPU EP (`--use_amdgpu`; includes the MIGraphX backend, plus the DirectML backend on Windows) | `x86_64-unknown-linux-gnu` (ROCm container), `x86_64-pc-windows-msvc` ⚠ |
| `onnxruntime-ep-qnn` | Qualcomm QNN plugin EP | `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc`, `arm64ec-pc-windows-msvc` ⚠, `arm64x-pc-windows-msvc` ⚠, `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu` ⚠, `aarch64-linux-android` ⚠ |
| `onnxruntime-ep-mlx` | MLX plugin EP (with the MIT-licensed libmlx) | `aarch64-apple-darwin` |
| `onnxruntime-ep-iree` | IREE plugin EP (with a wheel) | `x86_64-unknown-linux-gnu`, `x86_64-pc-windows-msvc`, `aarch64-apple-darwin` |
| `onnxruntime-ep-telum` | IBM Telum plugin EP (cross-compiled on x86_64) | `s390x-unknown-linux-gnu` |
| `onnxruntime-ep-vulkan` | Vulkan plugin EP | `x86_64-unknown-linux-gnu`, `x86_64-pc-windows-msvc` |

⚠ marks experimental entries (`experimental: true`); their failures do not fail the run or block the release. `arm64ec-pc-windows-msvc` and `arm64x-pc-windows-msvc` are not official Rust triples; they name the Windows ARM64EC and ARM64X binaries.

Not built in CI:

- The Zhouyi EP: its closed-source toolchain is not publicly available (see above).
- QNN's `build_ort_linux_aarch64_oe_gcc11_2` task: its OE GCC 11.2 toolchain is hosted on an internal Qualcomm server.
- Obsolete or deprecated EPs.
- Language bindings (Python wheels, NuGet, Maven, npm, CocoaPods): the packages contain binaries, libraries and headers only.

### Windows MIGraphX

There is no prebuilt MIGraphX for Windows: AMD's MIGraphX wheels (`stable.repo.amd.com/rocm/migraphx/whl-next`) are Linux-only, and on Windows the MIGraphX EP is otherwise only distributed through the Windows ML EP catalog. The Windows `onnxruntime-ep-migraphx-amd` and `onnxruntime-ep-amdgpu` jobs therefore follow AMDMIGraphX's own `windows.yaml` CI:

1. ROCm comes from the TheRock Windows SDK tarball (`rocm` component, pinned to 7.13.0 like the upstream CI).
2. The AMDMIGraphX sources (`amdmigraphx` component, `develop`) are built with `rbuild` (`gh-win-gpu` profile; `clang-cl` for the dependencies including rocMLIR, the ROCm `clang++` for MIGraphX itself).
3. The AMD plugin EP is then built against that MIGraphX install.

These jobs are heavy (rocMLIR is an LLVM-based build) and may exceed the 6-hour limit of GitHub-hosted runners; move them to a larger runner with `HEAVY_RUNNER_MAP`. The GPU architectures come from the `MIGRAPHX_GPU_TARGETS` repository variable (default `gfx1100;gfx1101;gfx1102;gfx1200;gfx1201`, i.e. RDNA3 / RDNA4, which the Windows ML MIGraphX EP requires).

### Usage

- **Triggers**: pushing a `v*` tag builds and publishes a release; pushing to `main` with changes in `ci/`, `scripts/`, submodules or the workflow only builds; a manual run (`workflow_dispatch`) accepts:
  - `filter`: only build entries matching this regex, e.g. `ep-qnn|wasm`, `onnxruntime-cuda13/x86_64` or `apple-ios`;
  - `cuda_archs`: CUDA architectures;
  - `publish`: publish a prerelease tagged `build-<version>`;
  - `sdk_archive` and `sdk_retention_days`: whether to create third-party SDK archives and for how many days to keep them.
- **Version**: tag builds use the tag without the leading `v`; other builds use `<ORT version>-dev.<date>.<commit>`.
- **CUDA architectures**: the default is `80-real;80-virtual`: SM 8.0 SASS plus a compute_80 PTX that the driver JIT-compiles on newer GPUs (ORT's CMake drops a list that has only `-virtual` entries). GitHub-hosted runners have 4 cores, and every extra CUDA architecture roughly doubles the build time, which easily exceeds the 6-hour job limit. To build several SASS architectures, change the `CUDA_ARCHITECTURES` repository variable or the manual-run input, and move the heavy jobs to larger runners with `HEAVY_RUNNER_MAP`, e.g. `{"ubuntu-24.04": "linux-16core", "windows-2025": "windows-16core"}`.
- **Third-party SDKs**: release packages contain only the libraries and headers built by this project, never third-party SDKs. `BUILD_INFO.json` in each package records the SDK versions used, and `scripts/fetch_ep_runtime.py --package-dir <package>` fetches the matching closed-source runtime files (see [Runtime files of closed-source SDKs](#runtime-files-of-closed-source-sdks)).
- **Visibility of SDK archives**: workflow artifacts of public GitHub repositories are visible to everyone, and most of these SDKs may not be redistributed, so `sdk-archive` only runs in private repositories; public repositories show a notice in the `plan` job and skip it.
- **Checking recipes locally**: `python ci/build.py --recipe ort --variant cuda --triple x86_64-unknown-linux-gnu --work w --stage s --sdk-root sdk --dry-run` prints the commands that would run; `python ci/plan.py` prints the matrix size and the version.
