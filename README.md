# onnxruntime-cicd

This repository collects the ONNX Runtime main repository and the Execution Provider (EP) implementations that are currently maintained:

- Every project with public git sources is included as a **git submodule** (shallow clone).
- SDKs and dependencies that cannot be fetched automatically at build time (CUDA, TensorRT, QAIRT, OpenVINO, CANN, ROCm, the Vitis AI runtime, ...) are downloaded into `packages/` by [`scripts/download_ep_packages.py`](scripts/download_ep_packages.py) from official sources that need no account; see [External dependencies and the download script](#external-dependencies-and-the-download-script).
- A configurator page on GitHub Pages turns a selection of target and EPs into a bundle request (an issue), and GitHub Actions builds one ready-to-use package with the headers, link files and libraries of ONNX Runtime and of every selected backend; see [Bundles](#bundles). Packages are built on demand from the [build recipes](#build-recipes) and reused while their sources do not change; there is no scheduled build of everything.

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
│   │   ├── onnxruntime-ep-mlx/          # Apple MLX plugin EP
│   │   ├── onnxruntime-ep-openvino/     # Intel OpenVINO plugin EP
│   │   └── onnxruntime-ep-neuropilot/   # MediaTek NeuroPilot plugin EP
│   └── vendor/                          # EP sources or forks maintained by hardware vendors
│       └── nvidia-tensorrt-rtx-ep/      # NVIDIA TensorRT RTX plugin EP
├── deps/
│   └── amdmigraphx/                     # ROCm/AMDMIGraphX (develop), built for the AMD plugin EPs
├── .github/
│   ├── workflows/bundle.yml             # bundle requests: builds the packages on demand and assembles the bundle
│   ├── workflows/pages.yml              # publishes site/ and catalog.json on GitHub Pages
│   ├── actions/build-entry/             # the build steps of one recipe (SDKs, ccache, build cache)
│   └── ISSUE_TEMPLATE/bundle.yml        # the bundle request form
├── ci/                                  # CI scripts: plan.py / build.py / sdk.py / package.py / matrix.json,
│                                        # catalog.py / bundle.py for bundle requests
├── site/                                # the bundle configurator (static page)
├── scripts/
│   ├── download_ep_packages.py          # downloads the external SDKs and dependencies EPs need
│   └── fetch_ep_runtime.py              # extracts the runtime / development files of closed-source SDKs
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

> The main repository and `onnxruntime-qnn` (a fork) are full ONNX Runtime source trees and therefore large. `.gitmodules` sets `shallow = true`. On Windows, run `git config --global core.longpaths true` first.

## Submodules

| Path | Upstream | Tracked branch | EP / target hardware | Maintainer | Notes |
|---|---|---|---|---|---|
| `onnxruntime` | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | `main` | All in-tree EPs (see below), plus the CUDA / WebGPU plugin EPs | Microsoft | Main repository |
| `eps/official/onnxruntime-qnn` | [onnxruntime/onnxruntime-qnn](https://github.com/onnxruntime/onnxruntime-qnn) | `main` | Qualcomm Snapdragon (QAIRT / HTP / NPU) | Qualcomm | Shipped as a standalone plugin EP since 2.0.0 (`onnxruntime-qnn>=2.0`) |
| `eps/official/onnxruntime-ep-tensorrt` | [onnxruntime/onnxruntime-ep-tensorrt](https://github.com/onnxruntime/onnxruntime-ep-tensorrt) | `main` | NVIDIA GPU (TensorRT 10/11) | ONNX Runtime team | Migrated from the in-tree TensorRT EP; requires ORT ≥ 1.23 |
| `eps/official/onnxruntime-ep-amdgpu` | [onnxruntime/onnxruntime-ep-amdgpu](https://github.com/onnxruntime/onnxruntime-ep-amdgpu) | `main` | AMD GPU / Instinct (AMDGPU, MIGraphX), Windows DX12 (DirectML) | AMD | Requires ORT ≥ 1.24; plugin versions of MIGraphX / DirectML |
| `eps/official/onnxruntime-ep-mlx` | [onnxruntime/onnxruntime-ep-mlx](https://github.com/onnxruntime/onnxruntime-ep-mlx) | `main` | Apple Silicon (MLX) | Community (moved into the onnxruntime organization) | PyPI package `onnxruntime-ep-mlx`; requires ORT ≥ 1.29 |
| `eps/official/onnxruntime-ep-openvino` | [onnxruntime/onnxruntime-ep-openvino](https://github.com/onnxruntime/onnxruntime-ep-openvino) | `main` | Intel CPU / GPU / NPU (OpenVINO) | Intel | Plugin version of the in-tree OpenVINO EP; requires OpenVINO ≥ 2026.4 and ORT ≥ 1.30 |
| `eps/official/onnxruntime-ep-neuropilot` | [onnxruntime/onnxruntime-ep-neuropilot](https://github.com/onnxruntime/onnxruntime-ep-neuropilot) | `main` | MediaTek NPU (NeuroPilot) on aarch64 Linux and Android | onnxruntime organization (MediaTek NeuroPilot) | Plugin EP; loads the closed-source prebuilt `libneuron_op_capability.so` from the repository's releases; requires ORT ≥ 1.24.1 |
| `eps/vendor/nvidia-tensorrt-rtx-ep` | [NVIDIA/TensorRT-RTX-EP-ABI](https://github.com/NVIDIA/TensorRT-RTX-EP-ABI) | `main` | NVIDIA RTX GPU (TensorRT-RTX) | NVIDIA | Plugin TensorRT RTX EP; PyPI package `onnxruntime-ep-nv-tensorrt-rtx` |
| `deps/amdmigraphx` | [ROCm/AMDMIGraphX](https://github.com/ROCm/AMDMIGraphX) | `develop` | MIGraphX itself, for the AMD plugin EPs | AMD | Built from source because the AMD plugin EPs follow its `develop` API; a submodule so that its version only changes with an update, which keeps the MIGraphX build cache valid |

## In-tree EPs of the main repository

Sources live in `onnxruntime/onnxruntime/core/providers/<directory>`.

| EP | Directory | Status | Notes / external implementation |
|---|---|---|---|
| CPU | `cpu` | Stable | Default EP, MLAS / KleidiAI kernels |
| CUDA | `cuda` | Stable | The plugin CUDA EP sources are in `cuda/plugin`, packaging in `plugin-ep-cuda/`, design docs in `docs/cuda_plugin_ep/` (plugin version 0.2 in v1.30) |
| TensorRT | `tensorrt` | Stable | Plugin version: `eps/official/onnxruntime-ep-tensorrt` |
| NVIDIA TensorRT RTX | `nv_tensorrt_rtx` | Stable | Plugin version: `eps/vendor/nvidia-tensorrt-rtx-ep` |
| DirectML | `dml` | Sustained engineering | New feature work moved to Windows ML; AMD's plugin version for AMD GPUs: `onnxruntime-ep-amdgpu/src/directx` |
| Intel OpenVINO | `openvino` | Stable | Development line: the `ovep-develop` branch of [intel/onnxruntime](https://github.com/intel/onnxruntime) (not included: the main repository has the same EP); plugin version: `eps/official/onnxruntime-ep-openvino` |
| Qualcomm QNN | `qnn` | In-tree version (provider bridge, 1.x) | New plugin EP: `eps/official/onnxruntime-qnn` |
| Apple CoreML | `coreml` | Preview | |
| Android NNAPI | `nnapi` | Stable | |
| XNNPACK | `xnnpack` | Stable | CPU acceleration for mobile / web |
| WebGPU | `webgpu` | Stable | Plugin packaging in `plugin-ep-webgpu/` (plugin version 0.4.0 in v1.30) |
| WebNN | `webnn` | Stable (web) | Used by onnxruntime-web |
| JSEP | `js` | Stable (web) | JavaScript EP of onnxruntime-web; front-end code in `js/` |
| AMD MIGraphX | `migraphx` | Stable | Plugin version: `eps/official/onnxruntime-ep-amdgpu/src/migraphx` |
| AMD Vitis AI | `vitisai` | Preview | The repository only contains a bridge layer; the closed-source backend is fetched by the download script (see below). Bundles offer it on Windows x64 only: AMD's public index has Windows wheels of the backend only |
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
   - Plugin EPs: the TensorRT / TensorRT RTX / AMD / OpenVINO / NeuroPilot plugins use FetchContent or vcpkg; the MLX plugin's `setup_mlx.sh` downloads mlx, mlx-c and metal-cpp; the Rust dependencies of the MLX plugin come from crates.io.
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
- Platforms: `linux-x86_64`, `linux-aarch64`, `windows-x86_64`, `windows-arm64`, `windows-arm64x`, `macos-arm64`; the default is the current host. Components without an upstream package for the platform are skipped with an explanation.
- If the GitHub API rate limit is hit, set the `GITHUB_TOKEN` environment variable (optional).

| Component | Used by | Default version | Source (no account needed) | Checksum |
|---|---|---|---|---|
| `cuda` | CUDA, TensorRT, TensorRT plugin, TensorRT RTX | Latest 13.x (`--cuda-major 12` switches to 12.x) | NVIDIA redist: `developer.download.nvidia.com/compute/cuda/redist` | sha256 |
| `cudnn` | CUDA | Latest 9.x | NVIDIA redist: `developer.download.nvidia.com/compute/cudnn/redist` | sha256 |
| `tensorrt` | TensorRT (in-tree), `onnxruntime-ep-tensorrt` | 10.15 (same as ORT CI); `latest` is 11.x | `developer.nvidia.com/downloads/...`, links taken from the README of each [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) release branch; requests from mainland China are redirected to `developer.download.nvidia.cn` | — |
| `tensorrt-rtx` | TensorRT RTX (in-tree + plugin) | 1.7 (the plugin's CMake requires ≥ 1.7 at configure time) | `rtx_sdk/secure/...` direct links listed on the [official TensorRT-RTX page](https://developer.nvidia.com/tensorrt-rtx) (verified to work without login) | — |
| `qairt` | QNN (in-tree + `onnxruntime-qnn`) | 2.51.0.260929 (from `onnxruntime-qnn/qcom/packages.yml`) | Qualcomm public download API `softwarecenter.qualcomm.com/api/download/...` (instead of QPM, which needs a Qualcomm ID) | sha256 |
| `hexagon-sdk` (optional) | QNN (only needed for custom HTP ops) | 6.5.0.0 (same source) | Same as above | sha256 |
| `openvino` | OpenVINO (in-tree), `onnxruntime-ep-openvino` | 2026.2.1 (same as ORT's CI); the plugin build pins 2026.4.1, the oldest it accepts | `storage.openvinotoolkit.org` | sha256 |
| `cann` | Huawei CANN | Latest release (currently 9.1.1) | Huawei Cloud Ascend artifactory `ascend.devcloud.huaweicloud.com/artifactory/cann-run` (instead of the official site, which needs a Huawei account) | sha256 |
| `rocm` | MIGraphX / AMDGPU (in-tree + `onnxruntime-ep-amdgpu`) | Latest TheRock multi-arch package (currently 7.14.1); the Windows MIGraphX CI build installs TheRock's 7.13.0 Python wheels instead | `repo.amd.com/rocm/tarball-multi-arch` | — |
| `rocm-wheels` | The ROCm runtime of the Windows MIGraphX / AMDGPU plugins (`fetch-restricted`) | The version the packages were built with (TheRock 7.13.0); `latest` for the newest | `repo.amd.com/rocm/whl-multi-arch`: `rocm-sdk-core`, `rocm-sdk-libraries` and `rocm-sdk-device-<gfx>` per GPU target | — |
| `hip-ep` | The HIP backend of `onnxruntime-ep-amdgpu-hip` (`fetch-restricted`) | The release the package was built with (its `BUILD_INFO.json` records it), else the latest release (currently 0.5.2, built with ROCm 7.16) | GitHub Releases of [ROCm/hip-ep](https://github.com/ROCm/hip-ep): `hipep_<version>_{windows,linux}.zip` | sha256 |
| `vitisai` | AMD Vitis AI (Ryzen AI NPU) | Latest version of each package | Public AMD PyPI index `pypi.amd.com/simple` | sha256 |
| `android-ndk` | NNAPI, QNN / XNNPACK on Android | 28.0.13004108 (same as ORT `android.yml`) | `dl.google.com/android/repository` | sha1 |
| `onnxruntime-sdk` | All plugin EPs (headers and libraries as `ORT_HOME`); `--platform windows-arm64x` selects the `win-arm64x` package | Latest release (currently 1.31.1); the QNN, TensorRT RTX and OpenVINO plugins use the release their own build names (currently 1.30.0), since their code follows its experimental API, which 1.31 changed | GitHub Releases | sha256 |
| `directml` | DirectML (in-tree + `onnxruntime-ep-amdgpu`) | 1.15.4 (same as `onnxruntime/packages.config` and the AMD plugin) | `api.nuget.org` (`Microsoft.AI.DirectML`) | — |
| `tim-vx` | VeriSilicon VSINPU | Latest tag | git: `github.com/VeriSilicon/TIM-VX` | — |
| `neuron-op-capability` | `onnxruntime-ep-neuropilot` (the prebuilt op capability library it builds against and loads at run time) | The release of the plugin version the submodule builds (the `project()` version of its CMakeLists), else the latest release | GitHub Releases of `onnxruntime/onnxruntime-ep-neuropilot` | sha256 |

Notes:

- **CUDA** is downloaded as per-component redist archives: nvcc, cudart, cublas, cufft, curand, nvrtc, nvjitlink, cccl, and so on. Extract them into one directory and use it as `CUDA_HOME`.
- **TensorRT**: the README only lists x86_64 packages. For linux-aarch64 the script derives the link from the official naming scheme and checks that the file exists before downloading (ORT CI downloads it the same way).
- **TensorRT-RTX**: 1.4 and earlier have no aarch64 or Windows ARM64 packages; the default 1.7 covers both.
- **OpenVINO** downloads the `ubuntu24` package on Linux by default; `--openvino-distro` selects another distro. If the requested distro has no package for the architecture, another distro is used automatically (for example, arm64 currently only has `ubuntu22`).
- **CANN**: only the toolkit is downloaded. The per-chip Ascend operator packages (`*-ops_*.run`) are in the same directory if needed.
- **Vitis AI**: the three packages are versioned independently (for example, Ryzen AI 1.7 is voe 1.7.0 / onnxruntime-vitisai 1.23.2), so pin each one with `--pin <package>=<version>`. `onnxruntime-vitisai` installs into the same directory as `onnxruntime` from PyPI, so install it last.
- **Licenses**: most of these SDKs may not be redistributed. `packages/` is in `.gitignore`; use the downloads locally or temporarily in CI only.

### Not obtainable

- **Windows ML EP catalog**: on Windows 11, vendor plugin EPs can be downloaded on demand by the OS through Windows ML's `ExecutionProviderCatalog` (for example, QNN EP 2.7.40 is only distributed through WinML). This channel does not offer direct file downloads.
- **Xcode / Apple SDKs** (CoreML, MLX): only available from the App Store or `xcode-select --install`; there is no legitimate mirror.
- **Arm China Zhouyi EP** ([Arm-China/Compass_Onnxruntime](https://github.com/Arm-China/Compass_Onnxruntime), a fork of ORT 1.22): building it needs closed-source Compass toolchain libraries under `AIPU_TOOLKIT_DIR`, such as `libaipu_runtime`, `libaiputoolchain` and `libaipu_buildtool`. They are not part of the open-source Compass_NPU_Driver and can only be requested from Arm China technical support (Support-AI@armchina.com) or taken from a board vendor's system image. GitHub Actions cannot build it, so it is not a submodule.

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

## Build recipes

Every package is built from [`ci/matrix.json`](ci/matrix.json): the core of a bundle from the core recipe of its target triple with the in-tree EPs the request selected, a plugin EP package from its entry (a recipe and variant of [`ci/build.py`](ci/build.py) for one target triple), through the [build steps](.github/actions/build-entry/action.yml) shared by all builds: check out only the submodules the entry needs (shallow), download the third-party SDKs (`ci/sdk.py`), build and stage `bin/ lib/ include/`, and pack it with [`ci/package.py`](ci/package.py) as `<package>-<inputs hash>-<target-triple>.tar.gz/.zip` with a `.sha256` and `BUILD_INFO.json`. Nothing builds them on a schedule: a [bundle request](#bundles) builds the packages it needs, and later requests reuse them (see [Builds on demand and caching](#builds-on-demand-and-caching)).

### Core recipes

`cores` in `ci/matrix.json` has one recipe per target triple: the build platform (the `ci/build.py` variant), the runner and the in-tree EPs a core can be built with there, each with its build requirements (packages, container, the CUDA major versions it was built with, long build, experimental). A bundle's core always contains the CPU EP plus the in-tree EPs the request compiles into it; cores with the same EP set are reused.

| Target | Platform (variant) | In-tree EPs a core can contain |
|---|---|---|
| `x86_64-unknown-linux-gnu` | desktop (`custom`) | XNNPACK, WebGPU, CUDA (12/13), TensorRT (12/13), TensorRT RTX (13), oneDNN, OpenVINO, MIGraphX (in the `rocm/dev-ubuntu-24.04:7.2.4-complete` container), CANN, VSINPU ⚠ (TIM-VX with the x86 simulator driver) |
| `aarch64-unknown-linux-gnu` | desktop | XNNPACK, WebGPU, CUDA (13), TensorRT (13), CANN |
| `x86_64-pc-windows-msvc` | desktop | XNNPACK, DirectML, WebGPU, CUDA (12/13), TensorRT (12/13), TensorRT RTX (13), oneDNN, OpenVINO, Vitis AI, QNN |
| `aarch64-pc-windows-msvc` | desktop | XNNPACK, DirectML, WebGPU, QNN |
| `aarch64-apple-darwin`, `x86_64-apple-darwin` | desktop (x86_64 cross-compiled on arm64) | XNNPACK, CoreML, WebGPU |
| `arm64x-pc-windows-msvc`, `arm64ec-pc-windows-msvc` | `arm64x`: an ARM64 pass plus an ARM64EC pass with `--buildasx`, like the official `win-arm64x` package; the ARM64EC target uses the ARM64X core | CPU only |
| `aarch64-apple-ios`, `aarch64-apple-ios-sim`, `x86_64-apple-ios` | Apple framework (`ort-apple`): a dynamic `onnxruntime.framework` per sysroot/arch with ORT's `build_apple_framework.py`, whose settings get the selected EPs | CoreML, XNNPACK |
| `aarch64-apple-ios-macabi`, `x86_64-apple-ios-macabi`, `*-apple-tvos*` ⚠, `*-apple-visionos*` ⚠ | Apple framework | CoreML (ORT's settings build XNNPACK for iOS only) |
| `aarch64-linux-android`, `armv7-linux-androideabi`, `i686-linux-android`, `x86_64-linux-android` | `android` | NNAPI, XNNPACK, WebGPU |
| `wasm32-unknown-emscripten` | one of five exclusive build modes, the same as ORT's `web.yml`: CPU, JSEP (+ WebNN), WebGPU (+ WebNN), WebGPU with JSPI (+ WebNN), or the static library `libonnxruntime_webassembly.a` | given by the mode |

### Plugin EP packages

`plugins` in `ci/matrix.json`:

| Package | Contents | Target triples |
|---|---|---|
| `onnxruntime-ep-cuda12` | CUDA plugin EP, CUDA 12 | `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `x86_64-pc-windows-msvc` |
| `onnxruntime-ep-cuda13` | CUDA plugin EP, CUDA 13 | `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc` |
| `onnxruntime-ep-webgpu` | WebGPU plugin EP | `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc`, `aarch64-apple-darwin` |
| `onnxruntime-ep-tensorrt` | TensorRT plugin EP | `x86_64-unknown-linux-gnu`, `x86_64-pc-windows-msvc` |
| `onnxruntime-ep-nv-tensorrt-rtx-cuda13` | TensorRT RTX plugin EP (NVIDIA), CUDA 13 | `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc` ⚠ (cross-compiled on x64) |
| `onnxruntime-ep-nv-tensorrt-rtx-cuda12` | TensorRT RTX plugin EP (NVIDIA), CUDA 12 | `x86_64-unknown-linux-gnu`, `x86_64-pc-windows-msvc` |
| `onnxruntime-ep-directml-amd` | DirectML EP from the AMD plugin repository | `x86_64-pc-windows-msvc` |
| `onnxruntime-ep-migraphx-amd` | MIGraphX EP from the AMD plugin repository | `x86_64-unknown-linux-gnu`, `x86_64-pc-windows-msvc` ⚠ (MIGraphX built from source on both, see below) |
| `onnxruntime-ep-amdgpu` | AMDGPU EP (`--use_amdgpu`, `USE_HIP=OFF`; includes the MIGraphX backend, plus the DirectML backend on Windows) | `x86_64-unknown-linux-gnu`, `x86_64-pc-windows-msvc` ⚠ |
| `onnxruntime-ep-amdgpu-hip` | The same with its HIP backend (`hip-backend`, a shim that loads the prebuilt `hipgpu` of ROCm/hip-ep, which `fetch-restricted` downloads) | `x86_64-unknown-linux-gnu` ⚠, `x86_64-pc-windows-msvc` ⚠ |
| `onnxruntime-ep-qnn` | Qualcomm QNN plugin EP | `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc`, `arm64ec-pc-windows-msvc` ⚠, `arm64x-pc-windows-msvc` ⚠, `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu` ⚠, `aarch64-linux-android` ⚠ |
| `onnxruntime-ep-mlx` | MLX plugin EP (with the MIT-licensed libmlx) | `aarch64-apple-darwin` |
| `onnxruntime-ep-openvino` | OpenVINO plugin EP, built with OpenVINO 2026.4.1 | `x86_64-unknown-linux-gnu` ⚠, `x86_64-pc-windows-msvc` ⚠ |
| `onnxruntime-ep-neuropilot` | MediaTek NeuroPilot plugin EP, built with ORT's headers from the `onnxruntime` submodule | `aarch64-unknown-linux-gnu` ⚠, `aarch64-linux-android` ⚠ (API level 29) |

⚠ marks experimental entries (`experimental: true`): known to fail at times, or not verified yet. The OpenVINO plugin DLL declares the OpenVINO runtime as a side-by-side private assembly; its Windows package contains the `openvino_runtime.manifest` that describes it, while `fetch-restricted` puts the unmodified OpenVINO DLLs next to it (the upstream install embeds manifests into re-signed copies instead). `arm64ec-pc-windows-msvc` and `arm64x-pc-windows-msvc` are not official Rust triples; they name the Windows ARM64EC and ARM64X binaries.

Not built:

- The Zhouyi EP: its closed-source toolchain is not publicly available (see above).
- QNN's `build_ort_linux_aarch64_oe_gcc11_2` task: its OE GCC 11.2 toolchain is hosted on an internal Qualcomm server.
- Obsolete or deprecated EPs.
- Language bindings (Python wheels, NuGet, Maven, npm, CocoaPods): the packages contain binaries, libraries and headers only.

### MIGraphX for the AMD plugin EPs

The AMD plugin EPs follow AMDMIGraphX `develop`, whose API is newer than any ROCm release, so MIGraphX is built from source. On Linux this happens in AMDMIGraphX's own CI image `rocm/migraphx-ci-ubuntu` (ROCm in `/opt/rocm`, dependencies preinstalled), like its `ci.yaml`.


There is no prebuilt MIGraphX for Windows: AMD's MIGraphX wheels (`stable.repo.amd.com/rocm/migraphx/whl-next`) are Linux-only, and on Windows the MIGraphX EP is otherwise only distributed through the Windows ML EP catalog. The Windows `onnxruntime-ep-migraphx-amd` and `onnxruntime-ep-amdgpu` jobs therefore follow AMDMIGraphX's own `windows.yaml` CI:

1. ROCm comes from TheRock's Python wheels (`rocm[libraries,devel,device-<gfx>]==7.13.0`, like the upstream CI); unlike the SDK tarball, the device packages also contain the device libraries HIP code needs.
2. The AMDMIGraphX sources (the `deps/amdmigraphx` submodule) are built with `rbuild` (`gh-win-gpu` profile; `clang-cl` for the dependencies including rocMLIR, the ROCm `clang++` for MIGraphX itself).
3. The AMD plugin EP is then built against that MIGraphX install.

A first build takes longer than the 6-hour job limit of a hosted runner (rocMLIR alone is an LLVM-based build of about 3.5 hours, MIGraphX itself about 2 hours), so the bundle workflow builds MIGraphX in two jobs of its own before these packages (`ci/build.py --recipe migraphx --variant deps|build`): `migraphx-deps` builds the dependencies, `migraphx` builds MIGraphX, and each saves its result in the build cache, from which the packages restore it; a stage whose result is already cached ends at once. A larger runner (`HEAVY_RUNNER_MAP`) shortens them. The GPU targets (`GPU_TARGETS`) are part of each bundle request (default `gfx1100;gfx1101;gfx1102;gfx1200;gfx1201`, i.e. RDNA3 / RDNA4, which the Windows ML MIGraphX EP requires); on Windows only the Radeon targets are offered, because TheRock has no Windows device libraries for the Instinct ones. The build records the ROCm release it used in `BUILD_INFO.json` (`third_party_sdks.rocm`).

### Builds on demand and caching

- **Package reuse**: [`ci/plan.py`](ci/plan.py) hashes the inputs of every package build: the entry, the commits of its submodules, the build scripts (`ci/build.py`, `ci/sdk.py`, `ci/package.py`, `scripts/`, the build-entry action), the resolved third-party SDK versions, the Docker image digest, and the CUDA architectures or AMD GPU targets of the request when the package compiles device code for them (the CUDA core and plugin, the TensorRT plugin, the MIGraphX and AMDGPU plugins; other packages are shared by requests with different GPU choices). The package artifact is named `pkg-<package>-<triple>-<hash>` and kept for 90 days; a bundle request whose package has the same hash downloads that artifact instead of building it. Changes to the bundle and page code (`ci/bundle.py`, `ci/catalog.py`, `site/`) do not invalidate packages. Cores of different EP sets of a target share one ccache directory (ccache key per target and CUDA major version), which helps only with the third-party code: ORT adds the `USE_<EP>` definitions of the selected EPs to every ORT source file. `python ci/plan.py` lists every plugin entry and the default core of every target with its hash and whether an artifact can be reused.
- **Compiler cache**: the CUDA and MIGraphX builds compile through ccache ([`hendrikmuhs/ccache-action`](https://github.com/hendrikmuhs/ccache-action), up to 2 GB per entry). CMake reads `CMAKE_C_COMPILER_LAUNCHER`, `CMAKE_CXX_COMPILER_LAUNCHER` and `CMAKE_CUDA_COMPILER_LAUNCHER` from the environment, so ONNX Runtime, MIGraphX with its dependencies and the EP repositories all use it (not the Visual Studio generator builds, nor container builds). A first build gains nothing; a build after a submodule update recompiles only what changed. The repository has 10 GB of Actions cache; the least recently used entries are evicted, and entries unused for 7 days expire.
- **MIGraphX on Windows**: the dependencies `rbuild prepare` builds (rocMLIR and others) are cached on their own by the `migraphx-deps` job, keyed by the dependency inputs (MIGraphX's requirement files, the ROCm version, the MSVC toolset); the `migraphx` job caches them with the MIGraphX install once it is complete, keyed in addition by the `deps/amdmigraphx` commit and the GPU targets. The dependencies do not depend on the GPU targets, so a new MIGraphX commit or another set of targets restores an entry of the same dependency inputs and rebuilds only MIGraphX; the same commit and targets reuse the install, which the `ep-migraphx-amd` and `ep-amdgpu` builds share. A package build that restores MIGraphX installs ROCm without the device libraries of the GPU targets, which only the MIGraphX build needs.
- **Third-party SDKs** are not cached: they take a few minutes at most, would exceed the cache quota, and caches of a public repository can be restored by pull-request workflows, which the SDK licenses do not allow.
- **Dependency updates**: Dependabot (`.github/dependabot.yml`) opens weekly pull requests that update the actions and move the submodules to the newest commits of their branches. Merging one builds nothing; the next request that needs an affected package builds it.
- **CUDA architectures**: each request selects the SM versions (`CMAKE_CUDA_ARCHITECTURES`, `<sm>-real` per version, optionally `<newest>-virtual` for PTX); the default is `80-real;80-virtual`: SM 8.0 SASS plus a compute_80 PTX that the driver JIT-compiles on newer GPUs (ORT's CMake drops a list that has only `-virtual` entries). GitHub-hosted runners have 4 cores, and every extra CUDA architecture lengthens the CUDA builds considerably, which easily exceeds the 6-hour job limit. For several SASS architectures, move the heavy builds to larger runners with `HEAVY_RUNNER_MAP`, e.g. `{"ubuntu-24.04": "linux-16core", "windows-2025": "windows-16core"}` (`RUNNER_MAP` replaces runners for every build).
- **macOS deployment target**: the desktop macOS packages are built with `MACOSX_DEPLOYMENT_TARGET=14.0`, like ORT's official macOS packages, so they run on macOS 14 and later rather than only on the runner's macOS version.
- **Checking recipes locally**: `python ci/build.py --recipe ort --variant custom --eps tensorrt,cuda,openvino --triple x86_64-unknown-linux-gnu --work w --stage s --sdk-root sdk --dry-run` prints the commands that would run.

## Bundles

A **bundle** is one package for one target triple with exactly the EPs a user selected, composed EP by EP: one core library with the in-tree EPs compiled into it plus the plugin EP packages of the other EPs, merged into a single `bin/ lib/ include/` tree together with the third-party runtimes, headers and link files of the selected backends, a CMake package and an example. Bundles are requested from a static configurator page and built by GitHub Actions.

```
site/ (GitHub Pages) --prefilled issue--> issue "[bundle] ..." --issues event--> .github/workflows/bundle.yml
  ^ catalog.json (ci/catalog.py, from ci/matrix.json)                              resolve -> build (packages not
  pages.yml <- push to main                                                        reused) -> assemble -> artifact link
```

### Configurator page

[`site/`](site/) is published by [`.github/workflows/pages.yml`](.github/workflows/pages.yml) (set **Settings → Pages → Source** to *GitHub Actions*) together with `catalog.json`, which [`ci/catalog.py`](ci/catalog.py) generates from the build recipes: per target, the in-tree EPs its core can contain (with their constraints) and the plugin EP packages. The page is redeployed when it or the recipes change.

1. **Target**: platform and architecture (every target triple of the matrix).
2. **Execution providers**: one row per EP the target offers, each with **Off / Plugin EP / In core**; the EPs the target offers in the core only come first. CPU is always in the core.
   - **Plugin EP** adds the EP's plugin EP package (`onnxruntime-ep-*`): a separate library registered at run time, built once per version and reused by every bundle.
   - **In core** compiles the EP into the core library, as built-in code or as a provider library next to it (oneDNN, Vitis AI, CANN, XNNPACK, CoreML, NNAPI, VSINPU have only this form; the others where the target's core recipe supports them). When the page opens, the core has XNNPACK where the target supports it.
   - **AMD GPU** is one row for AMD's plugin packages, offered in three versions: *AMDGPU* (the default: one EP with the MIGraphX and DirectML (Windows) backends that picks one per GPU; it always loads the MIGraphX backend and so needs ROCm), *MIGraphX* only, or *DirectML* only (Windows; AMD's port of the DirectML EP for AMD GPUs, needing no ROCm and no MIGraphX build). The *HIP backend* checkbox of the AMDGPU version (off by default, experimental) picks `onnxruntime-ep-amdgpu-hip`, whose HIP backend runs LLM sessions of onnxruntime-genai on Ryzen AI 300 / Max with the prebuilt HIP EP of ROCm/hip-ep. Without it the AMDGPU EP is built with `USE_HIP=OFF`: an AMDGPU EP built with the HIP backend fails to load when `hipgpu` is missing. On Windows hip-ep brings the ROCm runtime it is built with (7.16), newer than the one of MIGraphX (7.13), and that one then serves the whole process; the two are not tested together. Its *In core* form is ORT's own MIGraphX EP (Linux). The **DirectML** row is ORT's in-tree EP for any DirectX 12 GPU.
   - The page offers only what works: a dash marks a form the target does not offer, and a button that also changes other EPs says so before it is clicked (TensorRT in the core brings the CUDA EP into the core, since ORT builds them together; moving CUDA out of the core moves TensorRT to its plugin; a choice built for the other CUDA major version switches to it). The rows a click changed are highlighted with the reason. Switching the target keeps the choices the new target offers and names the ones it drops. Notes explain long builds and untested combinations (MIGraphX in the core builds the whole core in the ROCm container).
   - **Linkage** of an EP in the core, on the targets ORT's CI builds it for: WebGPU on Windows and Linux links Dawn statically (default) or as the shared `webgpu_dawn` library next to the core (`onnxruntime_BUILD_DAWN_SHARED_LIBRARY`); QNN on Windows is a provider library `onnxruntime_providers_qnn` (default, `--use_qnn shared_lib`) or linked into the core library (`--use_qnn static_lib`). A plugin EP is always a separate library, and ORT does not support a shared Dawn with the WebGPU plugin EP, so the choice disappears (back to the default) when the EP moves to its plugin. Each linkage builds its own core package.
   - WebAssembly offers the five build modes instead (single choice), because they are exclusive builds.
   - *Reset* goes back to the defaults.
3. **GPU code**, when the selection needs CUDA or a selected package compiles device code: the CUDA major version (only the versions the selection can be built with, or the reason when there is one), the CUDA SM versions (with the GPUs of each and how every listed GPU runs the selection: its own kernels, the kernels of an older SM version of the same major version, the PTX, or not at all) and whether to add PTX; the AMD GPU targets MIGraphX is built for (on Windows only the Radeon targets).
4. **Request**: the EPs and how each one is provided, which packages take long to build, the vendor SDK files `fetch-restricted` downloads (each runtime with its headers and import libraries, with the download size) and what to install yourself, with links to the licenses, the **requirements** on the user's machine (OS and C/C++ runtime baseline: glibc and libstdc++ of Ubuntu 24.04 on Linux, the Visual C++ v14 Redistributable on Windows, the deployment targets on Apple platforms, the Android API level; the CUDA driver, ROCm and oneTBB where the selection needs them), the **versions** the packages are built from (ONNX Runtime and the EP repositories at their submodule commits, and the third-party SDK versions resolved when the page was generated), and the license acknowledgement. *Create bundle request* opens a prefilled issue (`.github/ISSUE_TEMPLATE/bundle.yml`).

### Building a bundle

[`.github/workflows/bundle.yml`](.github/workflows/bundle.yml) runs for issues titled `[bundle] ...` whose author is a collaborator of the repository (`OWNER`, `MEMBER` or `COLLABORATOR`); everyone else's issues are ignored. Editing the issue builds it again (reusing the packages that exist); a collaborator's comment starting with `/rebuild` builds the packages and the bundle again without reusing anything.

| Job | Purpose |
|---|---|
| `resolve` | [`ci/bundle.py`](ci/bundle.py) `resolve` reads the configuration from the issue, validates it against the catalog (the same rules as the page) and plans every package with [`ci/plan.py`](ci/plan.py): its build job and inputs hash, and the artifact of an earlier run it can reuse. The status comment in the issue shows which packages are reused and which are built. An unexpired bundle with the same configuration and package hashes is linked instead of building it again |
| `build` | One job per package that cannot be reused (except the Windows AMD packages), through the shared [build steps](.github/actions/build-entry/action.yml) ([`build-package`](.github/actions/build-package/action.yml) adds packing and the upload); the core is built from the core recipe of its target with the selected in-tree EPs (`ci/build.py --variant custom --eps ...` on desktop targets), on its runner, with the packages and container its EPs need. Each package is kept as an artifact for 90 days, also when another package of the request fails |
| `migraphx-deps`, `migraphx` | Only when Windows AMD packages are built: the dependencies of MIGraphX, then MIGraphX itself, one job each per target and set of GPU targets (see [MIGraphX for the AMD plugin EPs](#migraphx-for-the-amd-plugin-eps)) |
| `build-amd` | The Windows AMD packages, after the MIGraphX jobs: they restore MIGraphX from the build cache and build only the EPs |
| `assemble` | Downloads the packages (`gh run download`, checked against their SHA-256), merges them, writes the CMake package, `plugins.json`, the example, `README.md` and `BUNDLE_INFO.json`, uploads the bundle as an artifact and links it in the issue, which is then closed |

The bundle artifact is kept for 90 days; downloading it needs a GitHub login. It contains `onnxruntime-bundle-<key>-<triple>.zip` (Windows) or `.tar.gz` with:

```
include/onnxruntime/      ONNX Runtime C/C++ API
include/<backend>/        backend headers: migraphx/, hip/, mlx/ from the packages; cuda/ (with cccl/), cudnn/,
                          tensorrt/, tensorrt_rtx/, qnn/, openvino/, DirectML.h from fetch-restricted
lib/                      import libraries (Windows) or shared libraries; lib/cmake/onnxruntime-bundle/
bin/                      DLLs (Windows)
plugins.json              plugin EP libraries to register with RegisterExecutionProviderLibrary
packages/<package>/       LICENSE, ThirdPartyNotices.txt and BUILD_INFO.json of every package
third_party_licenses/     licenses of the bundled third-party components
examples/                 CMake example that registers the plugin EPs and lists the EPs
fetch-restricted.sh/.ps1  only when the selection needs vendor SDK files (see below)
rocm/                     Windows AMD plugins: the ROCm runtime fetch-restricted unpacks (TheRock's wheels)
BUNDLE_INFO.json          configuration, packages, EPs, third-party parts, files that differed between packages, and on
                          Linux the glibc / libstdc++ versions its libraries need (linux_libraries)
README.md                 contents, files to fetch, requirements (also the measured glibc / libstdc++ versions on Linux)
```

```cmake
find_package(onnxruntime-bundle CONFIG REQUIRED PATHS "<bundle>/lib/cmake")
target_link_libraries(app PRIVATE onnxruntime::onnxruntime TensorRT::nvinfer CUDA::cudart)
onnxruntime_bundle_copy_runtime(app)   # copies the DLLs next to app on Windows
```

The CMake package defines `onnxruntime::onnxruntime` and a target per backend whose files are present (`CUDA::cudart`, `CUDA::cublas`, `CUDNN::cudnn`, `TensorRT::nvinfer`, `TensorRT::nvonnxparser`, `TensorRT_RTX::tensorrt_rtx`, `QNN::headers`, `OpenVINO::runtime`, `DirectML::DirectML`, `MIGraphX::migraphx`, `HIP::amdhip64`, `MLX::mlx`), and `ONNXRUNTIME_BUNDLE_PLUGIN_EPS` with the registration names and paths of the plugin EP libraries. When two packages contain the same file with different content, the newer Windows file version wins, otherwise the core's copy; `BUNDLE_INFO.json` and the bundle's README list such files.

### Third-party files in bundles

The third-party files of a bundle stay under their own licenses, not under the license of the ONNX Runtime binaries; the `ThirdPartyNotices.txt` files in `packages/<package>/` cover the open-source code linked into the binaries. Third-party files come in three groups:

**1. Open-source components built from source: always in the bundle.** The packages contain them with their license texts in `third_party_licenses/<component>/`:

| Component | In the bundle with | License |
|---|---|---|
| oneDNN (`dnnl.dll` / `libdnnl.so.3`) | oneDNN in the core | Apache-2.0 |
| TIM-VX (`libtim-vx.so`) | VSINPU in the core | MIT |
| DirectX Shader Compiler (`dxcompiler.dll`) | WebGPU on Windows (core or plugin) | NCSA |
| MIGraphX, built from its develop branch, with `include/migraphx/` and `migraphx*.lib` (Windows) | the MIGraphX or AMDGPU plugin | MIT |
| HIP headers (`include/hip/`) and `amdhip64.lib` (Windows); the HIP runtime comes with ROCm | the AMDGPU plugin | MIT |
| MLX, mlx-c (with fmt, nlohmann/json, gguflib, metal-cpp), with `include/mlx/` | the MLX plugin | MIT, Apache-2.0 |

The statically linked Dawn of the WebGPU plugin has no separate library to link against (ONNX Runtime does not support building the WebGPU plugin EP with Dawn as a shared library: `onnxruntime_USE_EP_API_ADAPTERS` and `onnxruntime_BUILD_DAWN_SHARED_LIBRARY` are mutually exclusive).

**2. Vendor SDK files: always downloaded on the user's machine.** The licenses of these SDKs restrict redistribution (some allow it only as part of an application, the import libraries and part of the headers not at all), so neither the packages nor the bundles contain them, and the public package artifacts redistribute no vendor file. When the selection needs any, the bundle contains `fetch-restricted.sh` / `fetch-restricted.ps1` (needs Python 3): it lists the licenses, asks for confirmation (`--yes` skips it) and runs [`scripts/fetch_ep_runtime.py`](scripts/fetch_ep_runtime.py) (copied into `tools/` with `download_ep_packages.py`) on the bundle. The script downloads the exact versions the packages were built with from the vendors' no-account sources: each runtime together with its headers and import libraries, which come from the same archives, and puts every file where the CMake package expects it. Skip it if you installed the SDKs yourself, or limit it with `--ep` (`--ep <ep>:runtime` leaves out the headers and import libraries). It also gets the ROCm runtime of the AMD plugins on Windows, which is open source but large and specific to the GPU targets. The download sizes are for x86_64:

| EP | SDK (license) | Runtime | Headers and import libraries | Download |
|---|---|---|---|---|
| CUDA (core or plugin) | CUDA (EULA, Attachment A), cuDNN | cudart, cuBLAS / cuBLASLt, cuFFT, cuRAND, NVRTC, nvJitLink; cuDNN | of the same libraries, with CCCL | about 2.5 GB (Windows), 2.2 GB (Linux) |
| TensorRT (core or plugin) | TensorRT (NVIDIA TensorRT license; its headers are Apache-2.0), CUDA | `nvinfer*` (with builder resources, lean / dispatch / plugin libraries), `nvonnxparser`; cudart, cuBLAS, nvJitLink | TensorRT and those CUDA libraries | about 2.5 GB (Windows), 8 GB (Linux) |
| TensorRT RTX (core or plugin) | TensorRT for RTX, CUDA | `tensorrt_rtx_*`, `tensorrt_onnxparser_rtx_*`; cudart | TensorRT RTX and cudart | about 100 MB |
| QNN (core or plugin) | QAIRT (Qualcomm AI Stack License: object code, as part of an application) | QAIRT backends (`Qnn*`, `Genie*`, `HtpPrepare`) and Hexagon skel libraries | `include/QNN` headers | about 3 GB |
| OpenVINO (core or plugin) | OpenVINO (Intel OpenVINO Distribution License for the runtime, Apache-2.0 for the headers) | runtime and device plugins listed in its `redist.txt` (Windows also oneTBB) | headers and import libraries | about 200 MB (Windows), 110 MB (Linux) |
| DirectML (core, AMD plugins on Windows) | `Microsoft.AI.DirectML` NuGet package (redistributable with applications) | `DirectML.dll` | `DirectML.h`, `DirectMLConfig.h`, `DirectML.lib` | about 195 MB |
| Vitis AI (in the core, Windows x64) | Vitis AI runtime (`voe` wheel; redistribution terms not verified) | `onnxruntime_vitisai_ep.dll`, `dyn_dispatch_core.dll`, `aiecompiler_client.dll`, `onnxruntime_vitis_ai_custom_ops.dll` | — | — |
| CANN (in the core, Linux) | CANN toolkit (not redistributable) | the whole toolkit: its vendor installer is run into `lib/cann` (source its `set_env.sh` before use) | — | — |
| NeuroPilot (plugin) | the prebuilt op capability library from the plugin's releases (no redistribution terms stated) | `libneuron_op_capability.so` | — | under 1 MB |
| HIP backend (`onnxruntime-ep-amdgpu-hip`) | ROCm/hip-ep (MIT; on Windows with the ROCm runtime it is built with, under the ROCm licenses) | Windows: `hipgpu.dll` and its ROCm runtime (`amdhip64_7`, comgr, hipRTC, hipBLASLt with its kernel library tree, `amdmlss`, `custom_kernels_*`) next to the plugin, so the process uses that runtime rather than `rocm/`; Linux: `libhipgpu.so`, `libcustom_kernels_*.so` and the clang resource directory `lib/clang/` (with the system ROCm) | import libraries (Windows; the archives have no headers) | about 300 MB (Windows), 475 MB (Linux) |
| MIGraphX, AMDGPU (plugins, Windows) | ROCm (TheRock's wheels, the version the packages were built with) | `rocm-sdk-core`, `rocm-sdk-libraries` and the `rocm-sdk-device-<gfx>` of the bundle's GPU targets, unpacked into `rocm/` in their own layout (rocBLAS, hipBLASLt and comgr find their kernels and device libraries relative to their files); put `rocm/_rocm_sdk_core/bin` and `rocm/_rocm_sdk_libraries/bin` on the application's `PATH` | — | about 0.8 GB plus 0.15 GB per GPU target |

The script also works outside a bundle, for example to collect the runtime files of a target or to check the extraction rules:

```bash
python scripts/fetch_ep_runtime.py --ep cuda tensorrt --target x86_64-unknown-linux-gnu --dest ./runtime
python scripts/fetch_ep_runtime.py --ep qnn:dev --target aarch64-pc-windows-msvc --dry-run
python scripts/fetch_ep_runtime.py --list
```

`--ep <ep>[:runtime|dev|runtime+dev]` selects the runtime files (the default), the development files (headers into `include/<backend>/` keeping their directory tree, Windows import libraries into `lib/`), or both. Targets: `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc`, `arm64ec-pc-windows-msvc`, `arm64x-pc-windows-msvc`, `aarch64-linux-android`, `aarch64-apple-darwin`. The script writes `THIRD_PARTY_RUNTIME.json` (versions, sources and extracted files) and copies the license texts into `third_party_licenses/`. Archives are only streamed, never fully unpacked; downloads are cached and verified when the upstream provides a checksum.

**3. Installed by the user.** ROCm (HIP, rocBLAS, MIOpen, ...) for the MIGraphX and AMDGPU EPs on Linux: the AMD packages expect the release they were built with in `/opt/rocm` (the HIP backend also the release hip-ep is built with). On Linux the OpenVINO runtime also needs the system oneTBB (`libtbb12`). The NeuroPilot plugin needs the MediaTek Neuron runtime of the device (`libneuronusdk_adapter.mtk.so`). GPU and NPU drivers are never included. The bundle's README says what its selection needs.

### Checking requests locally

```bash
python ci/catalog.py --out catalog.json                       # what the page offers
python ci/bundle.py resolve --config config.json --plan plan.json   # GITHUB_TOKEN finds reusable packages
python ci/bundle.py download --plan plan.json --dest packages # reused packages only; needs the GitHub CLI
python ci/bundle.py assemble --plan plan.json --packages packages --work work --out dist
```
