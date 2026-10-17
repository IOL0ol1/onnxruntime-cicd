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
│   │   └── onnxruntime-ep-mlx/          # Apple MLX plugin EP
│   ├── vendor/                          # EP sources or forks maintained by hardware vendors
│   │   └── nvidia-tensorrt-rtx-ep/      # NVIDIA TensorRT RTX plugin EP
│   └── community/                       # third-party / community plugin EPs
│       ├── onnxruntime-ep-iree/         # IREE (MLIR compiler) plugin EP
│       ├── onnxruntime-ep-telum/        # IBM Telum (IBM Z AI accelerator) plugin EP
│       └── onnxruntime-ep-vulkan/       # Vulkan compute plugin EP (experimental)
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
| `eps/vendor/nvidia-tensorrt-rtx-ep` | [NVIDIA/TensorRT-RTX-EP-ABI](https://github.com/NVIDIA/TensorRT-RTX-EP-ABI) | `main` | NVIDIA RTX GPU (TensorRT-RTX) | NVIDIA | Plugin TensorRT RTX EP; PyPI package `onnxruntime-ep-nv-tensorrt-rtx` |
| `deps/amdmigraphx` | [ROCm/AMDMIGraphX](https://github.com/ROCm/AMDMIGraphX) | `develop` | MIGraphX itself, for the AMD plugin EPs | AMD | Built from source because the AMD plugin EPs follow its `develop` API; a submodule so that its version only changes with an update, which keeps the MIGraphX build cache valid |
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
| Intel OpenVINO | `openvino` | Stable | Development line: the `ovep-develop` branch of [intel/onnxruntime](https://github.com/intel/onnxruntime) (not included: the main repository has the same EP); plugin binary package `onnxruntime-ep-openvino` |
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
| `openvino` | OpenVINO (in-tree) | 2026.2.1 | `storage.openvinotoolkit.org` | sha256 |
| `cann` | Huawei CANN | Latest release (currently 9.1.1) | Huawei Cloud Ascend artifactory `ascend.devcloud.huaweicloud.com/artifactory/cann-run` (instead of the official site, which needs a Huawei account) | sha256 |
| `rocm` | MIGraphX / AMDGPU (in-tree + `onnxruntime-ep-amdgpu`) | Latest TheRock multi-arch package (currently 7.14.1); the Windows MIGraphX CI build installs TheRock's 7.13.0 Python wheels instead | `repo.amd.com/rocm/tarball-multi-arch` | — |
| `vitisai` | AMD Vitis AI (Ryzen AI NPU) | Latest version of each package | Public AMD PyPI index `pypi.amd.com/simple` | sha256 |
| `android-ndk` | NNAPI, QNN / XNNPACK on Android | 28.0.13004108 (same as ORT `android.yml`) | `dl.google.com/android/repository` | sha1 |
| `onnxruntime-sdk` | All plugin EPs (headers and libraries as `ORT_HOME`); `--platform windows-arm64x` selects the `win-arm64x` package | Latest release (currently 1.30.0) | GitHub Releases | sha256 |
| `vulkan-sdk` | Vulkan plugin EP (provides glslc) | 1.3.296.0 (same as the plugin's CI; glslc output differs between versions) | `sdk.lunarg.com` (official LunarG) | sha256 |
| `tim-vx` | VeriSilicon VSINPU | Latest tag | git: `github.com/VeriSilicon/TIM-VX` | — |
| `zdnn` | IBM Telum plugin (`libzdnn` is dlopen'ed at runtime, s390x only) | Latest tag | git: `github.com/IBM/zDNN` | — |

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

## Third-party runtime files

The packages bundle the third-party runtime libraries whose licenses allow distributing them with an application, each with its license texts in `third_party_licenses/<component>/` (bundled vendor runtimes are also listed in `THIRD_PARTY_RUNTIME.json`). These files stay under their own licenses, not under the license of the ONNX Runtime binaries. `ThirdPartyNotices.txt` covers the open-source code linked into the binaries.

| Bundled in | Component | License |
|---|---|---|
| A core with oneDNN | oneDNN (`dnnl.dll` / `libdnnl.so.3`), built from source | Apache-2.0 |
| A core with VSINPU | TIM-VX (`libtim-vx.so`), built from source | MIT |
| Windows: a core with WebGPU, `onnxruntime-ep-webgpu` | DirectX Shader Compiler (`dxcompiler.dll`), built from source | NCSA |
| A core with DirectML, `onnxruntime-ep-directml-amd`, Windows `onnxruntime-ep-amdgpu` | DirectML (`DirectML.dll`) | Microsoft DirectML license (redistributable with applications) |
| A core with OpenVINO | OpenVINO runtime and device plugins (Windows also oneTBB) | Intel OpenVINO Distribution License: the files listed in its `redist.txt` |
| A core with QNN, `onnxruntime-ep-qnn` | QAIRT backends and Hexagon skel libraries | Qualcomm AI Stack License (object code, as part of an application) |
| A core with TensorRT RTX, `onnxruntime-ep-nv-tensorrt-rtx-*` | TensorRT for RTX and the CUDA runtime (`cudart`) | NVIDIA TensorRT for RTX license, CUDA EULA (Attachment A) |
| `onnxruntime-ep-migraphx-amd`, `onnxruntime-ep-amdgpu` | MIGraphX, built from its develop branch | MIT |
| `onnxruntime-ep-mlx` | MLX, mlx-c (with fmt, nlohmann/json, gguflib, metal-cpp) | MIT, Apache-2.0 |

**Development files of the plugin EP backends.** Each plugin EP package also carries the interface headers (`include/`) and link files (import libraries on Windows; the shared libraries above elsewhere) of its backend, unless the backend's license rules that out:

| Package | Headers | Link files |
|---|---|---|
| `onnxruntime-ep-migraphx-amd`, `onnxruntime-ep-amdgpu` | `include/migraphx/` | `migraphx*.lib` (Windows) |
| `onnxruntime-ep-amdgpu` (HIP backend) | `include/hip/` | `amdhip64.lib` (Windows); the HIP runtime comes with ROCm |
| `onnxruntime-ep-directml-amd`, Windows `onnxruntime-ep-amdgpu` | `include/DirectML.h` | `DirectML.lib` |
| `onnxruntime-ep-mlx` | `include/mlx/` (MLX and mlx-c) | the bundled `libmlx*.dylib` |
| `onnxruntime-ep-vulkan` | `include/vulkan/`, `include/vk_video/` | `vulkan-1.lib` (Windows); the loader comes with the graphics driver |

Not included: the QAIRT headers (its license allows object code only), the CUDA / cuDNN / TensorRT / TensorRT for RTX headers and import libraries (their licenses allow only the runtime files, accessed by your application), and the IREE, zDNN and statically linked Dawn backends of `onnxruntime-ep-iree`, `onnxruntime-ep-telum` and `onnxruntime-ep-webgpu`, which have no separate library to link against (ONNX Runtime does not support building the WebGPU plugin EP with Dawn as a shared library: `onnxruntime_USE_EP_API_ADAPTERS` and `onnxruntime_BUILD_DAWN_SHARED_LIBRARY` are mutually exclusive).

Not bundled, so install or fetch them separately:

- **CUDA libraries beyond cudart, cuDNN and TensorRT**: their licenses allow it, but they are several GB; the packages leave them out and [bundles](#bundles) add them.
- **The CANN toolkit**: Huawei's license does not allow redistribution.
- **The Vitis AI runtime**: its license could not be verified.
- **ROCm** (HIP, rocBLAS, MIOpen, ...): a system installation; the AMD packages expect it in `/opt/rocm` (Linux) or a HIP SDK / TheRock install (Windows). On Linux the OpenVINO runtime also needs the system oneTBB (`libtbb12`).

[`scripts/fetch_ep_runtime.py`](scripts/fetch_ep_runtime.py) downloads the CUDA, cuDNN, TensorRT, CANN and Vitis AI runtimes from the same no-account sources as the download script, extracts **only the runtime files**, and copies them into place together with the vendors' license texts.

```bash
# Point it at an extracted package: the EPs, target triple, CUDA major version and the exact SDK
# versions used by the build are read from BUILD_INFO.json; files go into the package's bin/ (Windows) or lib/
python scripts/fetch_ep_runtime.py --package-dir onnxruntime-<version>-x86_64-pc-windows-msvc

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
| `cann` (a Linux core with CANN) | CANN toolkit | Cannot be reduced to a few files; the vendor installer is run into `<dest>/cann` (source its `set_env.sh` before use) |

An EP can be suffixed with the parts to fetch: `--ep cuda:dev` extracts the backend's **development files** instead (headers into `include/<backend>/` keeping their directory tree, Windows import libraries into `lib/`), `cuda:runtime+dev` both. The development files exist for `cuda` (CUDA headers including CCCL, cuDNN), `tensorrt`, `nv-tensorrt-rtx`, `qnn` (QAIRT `include/QNN`) and `openvino` (headers and import libraries only; the packages bundle its runtime). The bundles of [bundle requests](#bundles) use this through their `fetch-restricted` script.

Targets: `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc`, `arm64ec-pc-windows-msvc`, `arm64x-pc-windows-msvc`, `aarch64-linux-android`. The script writes `THIRD_PARTY_RUNTIME.json` (versions, sources and extracted files) and copies the license texts into `third_party_licenses/`. Archives are only streamed, never fully unpacked; downloads are cached in `packages/` and verified when the upstream provides a checksum. The extracted files remain subject to their vendors' licenses; the CANN toolkit may not be redistributed at all. With `--package-dir`, the script fetches only what the package does not bundle (the QNN and TensorRT RTX runtimes are already in their packages). GPU / NPU drivers must be installed separately.

## Build recipes

Every package is built from [`ci/matrix.json`](ci/matrix.json): the core of a bundle from the core recipe of its target triple with the in-tree EPs the request selected, a plugin EP package from its entry (a recipe and variant of [`ci/build.py`](ci/build.py) for one target triple), through the [build steps](.github/actions/build-entry/action.yml) shared by all builds: check out only the submodules the entry needs (shallow), download the third-party SDKs (`ci/sdk.py`), build and stage `bin/ lib/ include/`, and pack it with [`ci/package.py`](ci/package.py) as `<package>-<inputs hash>-<target-triple>.tar.gz/.zip` with a `.sha256` and `BUILD_INFO.json`. Nothing builds them on a schedule: a [bundle request](#bundles) builds the packages it needs, and later requests reuse them (see [Builds on demand and caching](#builds-on-demand-and-caching)).

### Core recipes

`cores` in `ci/matrix.json` has one recipe per target triple: the build platform (the `ci/build.py` variant), the runner and the in-tree EPs a core can be built with there, each with its build requirements (packages, container, the CUDA major versions it was built with, long build, experimental). A bundle's core always contains the CPU EP plus the in-tree EPs the request compiles into it; cores with the same EP set are reused.

| Target | Platform (variant) | In-tree EPs a core can contain |
|---|---|---|
| `x86_64-unknown-linux-gnu` | desktop (`custom`) | XNNPACK, WebGPU, CUDA (12/13), TensorRT (12/13), TensorRT RTX (13), oneDNN, OpenVINO, MIGraphX (in the `rocm/dev-ubuntu-24.04:7.2.4-complete` container), Vitis AI, CANN, VSINPU ⚠ (TIM-VX with the x86 simulator driver) |
| `aarch64-unknown-linux-gnu` | desktop | XNNPACK, WebGPU, CUDA (13), TensorRT (13), CANN |
| `x86_64-pc-windows-msvc` | desktop | XNNPACK, DirectML, WebGPU, CUDA (12/13), TensorRT (12/13), TensorRT RTX (13), oneDNN, OpenVINO, Vitis AI, QNN |
| `aarch64-pc-windows-msvc` | desktop | XNNPACK, DirectML, WebGPU, QNN |
| `aarch64-apple-darwin`, `x86_64-apple-darwin` | desktop (x86_64 cross-compiled on arm64) | XNNPACK, CoreML, WebGPU |
| `arm64x-pc-windows-msvc`, `arm64ec-pc-windows-msvc` | `arm64x`: an ARM64 pass plus an ARM64EC pass with `--buildasx`, like the official `win-arm64x` package; the ARM64EC target uses the ARM64X core | CPU only |
| `aarch64-apple-ios`, `aarch64-apple-ios-sim`, `x86_64-apple-ios` | Apple framework (`ort-apple`): a dynamic `onnxruntime.framework` per sysroot/arch with ORT's `build_apple_framework.py`, whose settings get the selected EPs | CoreML, XNNPACK |
| `aarch64-apple-ios-macabi`, `x86_64-apple-ios-macabi`, `*-apple-tvos*` ⚠, `*-apple-visionos*` ⚠ | Apple framework | CoreML (ORT's settings build XNNPACK for iOS only) |
| `aarch64-linux-android`, `armv7-linux-androideabi`, `i686-linux-android`, `x86_64-linux-android` | `android` | NNAPI, XNNPACK, WebGPU |
| `wasm32-unknown-emscripten` | one of five exclusive build modes, the same as ORT's `web.yml`: CPU, JSEP (+ WebNN), WebGPU (+ WebNN), WebGPU with JSPI (+ WebNN), or the static library `libonnxruntime_webassembly.a` | given by the mode |
| `s390x-unknown-linux-gnu` ⚠ | `s390x`: cross-compiled on x86_64 with the Ubuntu s390x toolchain (MLAS has s390x kernels; ORT's CMake downloads a protoc for the build host) | CPU only |

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
| `onnxruntime-ep-amdgpu` | AMDGPU EP (`--use_amdgpu`; includes the MIGraphX backend, plus the DirectML backend on Windows) | `x86_64-unknown-linux-gnu`, `x86_64-pc-windows-msvc` ⚠ |
| `onnxruntime-ep-qnn` | Qualcomm QNN plugin EP | `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc`, `arm64ec-pc-windows-msvc` ⚠, `arm64x-pc-windows-msvc` ⚠, `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu` ⚠, `aarch64-linux-android` ⚠ |
| `onnxruntime-ep-mlx` | MLX plugin EP (with the MIT-licensed libmlx) | `aarch64-apple-darwin` |
| `onnxruntime-ep-iree` | IREE plugin EP (with a wheel) | `x86_64-unknown-linux-gnu`, `x86_64-pc-windows-msvc`, `aarch64-apple-darwin` |
| `onnxruntime-ep-telum` | IBM Telum plugin EP (cross-compiled on x86_64) | `s390x-unknown-linux-gnu` |
| `onnxruntime-ep-vulkan` | Vulkan plugin EP | `x86_64-unknown-linux-gnu`, `x86_64-pc-windows-msvc` |

⚠ marks experimental entries (`experimental: true`): known to fail at times. `arm64ec-pc-windows-msvc` and `arm64x-pc-windows-msvc` are not official Rust triples; they name the Windows ARM64EC and ARM64X binaries.

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

These jobs are heavy (rocMLIR is an LLVM-based build) and may exceed the 6-hour limit of GitHub-hosted runners; move them to a larger runner with `HEAVY_RUNNER_MAP`. The GPU architectures come from the `MIGRAPHX_GPU_TARGETS` repository variable (default `gfx1100;gfx1101;gfx1102;gfx1200;gfx1201`, i.e. RDNA3 / RDNA4, which the Windows ML MIGraphX EP requires).

### Builds on demand and caching

- **Package reuse**: [`ci/plan.py`](ci/plan.py) hashes the inputs of every package build: the entry, the commits of its submodules, the build scripts (`ci/build.py`, `ci/sdk.py`, `ci/package.py`, `scripts/`, the build-entry action), the resolved third-party SDK versions, the Docker image digest and the CUDA architectures. The package artifact is named `pkg-<package>-<triple>-<hash>` and kept for 90 days; a bundle request whose package has the same hash downloads that artifact instead of building it. Changes to the bundle and page code (`ci/bundle.py`, `ci/catalog.py`, `site/`) do not invalidate packages. Cores of different EP sets of a target share one ccache directory (ccache key per target and CUDA major version), which helps only with the third-party code: ORT adds the `USE_<EP>` definitions of the selected EPs to every ORT source file. `python ci/plan.py` lists every plugin entry and the default core of every target with its hash and whether an artifact can be reused.
- **Compiler cache**: the CUDA and MIGraphX builds compile through ccache ([`hendrikmuhs/ccache-action`](https://github.com/hendrikmuhs/ccache-action), up to 2 GB per entry). CMake reads `CMAKE_C_COMPILER_LAUNCHER`, `CMAKE_CXX_COMPILER_LAUNCHER` and `CMAKE_CUDA_COMPILER_LAUNCHER` from the environment, so ONNX Runtime, MIGraphX with its dependencies and the EP repositories all use it (not the Visual Studio generator builds, nor container builds). A first build gains nothing; a build after a submodule update recompiles only what changed. The repository has 10 GB of Actions cache; the least recently used entries are evicted, and entries unused for 7 days expire.
- **MIGraphX on Windows**: the dependencies `rbuild prepare` builds (rocMLIR and others, about 2.5 hours) and the MIGraphX install share one Actions cache entry, keyed by the dependency inputs (MIGraphX's requirement files, the ROCm version, the MSVC toolset, the GPU targets) and the `deps/amdmigraphx` commit. A new MIGraphX commit restores the entry of the same dependency inputs and rebuilds only MIGraphX; an unchanged commit reuses the install, which the `ep-migraphx-amd` and `ep-amdgpu` builds share.
- **Third-party SDKs** are not cached: they take a few minutes at most, would exceed the cache quota, and caches of a public repository can be restored by pull-request workflows, which the SDK licenses do not allow.
- **Dependency updates**: Dependabot (`.github/dependabot.yml`) opens weekly pull requests that update the actions and move the submodules to the newest commits of their branches. Merging one builds nothing; the next request that needs an affected package builds it.
- **CUDA architectures**: the default is `80-real;80-virtual`: SM 8.0 SASS plus a compute_80 PTX that the driver JIT-compiles on newer GPUs (ORT's CMake drops a list that has only `-virtual` entries). GitHub-hosted runners have 4 cores, and every extra CUDA architecture roughly doubles the build time, which easily exceeds the 6-hour job limit. To build several SASS architectures, set the `CUDA_ARCHITECTURES` repository variable and move the heavy builds to larger runners with `HEAVY_RUNNER_MAP`, e.g. `{"ubuntu-24.04": "linux-16core", "windows-2025": "windows-16core"}` (`RUNNER_MAP` replaces runners for every build).
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
2. **Execution providers**: one row per EP the target offers, each with **Off / Plugin EP / In core**. CPU is always in the core.
   - **Plugin EP** adds the EP's plugin EP package (`onnxruntime-ep-*`): a separate library registered at run time, built once per version and reused by every bundle. It is the default for every EP that has one (CUDA, TensorRT, TensorRT RTX, QNN, MIGraphX, DirectML, WebGPU, AMDGPU, MLX, IREE, Telum, Vulkan), also where the plugin is experimental.
   - **In core** compiles the EP into the core library, as built-in code or as a provider library next to it (OpenVINO, oneDNN, Vitis AI, CANN, XNNPACK, CoreML, NNAPI, VSINPU have only this form; the others where the target's core recipe supports them). When the page opens, the core has XNNPACK where the target supports it.
   - The page shows only what the target supports and disables options that do not work with the current selection, with the reason: TensorRT in the core brings the CUDA EP into the core (ORT builds them together; moving CUDA out of the core moves TensorRT to its plugin); one CUDA major version per bundle (a selector appears as soon as an EP needs CUDA, and options not built for it are disabled); EPs a target's core cannot contain. Notes explain long builds and untested combinations (MIGraphX in the core builds the whole core in the ROCm container).
   - WebAssembly offers the five build modes instead (single choice), because they are exclusive builds.
   - *Select all* puts every EP of the target in its default form; *Reset* goes back to the defaults.
3. **Options**: whether to include the headers and link files of the backends, and which parts `fetch-restricted` should fetch.
4. **Request**: the EPs and how each one is provided, which packages take long to build, the third-party components with links to their licenses, and the license acknowledgement. *Create bundle request* opens a prefilled issue (`.github/ISSUE_TEMPLATE/bundle.yml`).

### Building a bundle

[`.github/workflows/bundle.yml`](.github/workflows/bundle.yml) runs for issues titled `[bundle] ...` whose author is a collaborator of the repository (`OWNER`, `MEMBER` or `COLLABORATOR`); everyone else's issues are ignored. Editing the issue builds it again (reusing the packages that exist); a collaborator's comment starting with `/rebuild` builds the packages and the bundle again without reusing anything.

| Job | Purpose |
|---|---|
| `resolve` | [`ci/bundle.py`](ci/bundle.py) `resolve` reads the configuration from the issue, validates it against the catalog (the same rules as the page) and plans every package with [`ci/plan.py`](ci/plan.py): its build job and inputs hash, and the artifact of an earlier run it can reuse. The status comment in the issue shows which packages are reused and which are built. An unexpired bundle with the same configuration and package hashes is linked instead of building it again |
| `build` | One job per package that cannot be reused, through the shared [build steps](.github/actions/build-entry/action.yml); the core is built from the core recipe of its target with the selected in-tree EPs (`ci/build.py --variant custom --eps ...` on desktop targets), on its runner, with the packages and container its EPs need. Each package is kept as an artifact for 90 days, also when another package of the request fails |
| `assemble` | Downloads the packages (`gh run download`, checked against their SHA-256), merges them, adds the redistributable third-party runtimes with the SDK versions the packages were built with, writes the CMake package, `plugins.json`, the example, `README.md` and `BUNDLE_INFO.json`, uploads the bundle as an artifact and links it in the issue, which is then closed |

The bundle artifact is kept for 90 days; downloading it needs a GitHub login. It contains `onnxruntime-bundle-<key>-<triple>.zip` (Windows) or `.tar.gz` with:

```
include/onnxruntime/      ONNX Runtime C/C++ API
include/<backend>/        backend headers: cuda/ (with cccl/), cudnn/, tensorrt/, tensorrt_rtx/, qnn/, openvino/,
                          migraphx/, hip/, vulkan/, mlx/, DirectML.h
lib/                      import libraries (Windows) or shared libraries; lib/cmake/onnxruntime-bundle/
bin/                      DLLs (Windows)
plugins.json              plugin EP libraries to register with RegisterExecutionProviderLibrary
packages/<package>/       LICENSE, ThirdPartyNotices.txt and BUILD_INFO.json of every package
third_party_licenses/     licenses of the bundled third-party components
examples/                 CMake example that registers the plugin EPs and lists the EPs
fetch-restricted.sh/.ps1  only when the selection needs files that may not be redistributed (see below)
BUNDLE_INFO.json          configuration, packages, EPs, third-party parts, files that differed between packages
```

```cmake
find_package(onnxruntime-bundle CONFIG REQUIRED PATHS "<bundle>/lib/cmake")
target_link_libraries(app PRIVATE onnxruntime::onnxruntime TensorRT::nvinfer CUDA::cudart)
onnxruntime_bundle_copy_runtime(app)   # copies the DLLs next to app on Windows
```

The CMake package defines `onnxruntime::onnxruntime` and a target per backend whose files are present (`CUDA::cudart`, `CUDA::cublas`, `CUDNN::cudnn`, `TensorRT::nvinfer`, `TensorRT::nvonnxparser`, `TensorRT_RTX::tensorrt_rtx`, `QNN::headers`, `OpenVINO::runtime`, `DirectML::DirectML`, `MIGraphX::migraphx`, `HIP::amdhip64`, `Vulkan::Vulkan`, `MLX::mlx`), and `ONNXRUNTIME_BUNDLE_PLUGIN_EPS` with the registration names and paths of the plugin EP libraries. When two packages contain the same file with different content, the newer Windows file version wins, otherwise the core's copy; `BUNDLE_INFO.json` and the bundle's README list such files.

### Third-party files in bundles

| Part | Handling |
|---|---|
| Runtimes whose licenses allow redistributing them with an application: oneDNN, DirectML, DXC, OpenVINO (its `redist.txt`), QAIRT, TensorRT RTX, MIGraphX, MLX, **and in bundles also the CUDA libraries, cuDNN and TensorRT** (artifacts have no 2 GiB limit) | In the bundle, with their licenses in `third_party_licenses/` |
| Headers and import libraries whose licenses do not allow redistribution: CUDA, cuDNN, TensorRT, TensorRT RTX, QAIRT; the OpenVINO development files; the Vitis AI runtime (terms not verified); the CANN toolkit (not redistributable) | `fetch-restricted.sh` / `fetch-restricted.ps1` in the bundle (needs Python 3) lists the licenses, asks for confirmation (`--yes` skips it) and downloads the exact SDK versions the packages were built with from the vendors' no-account sources (`scripts/fetch_ep_runtime.py` and `download_ep_packages.py` are copied into `tools/`), putting every file where the CMake package expects it |
| ROCm (MIGraphX, AMDGPU), GPU / NPU drivers | Installed by the user; the bundle's README says what is needed |

### Checking requests locally

```bash
python ci/catalog.py --out catalog.json                       # what the page offers
python ci/bundle.py resolve --config config.json --plan plan.json   # GITHUB_TOKEN finds reusable packages
python ci/bundle.py download --plan plan.json --dest packages # reused packages only; needs the GitHub CLI
python ci/bundle.py assemble --plan plan.json --packages packages --work work --out dist
```
