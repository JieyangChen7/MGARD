<img src="./doc/images/MGARD-logo.png" width="200" /> 

[![build status][push workflow badge]][push workflow] [![format status][format workflow badge]][format workflow]

MGARD (MultiGrid Adaptive Reduction of Data) is a technique for multilevel lossy compression and refactoring of scientific data based on the theory of multigrid methods.
We encourage you to [make a GitHub issue][issue form] if you run into any problems using MGARD, have any questions or suggestions, etc.



[push workflow]: https://github.com/CODARcode/MGARD/actions/workflows/build.yml
[push workflow badge]: https://github.com/CODARcode/MGARD/actions/workflows/build.yml/badge.svg
[format workflow]: https://github.com/CODARcode/MGARD/actions/workflows/format.yml
[format workflow badge]: https://github.com/CODARcode/MGARD/actions/workflows/format.yml/badge.svg
[issue form]: https://github.com/CODARcode/MGARD/issues/new/choose

[<img src="./doc/images/MGARD-family2.png" width="800" />](./doc/images/MGARD-family2.png)

MGARD framework consists of the following modules. Please see the detailed instructions for each module to build and install MGARD.

## ***MGARD-CPU***: MGARD compression implementation for CPUs
*MGARD-CPU* is designed for running compression on CPUs. See the detailed user guide [here][mgard-cpu]. In addition, *MGARD-CPU* can be configured to preserve region-of-interest ([RoI user guide][mgard-roi]) and linear quantity-of-interest ([QoI user guide][mgard-qoi]) during data compression.

[mgard-cpu]: doc/MGARD-CPU.md
[mgard-roi]: doc/MGARD-RoI.md
[mgard-qoi]: doc/MGARD-QoI.md

## ***MGARD-X***: Accelerated and portable compression
*MGARD-X* is designed for portable compression on NVIDIA GPUs, AMD GPUs, and CPUs. See the detailed user guide [here][mgard_x instructions].
In addition, *MGARD-X* can be configured to preserve region-of-interest ([ROI user guide][mgard-x-roi]) and linear quantity-of-interest ([QoI user guide][mgard-qoi]) during data compression.

[mgard_x instructions]: doc/MGARD-X.md
[mgard-x-roi]: doc/MGARD-RoI2.md
[mgard-qoi]: doc/MGARD-QoI.md

## ***MGARD***$\text{-}\lambda$: Preserving Non-Linear Quantity-of-Interest
*MGARD*$\text{-}\lambda$ is specifically designed for preserving non-linear quantity-of-interest during data compression. This is an experimental part of MGARD. Currently, it only supports certain QoIs derived from XGC 5D data. See theory in [here][mgard-lambda-theory] and example in [here][mgard-lambda].

[mgard-lambda-theory]: doc/images/post-processing.pdf
[mgard-lambda]: ./examples/lambda

## ***MDR/MDR-X***: Fine-grain progressive data reconstruction
*MDR* and *MDR-X* are designed to enable fine-grain data refactoring and progressive data reconstruction. See the detailed user guide [here][mdr_x instructions].

[mdr_x instructions]: doc/MDR-X.md

## Self-describing format for compressed and refactored data
Data produced by MGARD, MGARD-X, and MDR-X are designed to follow a unified self-describing format. See format details in [here][mgard format].

[mgard format]: doc/MGARD-format.md

## Version history
Detailed release notes (features added, changes, and bug fixes) are linked below for each version.
* [MGARD 1.7.0](doc/release%20notes/1.7.0.md) (Unreleased) — Delivered ***HP-MDR***, a high-performance MDR-X refactoring and reconstruction pipeline; introduced ***BlockMGARD***, a block-based hybrid hierarchy compression pipeline with region-of-interest support; deprecated and removed the legacy standalone MGARD-CUDA backend in favor of ***MGARD-X***; added new rANS and BlockDelta lossless backends and a portable warp-cooperative LZ4 implementation; added Blackwell (sm_120) GPU build support; numerous performance improvements and bug fixes.
* [MGARD 1.6.0](doc/release%20notes/1.6.0.md) (Aug. 2025) — Redesigned the compression/decompression pipeline for higher end-to-end throughput; improved OpenMP and Huffman CPU performance and ZSTD linking; removed the prefetch option (now always enabled) and the coordinate-normalization build option; fixed issues with LZ4 compression, thread safety, MDR-X L2 error control, ADIOS2 integration, and HIP builds.
* [MGARD 1.5.2](doc/release%20notes/1.5.2.md) (Sep. 2023) — Added compression status reporting to the high-level API, an ADIOS2 operator build example, autotuning for Huffman kernels, asynchronous LZ4/Zstd compression, a pipeline optimized for compressing time-series data, and improved memory-usage estimation; fixed bugs in Huffman codebook generation, the domain decomposer, reduced-memory-footprint mode, and MDR-X reconstruction.
* [MGARD 1.5.0](doc/release%20notes/1.5.0.md) (Apr. 2023) — Added the ***MGARD-$\lambda$*** pipeline for preserving non-linear QoIs in XGC data and the ***MGARD-RoI*** pipeline for region-of-interest preservation; added a GPU pipeline for out-of-core, large-scale compression; added Apple Silicon (ARM) support; fixed issues with CUDA (older versions and 12+), the NVIDIA HPC SDK, MDR-X compilation, and linear quantization overflow.
* [MGARD 1.4.0](doc/release%20notes/1.4.0.md) (Jan. 2023) — Added multi-device support for compression/decompression, RuntimeX, and Array; added workspace pre-allocation, a new OpenMP backend, block-based domain decomposition, and high-level MDR-X APIs; modularized the compression and refactoring workflows; reduced build time via optional autotuning; fixed bugs affecting GCC 9, Huffman encoding synchronization, Xcode, and the SYCL backend.
* [MGARD 1.3.0](doc/release%20notes/1.3.0.md) (Sep. 2022) — Introduced ***MGARD-X***: portable compression for CPU (serial and multi-threaded), NVIDIA GPUs, AMD GPUs, and Intel GPUs, with a self-describing format, automatic domain decomposition, multi-GPU parallel compression, and high-/low-level APIs. Introduced ***MDR-X*** for portable multi-precision data refactoring on CPU and GPU.
* [MGARD 1.0.0](doc/release%20notes/1.0.0.md) (Sep. 2021) — Improved CPU compression/decompression speed (iterator optimizations, index precomputation, memory-access-pattern improvements); added OpenMP parallelization; added a self-describing command-line executable and high-level APIs; added support pluggable lossless compressors; fixed several multilevel-decomposition bugs.
* [MGARD 0.1.0](doc/release%20notes/0.1.0.md) (Sep. 2020) — Added initial support for unstructured data; restructured code for extensibility; added Nvidia GPU support for 2D/3D; added Huffman entropy encoding and ZSTD integration; added FP64 support; added continuous integration (Travis CI).
* [MGARD 0.0.0.2](doc/release%20notes/0.0.0.2.md) (Sep. 2019) — Initial public release. Lossy compression with preservation of $L_\infty$, $L_2$ and S-norm on primary data, and linear QoIs; added FP32 support.

## Publications
The following works either contribute to the MGARD framework and/or extend and apply MGARD for various applications, systems, and use cases.
### MGARD Foundations
* Qian Gong et al. [MGARD: A multigrid framework for high-performance, error-controlled data compression and refactoring.][mgard-softwarex] *SoftwareX*, Dec. 2023
* Xin Liang et al. [MGARD+: Optimizing Multilevel Methods for Error-bounded Scientific Data Reduction.][mgard+] *IEEE Transactions on Computers*, 2021
* Mark Ainsworth et al. [Multilevel Techniques for Compression and Reduction of Scientific Data—The Unstructured Case.][unstructured] *SIAM Journal on Scientific Computing*, 42 (2), A1402–A1427, 2020.
* Mark Ainsworth et al. [Multilevel Techniques for Compression and Reduction of Scientific Data—Quantitative Control of Accuracy in Derived Quantities.][quantities] *SIAM Journal on Scientific Computing* 41 (4), A2146–A2171, 2019.
* Mark Ainsworth et al. [Multilevel Techniques for Compression and Reduction of Scientific Data—The Multivariate Case.][multivariate] *SIAM Journal on Scientific Computing* 41 (2), A1278–A1303, 2019.
* Mark Ainsworth et al. [Multilevel Techniques for Compression and Reduction of Scientific Data—The Univariate Case.][univariate] *Computing and Visualization in Science* 19, 65–76, 2018.
* Ben Whitney. [Multilevel Techniques for Compression and Reduction of Scientific Data.][thesis] PhD thesis, Brown University, 2018.

### Preserving Quantities of Interest (QoIs)
* Qian Gong et al. [Physics-Aware Adaptive Checkpointing with Shadow Systems for Nonlinear PDE Simulations.][shadow-pde] *Journal of Computational Science*, Sep. 2026
* Jaemoon Lee et al. [Error-Guaranteed Compression with Preservation of Downstream Quantities for Electron Microscopy.][em-qoi] *Microscopy and Microanalysis*, Aug. 2026
* Qian Gong et al. [Stability-preserving Lossy Compression for Large-scale Partial Differential Equations.][stability-pde] *the International Conference for High Performance Computing, Networking, Storage and Analysis 2025*, Nov. 2025
* Richard Dodson et al. [Optimising the Processing and Storage of Visibilities using lossy compression.][visibilities] *Publications of the Astronomical Society of Australia*, Jul. 2025
* Qian Gong et al. [A General Framework for Error-controlled Unstructured Scientific Data Compression.][unstructured-framework] *2024 IEEE 20th International Conference on e-Science (e-Science)*, Sep. 2024
* Tania Banerjee et al. [Fast Algorithms for Scientific Data Compression.][fast-algo] *2023 IEEE 30th International Conference on High Performance Computing, Data, and Analytics (HiPC)*, Dec. 2023
* Qian Gong et al. [Spatiotemporally adaptive compression for scientific dataset with feature preservation–a case study on simulation data with extreme climate events analysis.][climate-qoi] *2023 IEEE 19th International Conference on e-Science (e-Science)*, Oct. 2023
* Tania Banerjee et al. [Online and Scalable Data Compression Pipeline with Guarantees on Quantities of Interest.][online-qoi] *2023 IEEE 19th International Conference on e-Science (e-Science)*, Oct. 2023
* Tania Banerjee et al. [Scalable Hybrid Learning Techniques for Scientific Data Compression.][pp3], *Arxiv*, 2022
* Qian Gong et al. [Region-adaptive, Error-controlled Scientific Data Compression using Multilevel Decomposition.][roi2] *the 34th International Conference on Scientific and Statistical Database Management*, Jul. 2022
* Tania Benerjee et al. [An algorithmic and software pipeline for very large-scale scientific data compression with error guarantees.][qoi2] *International Conference on High Performance Computing, Data, and Analytics*, 2022
* Jaemoon Lee et al. [Error-bounded learned scientific data compression with preservation of derived quantities.][pp] *Applied Sciences*, 2022
* Qian Gong et al. [Maintaining trust in reduction: Preserving the accuracy of quantities of interest for lossy compression.][roi] *21st Smoky Mountains Computational Sciences and Engineering Conference*, Oct. 2021

### Progressive Retrieval
* Wenbo Li et al. [QProR: An Efficient Framework for Quantity-of-Interest Based Progressive Retrieval with Guaranteed Error Control.][qpror] *the 35th International Symposium on High-Performance Parallel and Distributed Computing*, Jul. 2026
* Yanliang Li et al. [HP-MDR: High-performance and Portable Data Refactoring and Progressive Retrieval with Advanced GPUs.][hp-mdr] *the International Conference for High Performance Computing, Networking, Storage and Analysis 2025*, Nov 2025
* Xuan Wu et al. [Error-controlled Progressive Retrieval of Scientific Data under Derivable Quantities of Interest.][qoi] *the International Conference for High Performance Computing, Networking, Storage and Analysis 2024*, Nov. 2024 
* Jinzheng Wang et al. [Improving Progressive Retrieval for HPC Scientific Data using Deep Neural Network.][progressive-dnn] *IEEE International Conference on Data Engineering (ICDE)*, 2023 
* Xin Liang et al. [Error-controlled, progressive, and adaptable retrieval of scientific data with multilevel decomposition.][mdr] *the International Conference for High Performance Computing, Networking, Storage and Analysis 2021*, Nov. 2021

### Parallelization and GPU Acceleration
* Yanliang Li et al. [BlockMGARD: Accelerating Adaptive Scientific Data Reduction with Region-of-Interest Error Control on GPUs.][gpu4] *the International Conference for High Performance Computing, Networking, Storage and Analysis 2026*, Nov. 2026
* Jieyang Chen et al. [HPDR: High-Performance Portable Scientific Data Reduction Framework.][gpu3] *39th IEEE International Parallel and Distributed Processing Symposium*, June. 2025
* Jieyang Chen et al. [Scalable Multigrid-based Hierarchical Scientific Data Refactoring on GPUs.][gpu2] *Arxiv*
* Jieyang Chen et al. [Accelerating Multigrid-based Hierarchical Scientific Data Refactoring on GPUs.][gpu] *35th IEEE International Parallel & Distributed Processing Symposium*, May. 2021.

### System Optimizations
* Vladislav Esaulov et al. [JANUS: Resilient and Adaptive Data Transmission for Enabling Timely and Efficient Cross-Facility Scientific Workflows.][janus] *Arxiv*, Jun. 2025
* Lipeng Wan et al. [RAPIDS: Reconciling Availability, Accuracy, and Performance in Managing Geo-Distributed Scientific Data.][rapids] *The International ACM Symposium on High-Performance Parallel and Distributed Computing*, Jun. 2023
* Xinying Wang et al. [Unbalanced Parallel I/O: An Often-Neglected Side Effect of Lossy Scientific Data Compression.][unbalanced-io] *7th International Workshop on Data Analysis and Reduction for Big Scientific Data*, Nov. 2021

[thesis]: https://doi.org/10.26300/ya1v-hn97
[mgard-softwarex]: https://doi.org/10.1016/j.softx.2023.101590
[univariate]: https://doi.org/10.1007/s00791-018-00303-9
[multivariate]: https://doi.org/10.1137/18M1166651
[quantities]: https://doi.org/10.1137/18M1208885
[unstructured]: https://doi.org/10.1137/19M1267878
[gpu]: https://ieeexplore.ieee.org/abstract/document/9460526/
[gpu2]: https://arxiv.org/abs/2105.12764
[gpu3]: https://ieeexplore.ieee.org/document/11078565
[gpu4]: https://arxiv.org/abs/2609.00205
[mgard+]: https://ieeexplore.ieee.org/abstract/document/9479913/
[unbalanced-io]: https://ieeexplore.ieee.org/abstract/document/9652573/
[mdr]: https://dl.acm.org/doi/abs/10.1145/3458817.3476179
[hp-mdr]: https://dl.acm.org/doi/10.1145/3712285.3759845
[roi]: https://link.springer.com/chapter/10.1007/978-3-030-96498-6_2
[roi2]: https://dl.acm.org/doi/abs/10.1145/3538712.3538717
[pp]: https://www.mdpi.com/1709018 
[pp3]: https://arxiv.org/abs/2212.10733
[qoi]: https://ieeexplore.ieee.org/abstract/document/10793162
[rapids]: https://dl.acm.org/doi/10.1145/3588195.3592983
[progressive-dnn]: https://ieeexplore.ieee.org/document/10184595/
[qoi2]: https://ieeexplore.ieee.org/document/10106324
[em-qoi]: https://doi.org/10.1093/mam/ozag084
[stability-pde]: https://dl.acm.org/doi/10.1145/3712285.3759878
[shadow-pde]: https://doi.org/10.1016/j.jocs.2026.102986
[janus]: https://arxiv.org/abs/2506.17084
[visibilities]: https://doi.org/10.1017/pasa.2025.29
[qpror]: https://doi.org/10.1145/3806645.3807579
[online-qoi]: https://ieeexplore.ieee.org/document/10254934/
[climate-qoi]: https://arxiv.org/abs/2401.03317
[fast-algo]: https://doi.org/10.1109/HiPC58850.2023.00030
[unstructured-framework]: https://ieeexplore.ieee.org/document/10678699/




