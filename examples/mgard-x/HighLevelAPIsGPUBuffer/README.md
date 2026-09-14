# Compressing with MGARD-X High-level APIs (GPU-resident buffers)

First, build and install MGARD-X.
Then, run the following in `examples/mgard-x/HighLevelAPIsGPUBuffer`.

Build with CMake as follows or use the 'build_scripts.sh'.
```console
$ cmake -S . -B build
$ cmake --build build
$ build/Example
```


`build/Example` creates a dataset, compresses it with MGARD-X on NVIDIA GPU, and decomrpess it on CPU, passing GPU-resident buffers directly to the high-level API instead of host buffers.
Read `Example.cu` to see how the high-level compression API is used with GPU buffers.
