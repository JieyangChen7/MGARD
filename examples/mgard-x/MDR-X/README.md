# Refactor and progressively reconstruct data with MDR-X

First, build and install MGARD-X.
Then, run the following in `examples/mgard-x/MDR-X/SERIAL`, `examples/mgard-x/MDR-X/CUDA`, `examples/mgard-x/MDR-X/HIP`. Each folder contains a CMake project dedicated for a different kind of processor.

Build with CMake as follows or use the 'build_scripts.sh'.
```console
$ cmake -S . -B build
$ cmake --build build
$ build/refactor <args>
```

`build/refactor` reads in a dataset, refactors it with MDR-X on GPU or CPU, and then progressively reconstructs it in-memory at each of the given error bounds, printing the reconstruction error against the original data.
Read `refactor.cpp/refactor.cu` to see how the MDR-X API (`mgard_x::MDR::ComposedRefactor`/`ComposedReconstructor`) is used.

The `refactor` executable takes:

* `<input data>`
* `<number of decomposition levels>`
* `<number of bitplanes>`
* `<number of dimensions N> <dim 1> <dim 2> .. <dim N>` (currently 3D only)
* `<number of tolerances M> <tol 1> <tol 2> ... <tol M>`: L-infinity error bounds to reconstruct at, one after another
* `<s>`: smoothness parameter (use `0` for L2-style error control)

Example: `build/refactor data.bin 3 32 3 64 64 64 3 0.1 0.01 0.001 0`
