# MGARD-X Region-of-Interest (ROI) Compression

MGARD-X supports a different L<sup>&infin;</sup> error tolerance for each block of the hybrid hierarchy. This allows important regions to use a tighter error bound while the background uses a looser error bound and achieves a higher compression ratio.

## Supporting features

* **Dimensions:** 1D-3D
* **Error-bound type:** L<sup>&infin;</sup>
* **Error-bound mode:** Absolute and relative
* **Data structure:** Uniform Cartesian grid
* **Interfaces:** Command line interface and high-level APIs

The ROI feature is not currently exposed through the public low-level API. Compression must use the hybrid hierarchy, and the input must be processed as one subdomain. Domain decomposition does not currently apply a global ROI map correctly.

## ROI tolerance map

The input map contains one tolerance for every block at the finest block-local level. MGARD-X currently uses a block edge length of 8, so data with shape

```text
N0 N1 ... ND-1
```

requires a map with

```text
ceil(N0 / 8) * ceil(N1 / 8) * ... * ceil(ND-1 / 8)
```

entries. The file format is a headerless binary array of native-endian IEEE 754 ```double``` values. Entries use row-major order: the last dimension is the fastest-varying dimension. For example, a 3D block coordinate ```(b0, b1, b2)``` is stored at

```text
(b0 * B1 + b1) * B2 + b2
```

where ```Bd = ceil(Nd / 8)```. In relative mode, each value is a relative tolerance with respect to the L<sup>&infin;</sup> norm of the complete dataset. In absolute mode, each value is an absolute tolerance.

Only the finest-level map is provided by the user. When multiple local levels are used, MGARD-X derives the coarser maps by propagating the minimum tolerance of the contributing fine blocks. The global stage also uses the minimum propagated tolerance.

## Generate a tolerance map

The repository provides a zero-dependency [ROI map generator][roi-generator]. It initializes every block with a background tolerance and then applies one or more voxel-space ROI boxes. Coordinates follow the same slowest-to-fastest dimension order as the MGARD-X shape, and every ```START END``` interval is half-open: ```START``` is included and ```END``` is excluded.

For example, the following command creates a map for a ```512 x 512 x 512``` dataset. The background tolerance is ```1e-2```, while voxels in ```[200, 300) x [200, 300) x [200, 300)``` use ```1e-5```:

```console
$ python3 scripts/generate_mgard_x_roi_map.py \
    -o roi.bin -dim 512 512 512 -bg 1e-2 \
    -roi 1e-5 200 300 200 300 200 300
```

The ```-roi``` option may be repeated. Each ROI is expanded to cover every intersecting block whose edge length is 8, and overlapping regions use the smallest requested tolerance.

## Command line interface

Compress with the hybrid hierarchy, ROI mode, and ```s = inf```:

```console
$ mgard-x -z -i input.f32 -o compressed.mgard \
    -dt s -dim 3 512 512 512 \
    -em rel -r roi.bin -roi -s inf \
    -l huffman -d cuda -hh -ll 1 -gl 2
```

The ROI map, hierarchy levels, and projection mode are stored in the compressed-data metadata. Decompression therefore does not require the original map or the hybrid compression options:

```console
$ mgard-x -x -i compressed.mgard -o reconstructed.f32 -d cuda
```

Do not set a restrictive maximum-memory option that causes domain decomposition. The current ROI implementation expects the complete map to describe one subdomain.

## High-level APIs

Enable the hybrid hierarchy and copy the finest-level tolerance map into ```Config``` before compression:

```cpp
mgard_x::Config config;
config.decomposition = mgard_x::decomposition_type::Hybrid;
config.enable_roi = true;
config.roi_tolerance_map = tolerance_map; // std::vector<double>
config.num_local_refactoring_level = 1;
config.num_global_refactoring_level = 2;
config.hybrid_projection_mode =
    mgard_x::hybrid_projection_mode_type::Orthogonal;
```

The vector length and ordering must follow the binary map format described above. Pass this ```Config``` to the regular high-level ```mgard_x::compress``` API. The regular high-level ```mgard_x::decompress``` API restores the ROI configuration from metadata.

[roi-generator]: ../scripts/generate_mgard_x_roi_map.py
