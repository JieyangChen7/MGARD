#!/usr/bin/env python3

"""Generate a block-level tolerance map for the MGARD-X hybrid hierarchy."""

import argparse
import math
from array import array
from itertools import product
from pathlib import Path


BLOCK_SIZE = 8


def positive_tolerance(value):
    tolerance = float(value)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise argparse.ArgumentTypeError("tolerances must be finite and positive")
    return tolerance


def linear_index(coordinate, shape):
    index = 0
    for position, extent in zip(coordinate, shape):
        index = index * extent + position
    return index


def parse_roi(tokens, shape, parser):
    expected = 1 + 2 * len(shape)
    if len(tokens) != expected:
        parser.error(
            "each --roi requires TOL followed by one START END pair per "
            f"dimension ({expected} values for {len(shape)}D data)"
        )

    try:
        tolerance = positive_tolerance(tokens[0])
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    bounds = []
    for dimension, extent in enumerate(shape):
        try:
            start = int(tokens[1 + 2 * dimension])
            end = int(tokens[2 + 2 * dimension])
        except ValueError:
            parser.error("ROI bounds must be integers")
        if start < 0 or end <= start or end > extent:
            parser.error(
                f"ROI dimension {dimension} must satisfy "
                f"0 <= START < END <= {extent}"
            )
        bounds.append((start, end))
    return tolerance, bounds


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate the raw float64 tolerance map consumed by MGARD-X "
            "hybrid ROI compression."
        )
    )
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument(
        "-dim",
        "--shape",
        required=True,
        nargs="+",
        type=int,
        metavar="N",
        help="1D-3D data shape in slowest-to-fastest dimension order",
    )
    parser.add_argument(
        "-bg",
        "--background",
        required=True,
        type=positive_tolerance,
        metavar="TOL",
        help="tolerance assigned to blocks outside every ROI",
    )
    parser.add_argument(
        "-roi",
        "--roi",
        action="append",
        default=[],
        nargs="+",
        metavar="VALUE",
        help="TOL START0 END0 [START1 END1 ...]; may be repeated",
    )
    args = parser.parse_args()

    if not 1 <= len(args.shape) <= 3:
        parser.error("the hybrid hierarchy supports 1D-3D data only")
    if any(extent <= 0 for extent in args.shape):
        parser.error("all shape extents must be positive")

    block_shape = [
        (extent + BLOCK_SIZE - 1) // BLOCK_SIZE for extent in args.shape
    ]
    total_blocks = math.prod(block_shape)
    tolerances = array("d", [args.background]) * total_blocks

    parsed_rois = [parse_roi(tokens, args.shape, parser) for tokens in args.roi]
    for tolerance, bounds in parsed_rois:
        block_ranges = []
        for start, end in bounds:
            first = start // BLOCK_SIZE
            past_last = (end + BLOCK_SIZE - 1) // BLOCK_SIZE
            block_ranges.append(range(first, past_last))
        for coordinate in product(*block_ranges):
            index = linear_index(coordinate, block_shape)
            tolerances[index] = min(tolerances[index], tolerance)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        tolerances.tofile(output)

    print("Data shape:", " x ".join(map(str, args.shape)))
    print("Block shape:", " x ".join(map(str, block_shape)))
    print("Block size:", BLOCK_SIZE)
    print("ROI regions:", len(parsed_rois))
    print(f"Wrote {total_blocks} float64 values to {args.output}")


if __name__ == "__main__":
    main()
