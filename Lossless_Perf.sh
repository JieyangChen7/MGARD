#!/bin/bash

MdrXExec=mdr-x

DATA_DIR=/projects/cdux/jieyang/data
VERBOSE=3;


DATA=$DATA_DIR/SDRBENCH-EXASKY-NYX-512x512x512/temperature.f32
$MdrXExec --refactor --input $DATA --output refactored.mgard -dt s -dim 3 512 512 512 -dd max-dim -d $1 -v $VERBOSE
$MdrXExec --reconstruct --input refactored.mgard -o reconstructed.mgard -g $DATA -dt s -dim 3 512 512 512 -m abs -me 3 \
$(echo "4783000.2417 * 0.1" | bc) \
$(echo "4783000.2417 * 0.01" | bc) \
$(echo "4783000.2417 * 0.001" | bc) \
-s inf -ar 0 -d $1 -v $VERBOSE

DATA=$DATA_DIR/SDRBENCH-SCALE_98x1200x1200/PRES-98x1200x1200.f32
$MdrXExec --refactor --input $DATA --output refactored.mgard -dt s -dim 3 98 1200 1200 -dd max-dim -d $1 -v $VERBOSE
$MdrXExec --reconstruct --input refactored.mgard -o reconstructed.mgard -g $DATA -dt s -dim 3 98 1200 1200 -m abs -me 3 \
$(echo "101820.218750 * 0.1" | bc) \
$(echo "101820.218750 * 0.01" | bc) \
$(echo "101820.218750 * 0.001" | bc) \
-s inf -ar 0 -d $1 -v $VERBOSE

DATA=$DATA_DIR/SDRBENCH-Miranda-256x384x384/velocityz.d64
$MdrXExec --refactor --input $DATA --output refactored.mgard -dt d -dim 3 256 384 384 -dd max-dim -d $1 -v $VERBOSE
$MdrXExec --reconstruct --input refactored.mgard -o reconstructed.mgard -g $DATA -dt d -dim 3 256 384 384 -m abs -me 3 \
$(echo "8.996110 * 0.1" | bc) \
$(echo "8.996110 * 0.01" | bc) \
$(echo "8.996110 * 0.001" | bc) \
 -s inf -ar 0 -d $1 -v $VERBOSE

DATA=$DATA_DIR/100x500x500/Pf48.bin.f32
$MdrXExec --refactor --input $DATA --output refactored.mgard -dt s -dim 3 100 500 500 -dd max-dim -d $1 -v $VERBOSE
$MdrXExec --reconstruct --input refactored.mgard -o reconstructed.mgard -g $DATA -dt s -dim 3 100 500 500 -m abs -me 3 \
$(echo "3411.740723 * 0.1" | bc) \
$(echo "3411.740723 * 0.01" | bc) \
$(echo "3411.740723 * 0.001" | bc) \
-s inf -ar 0 -d $1 -v $VERBOSE

