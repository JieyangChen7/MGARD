#!/bin/bash

MdrXExec=mdr-x

DATA_DIR=/projects/cdux/jieyang/data
VERBOSE=0;


DATA=$DATA_DIR/SDRBENCH-EXASKY-NYX-512x512x512/temperature.f32
$MdrXExec --refactor --input $DATA --output refactored.mgard -dt s -dim 3 512 512 512 -dd max-dim -d $1 -v $VERBOSE
$MdrXExec --reconstruct --input refactored.mgard -o reconstructed.mgard -g none -dt s -dim 3 512 512 512 -m abs -me 1 1e-3 -s inf -ar 0 -d $1 -v $VERBOSE
