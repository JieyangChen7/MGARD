#!/bin/bash
# Suppose you've successfully downloaded and sliced data into mini-JHTDB
# You have a directory named mini-JHTDB with VelocityXYZ.dat in it exists under current directory.
# make sure you have enough space (~16GB) to store mini-JHTDB and refactored data
set -x
set -e

a1=0.1
a2=0.05
r=0.1
n=5
error_bounds=()

a=$a1
for ((i = 1; i <= n; i++)); do
    error_bounds+=($a)
    a=$(echo "scale=10; $a * $r" | bc)
done

a=$a2
for ((i = 1; i <= n; i++)); do
    error_bounds+=($a)
    a=$(echo "scale=10; $a * $r" | bc)
done

error_bounds=($(printf "%s\n" "${error_bounds[@]}" | sort -nr))

build_dir=$(ls | grep '^build-' | head -n 1)
IFS='-' read -r _ device _ <<< "$build_dir"
echo "$device"
exe="./$build_dir/mgard/bin/mdr-x-qoi"

output_file="mini-JHTDB_output.txt"
tmp_file="mini-JHTDB_tmp.txt"
>$output_file
>$tmp_file

# When you have a server and need to use slurm
# salloc -A CSC143 -J test -t 0:30:00 -p batch -N 1 --ntasks-per-node=1 --gpus-per-node=1 --gpu-bind=closest
# SRUN="srun -A CSC143 -N 1 -n 1 --ntasks-per-node=1 --gpus-per-node=1 --gpu-bind=closest"
# $SRUN $exe --refactor -i mini-JHTDB/VelocityXYZ.dat -o mini-JHTDB/XYZ -dt s -dim 3 1536 1024 1024 -dd max-dim -d $device -v 0

$exe --refactor -i mini-JHTDB/VelocityXYZ.dat -o mini-JHTDB/XYZ -dt s -dim 3 1536 1024 1024 -dd max-dim -d $device -v 0

method_names=("CP" "MA" "MAPE(c=2)" "MAPE(c=10)")

for ((i = 0; i < 4; i++)); do
    for error_bound in "${error_bounds[@]}"; do
        # $SRUN $exe --reconstruct -i mini-JHTDB/XYZ -o none -g mini-JHTDB/VelocityXYZ.dat -dt s -dim 3 1536 1024 1024 -m abs -e $error_bound -s inf -ar 0 -d $device -v 0 -dm $i > $tmp_file
        $exe --reconstruct -i mini-JHTDB/XYZ -o none -g mini-JHTDB/VelocityXYZ.dat -dt s -dim 3 1536 1024 1024 -m abs -e $error_bound -s inf -ar 0 -d $device -v 0 -dm $i > $tmp_file
        bitrate=$(grep "Bitrate" $tmp_file | head -n 1)
        time=$(grep "pipeline" $tmp_file | head -n 1)
        requested_max_error=$(grep "Requested_Tau" $tmp_file | head -n 1)
        est_max_error=$(grep "Est_max_error" $tmp_file | head -n 1)
        real_max_error=$(grep "Real_max_error" $tmp_file | head -n 1)
        method=${method_names[$i]}
        echo "$method, Request eb = $error_bound, $bitrate, $time, $requested_max_error, $est_max_error, $real_max_error" >> $output_file
    done
done

cat $output_file

rm $tmp_file
rm $output_file
