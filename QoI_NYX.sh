#!/bin/bash
wget https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/EXASKY/NYX/SDRBENCH-EXASKY-NYX-512x512x512.tar.gz
tar -xzf SDRBENCH-EXASKY-NYX-512x512x512.tar.gz
cat SDRBENCH-EXASKY-NYX-512x512x512/velocity_x.f32 SDRBENCH-EXASKY-NYX-512x512x512/velocity_y.f32 SDRBENCH-EXASKY-NYX-512x512x512/velocity_z.f32 > SDRBENCH-EXASKY-NYX-512x512x512/VelocityXYZ.dat

set -x
set -e

ml rocm/6.3.1
ml hdf5
module load cmake

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


output_file="NYX_output.txt"
tmp_file="NYX_tmp.txt"
>$output_file
>$tmp_file

# salloc -A CSC143 -J test -t 0:30:00 -p batch -N 1 --ntasks-per-node=1 --gpus-per-node=1 --gpu-bind=closest
# SRUN="srun -A CSC143 -N 1 -n 1 --ntasks-per-node=1 --gpus-per-node=1 --gpu-bind=closest"
# $SRUN $exe --refactor -i SDRBENCH-EXASKY-NYX-512x512x512/VelocityXYZ.dat -o SDRBENCH-EXASKY-NYX-512x512x512/XYZ -dt s -dim 3 1536 512 512 -dd max-dim -d $device -v 0

$exe --refactor -i SDRBENCH-EXASKY-NYX-512x512x512/VelocityXYZ.dat -o SDRBENCH-EXASKY-NYX-512x512x512/XYZ -dt s -dim 3 1536 512 512 -dd max-dim -d $device -v 0

method_names=("CP" "MA" "MAPE(c=2)" "MAPE(c=10)")

for ((i = 0; i < 4; i++)); do
    for error_bound in "${error_bounds[@]}"; do
        # $SRUN $exe --reconstruct -i SDRBENCH-EXASKY-NYX-512x512x512/XYZ -o none -g SDRBENCH-EXASKY-NYX-512x512x512/VelocityXYZ.dat -dt s -dim 3 1536 512 512 -m abs -e $error_bound -s inf -ar 0 -d $device -v 0 -dm $i > $tmp_file
        $exe --reconstruct -i SDRBENCH-EXASKY-NYX-512x512x512/XYZ -o none -g SDRBENCH-EXASKY-NYX-512x512x512/VelocityXYZ.dat -dt s -dim 3 1536 512 512 -m abs -e $error_bound -s inf -ar 0 -d $device -v 0 -dm $i > $tmp_file
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
