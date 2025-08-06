#!/bin/bash
#SBATCH -A CSC143
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --gpu-bind=closest
#SBATCH -J GPUJHTDB
#SBATCH -o GPUJHTDB.out
# Suppose you've successfully downloaded and sliced data into JHTDB (1024x2048x2048, [1536:2560, 1024:3072, 1024:3072] from 4096x4096x4096 isotropic4096 data)
# You have 8 GPUs, each GPU has over 64 GB memory
# Slice the JHTDB VelocityX, VelocityY, VelocityZ into 512x1024x1024 VelocityX[0~7].dat, VelocityY[0~7].dat, VelocityZ[0~7].dat
# Then concat them into VelocityXYZ[0~7].dat
# E.g. "cat VelocityX0.dat VelocityY0.dat VelocityZ0.dat > VelocityXYZ0.dat"
# You have a directory named JHTDB with VelocityXYZ[0~7].dat in it exists under current directory.
# make sure you have enough space (~128GB) to store JHTDB and refactored data

set -x

ml rocm/6.3.1
ml hdf5
module load cmake

a1=0.1
r=0.1
n=5
error_bounds=()

a=$a1
for ((i = 1; i <= n; i++)); do
    error_bounds+=($a)
    a=$(echo "scale=10; $a * $r" | bc)
done

error_bounds=($(printf "%s\n" "${error_bounds[@]}" | sort -nr))

build_dir=$(ls | grep '^build-' | head -n 1)
IFS='-' read -r _ device _ <<< "$build_dir"
echo "$device"
exe="./$build_dir/mgard/bin/pmdr-x-qoi"
ioexe="./$build_dir/mgard/bin/pmdr-x-qoi-io"

output_file="JHTDB_output.txt"
tmp_file="JHTDB_tmp.txt"
>$output_file
>$tmp_file

# salloc -A CSC143 -J test -t 0:59:00 -p batch -N 1 --ntasks-per-node=1 --gpus-per-node=1 --gpu-bind=closest

SRUN="srun -A CSC143 -N 1 -n 8 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest"

# mpirun -n 2 $exe --refactor --input ~/Polaris/Dataset/mini_NYX/data/VelocityXYZ --output ~/Polaris/MGARD/build-cuda-turing/mgard/miniNYX/XYZ -dt s -dim 3 768 256 256 -dd max-dim -d $device -v 0

$SRUN $exe --refactor --input JHTDB/VelocityXYZ --output JHTDB/XYZ -dt s -dim 3 1536 1024 1024 -dd max-dim -d $device -v 0 > $tmp_file
time=$(grep "max_elapsed_time" $tmp_file | head -n 1)
echo "Refactor: $time" >> $output_file

for error_bound in "${error_bounds[@]}"; do
    # $SRUN $exe --reconstruct -i JHTDB/XYZ -o sda -g JHTDB/VelocityXYZ -dt s -dim 3 1536 1024 1024 -m abs -e $error_bound -s inf -ar 0 -d $device -v 0 -dm 4 >> $output_file
    $SRUN $exe --reconstruct -i JHTDB/XYZ -o sda -g JHTDB/VelocityXYZ -dt s -dim 3 1536 1024 1024 -m abs -e $error_bound -s inf -ar 0 -d $device -v 0 -dm 4 > $tmp_file
    bitrate=$(grep "Bitrate" $tmp_file | head -n 1)
    kerneltime=$(grep "max_kernel_time" $tmp_file | head -n 1)
    time=$(grep "max_elapsed_time" $tmp_file | head -n 1)
    requested_max_error=$(grep "Requested_Tau" $tmp_file | head -n 1)
    est_max_error=$(grep "Est_max_error" $tmp_file | head -n 1)
    real_max_error=$(grep "Real_max_error" $tmp_file | head -n 1)
    $SRUN $ioexe --reconstruct -i JHTDB/XYZ -o sda -g JHTDB/VelocityXYZ -dt s -dim 3 1536 1024 1024 -m abs -e $error_bound -s inf -ar 0 -d $device -v 0 -dm 4 > $tmp_file
    readtime=$(grep "IO_time" $tmp_file | head -n 1)
    echo "Request eb = $error_bound, $bitrate, $readtime, $kerneltime, $time, $requested_max_error, $est_max_error, $real_max_error" >> $output_file
done

cat $output_file
rm $tmp_file
rm $output_file
