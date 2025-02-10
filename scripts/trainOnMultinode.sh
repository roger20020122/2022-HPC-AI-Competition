#!/bin/bash

# change the path to your own directory
#path="/home/552/lh9988/HPC/moguy/2022-APAC-HPC-AI/Deep_Learning_Based_DNA_Sequence_Fast_Decoding/"
#output_path="/home/552/lh9988/HPC/moguy/2022-APAC-HPC-AI/Deep_Learning_Based_DNA_Sequence_Fast_Decoding/outputs"

## To train a single model with two gpus
#horovodrun -np ${PBS_NGPUS} --verbose --gloo -H ${host_flag} -p 1212 python3 $path/deep_tf.py -m unet
#horovodrun -np 3 --verbose --timeline-filename $output_path/unet_timeline.json python3 $path/deep_tf.py -m unet
#horovodrun -np 8 -H gadi-gpu-v100-0020:4,gadi-gpu-v100-0021:4 --verbose --timeline-filename $output_path/unet_timeline.json python3 $path/deep_tf.py -m unet
mpirun -np ${PBS_NGPUS} --map-by node  --bind-to socket python3 ./deep_tf.py -m unet

## To train multiple models
# array=( cnn unet se_cnn )
# for i in "${array[@]}"
# do
# 	horovodrun -np 2 --timeline-filename $output_path/"$i"_timeline.json python3 "$path"/deep_tf.py -m $i
# done

