# Stage2: GPU cluster setting tuning

# Table of contents
1. [Conda](#paragraph1)
2. [Jobs and Submission](#paragraph2)

## Prerequisites <a name="paragraph1"></a>

### 1. Conda <a name="subparagraph1"></a>

Download Conda installer from [here](https://www.anaconda.com/products/distribution). For Gadi, please use [64-Bit (x86) Installer](https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh)

To install package "numpy":

    conda install numpy

To setup conda environment for the challenge:

    conda env create --name leopard --file environment.yml
    conda activate leopard

## Jobs and Submission <a name="paragraph2"></a>

To specify the number of nodes, please modify the file multinodes_leopard.sh . The number of cpus must be 12 times the number of gpus. 

To specify the other parameters, please modify the file multinodes_train.sh, adding options such as --models, --batch_size, etc.. For instance, to train our model:

    mpirun -np ${PBS_NGPUS} --map-by node  --bind-to socket python3 ./deep_tf.py -m unet 

To specify the file path, please modify the file multinodes_train.sh to link to the file deep_tf.py and add an option --path to specify the path of the data.

To submit the job:

    qusb mutinodes_leopard.sh

