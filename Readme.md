# Deep Learning Based DNA Sequence Fast Decoding
This project originates from the 2022 APAC HPC AI Competition, specifically from the problem 3.

Our work on this task consists of 2 stages:
1. Hyperparameter tuning
2. GPU cluster setting tuning

First, we search the optimal modification of the Leopard U-Net. The objective of this stage is to make the model achieve as high PRAUC as possible. After that, we move the optimal model to the second stage and compare the efficiency between different GPU cluster settings. 
## Setup
```shell
$ conda create -n "dna2" python=3.9.2
$ conda activate dna2
$ pip install -r requirements.txt 
```
## Hyperparameter Tuning and GPU Cluster Setting Tuning
### Hyperparameter Tuning
To find out the optimal configuration of the U-Net model we used in the task, we perform automatic hyperparameter tuning with the Hyperopt python library.

This code base contains a complete training procedure and needed utilitiy files. We use PyTorch as the deep learning framwork on this stage.

The model architecture is the Leopard U-Net from https://github.com/GuanLab/Leopard. We picked hyperparameters from the model to be tuned:

1. `dropout`: the dropout rate of all dropout layers
2. `initial_filter`: the number of filters in the top U-Net block
3. `kernel_size`: convolution kernel size
4. `num_blocks`: number of U-Net blocks
5. `pos_weight`: weighting of positive samples in the loss function
6. `scale_filter`: the factor (number of filters in the $n^{th}$ layer) / (number of filters in the $n-1^{th}$ layer)

The specific range of the search space is defined in the file `deepLearningBasedDNASequenceFastDecoding/optim/optim_hyp.py`.
### Usage
1. Step 1: Converting the DNA dataset to a format which is compatible to torch. The output will be place at `data/`
    ```shell
    $ python deepLearningBasedDNASequenceFastDecoding/data/convert.py deepLearningBasedDNASequenceFastDecoding/data/preprocessedCTCFFimoData/
    ```
2. Step 2: Submitting the `script/optim.sh` job script. 
    ```
    $ qsub script/optim.sh
    ```
    The script lanches `optim.py`, which is the script doing hyperparameter tuning. In the script, we use hyperopt.fmin optimizer to search for the best model. The fmin optimizer will repeatly evaluate the `objective` function, which calls `train_dist.py` to train the model and reports the PRAUC score of each trials back to fmin.
3. Step 3: Inspectting the real-time progress with Tensorboard
    ```
    $ tensorboard --logdir log
    ```
    The command launches a tensorboard instance watching tensorboard log files in `log/`. Access it with a browser, for example, `http://localhost:6006`. It will visualize the real-time training progress. You can see the highest points of PRAUC score curves increases as more trials be issued.

4. Step 4: Retrieving the optimal hyperparameters
    After about 100 trials, you can stop the job and look at the tensorboard to get the optimal hyperparameters. Select the trial that achieved the highest PRAUC, and see the value of the hyperparameters of this trials from its name.
### GPU Cluster Setting Tuning
#### Submitting Jobs
```
qusb scripts/trainOnMultinode.sh
```