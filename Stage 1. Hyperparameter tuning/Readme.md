# Task 3. Deep Learning Based DNA Sequence Fast Decoding - Stage 1. Hyperparameter tuning

To find out the optimal configuration of the U-Net model we used in the task, we perform automatic hyperparameter tuning with the Hyperopt python library.

This code base contains a complete training procedure and needed utilitiy files. We use PyTorch as the deep learning framwork on this stage.

The model architecture is the Leopard U-Net from https://github.com/GuanLab/Leopard. We picked hyperparameters from the model to be tuned:

- dropout: the dropout rate of all dropout layers
- initial_filter: the number of filters in the top U-Net block
- kernel_size: convolution kernel size
- num_blocks: number of U-Net blocks
- pos_weight: weighting of positive samples in the loss function
- scale_filter: the factor (number of filters in the $n^{th}$ layer) / (number of filters in the $n-1^{th}$ layer)

The specific range of the search space is defined in the file `optim_hyp.py`.

## Installation

```
conda create -n "dna2" python=3.9.2
conda activate dna2
pip install -r requirements.txt 
```

## Usage

1. Convert the DNA dataset to a format which is compatible to torch. The output will be place at `data/`

    `python convert.py <path_to_dataset>/preprocessed_CTCF_fimo_data/`

2. Submit the `jobs/single.sh` job script. 

    `qsub jobs/single.sh`

    The job lanches `optim.py`, which is the script doing hyperparameter tuning. In the script, we use hyperopt.fmin optimizer to search for the best model. The fmin optimizer will repeatly evaluate the `objective` function, which calls `train_dist.py` to train the model and reports the PRAUC score of each trials back to fmin.

    > The process of hyperparameter tuning runs on a single GPU. We've tried hard to make the distributed version of code run on GADI, but it always hangs at the line calling function `torch.nn.parallel.DistributedDataParallel()`.

3. Inspect the real-time progress with Tensorboard

    `tensorboard --logdir log`

    The command launches a tensorboard instance watching tensorboard log files in `log/`. Access it with a browser, for example, `http://localhost:6006`. It will visualize the real-time training progress. You can see the highest points of PRAUC score curves increases as more trials be issued.

4. Retrieve the optimal hyperparameters

    After about 100 trials, you can stop the job and look at the tensorboard to get the optimal hyperparameters. Select the trial that achieved the highest PRAUC, and see the value of the hyperparameters of this trials from its name.

5. Continue to the second stage of our work.
