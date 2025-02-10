import tensorflow as tf
import torch
import sys
from os.path import join

def tf_ds_to_torch(inp,out):
    ds = tf.data.experimental.load(inp)
    a=ds.batch(10000000).as_numpy_iterator().next()
    torch.save(a,out,pickle_protocol=4)

src = sys.argv[1]

tf_ds_to_torch(join(src,'test_data_0'),'data/test')
tf_ds_to_torch(join(src,'train_data_0'),'data/train')
tf_ds_to_torch(join(src,'validation_data_0'),'data/val')