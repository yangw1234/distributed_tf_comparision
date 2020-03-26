from __future__ import print_function
import tensorflow as tf
import tensorflow_datasets as tfds
from zoo.tfpark import TFDataset, TFOptimizer
from zoo import init_nncontext
from bigdl.optim.optimizer import *

def map_func(data):
    image = data['image']
    label = data['label']

    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int64)
    return (image, label)


def creat_dataset(batch_size):
    ds = tfds.load("mnist", split="train")
    ds = ds.map(map_func)
    ds = ds.shuffle(1000)
    dataset = TFDataset.from_tf_data_dataset(ds, batch_size=batch_size)
    return dataset


def create_model(dataset):
    images, labels = dataset.tensors
    flatten = tf.layers.flatten(images)
    logits = tf.layers.dense(flatten, 10)
    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))
    return loss


def main():
    sc = init_nncontext()

    global_batch_size = 256

    loss = create_model(creat_dataset(global_batch_size))

    optimizer = TFOptimizer.from_loss(loss, SGD(1e-3),
                                      model_dir="/tmp/lenet/")
    optimizer.optimize(end_trigger=MaxIteration(20))


if __name__ == '__main__':
    main()
