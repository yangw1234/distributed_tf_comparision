from __future__ import print_function
import tensorflow as tf
import tensorflow_datasets as tfds
import horovod.tensorflow as hvd

FLAGS = None


def map_func(data):
    image = data['image']
    label = data['label']

    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int64)
    return (image, label)


def create_dataset(task_index, task_num, batch_size):
    ds = tfds.load("mnist", split="train")
    ds = ds.map(map_func)
    dataset = ds.shard(task_num, task_index).repeat().shuffle(1000).batch(batch_size // task_num)
    return dataset


def create_model(dataset):
    iter = dataset.make_one_shot_iterator()
    images, labels = iter.get_next()
    flatten = tf.layers.flatten(images)
    logits = tf.layers.dense(flatten, 10)
    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))
    return loss


def main():
    hvd.init()

    global_batch_size = 256
    global_step = tf.train.get_or_create_global_step()
    loss = create_model(create_dataset(hvd.rank(), hvd.size(), global_batch_size))
    # create an optimizer then wrap it with SynceReplicasOptimizer
    optimizer = tf.train.GradientDescentOptimizer(.0001)

    optimizer1 = hvd.DistributedOptimizer(optimizer, op=hvd.Average)

    opt = optimizer1.minimize(loss, global_step=global_step)  # averages gradients

    hooks = [
        # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
        # from rank 0 to all other processes. This is necessary to ensure consistent
        # initialization of all workers when training is started with random weights
        # or restored from a checkpoint.
        hvd.BroadcastGlobalVariablesHook(0),

        # Horovod: adjust number of steps based on number of GPUs.
        tf.train.StopAtStepHook(last_step=20),
    ]

    with tf.train.MonitoredTrainingSession(hooks=hooks) as mon_sess:
        while not mon_sess.should_stop():
            _, r, gs = mon_sess.run([opt, loss, global_step])
            print(r, 'step: ', gs)


if __name__ == '__main__':
    main()
