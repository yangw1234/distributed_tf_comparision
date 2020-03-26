from __future__ import print_function
import tensorflow as tf
import tensorflow_datasets as tfds

def map_func(data):
    image = data['image']
    label = data['label']

    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int64)
    return (image, label)


def create_dataset(batch_size):
    ds = tfds.load("mnist", split="train")
    ds = ds.map(map_func)
    dataset = ds.repeat().shuffle(1000).batch(batch_size)
    return dataset


def create_model(dataset):
    iter = dataset.make_one_shot_iterator()
    images, labels = iter.get_next()
    flatten = tf.layers.flatten(images)
    logits = tf.layers.dense(flatten, 10)
    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))
    return loss


def main():
    global_batch_size = 256
    global_step = tf.train.get_or_create_global_step()

    # build model
    loss = create_model(create_dataset(global_batch_size))

    optimizer = tf.train.GradientDescentOptimizer(.0001)
    opt = optimizer.minimize(loss, global_step=global_step)
    stop_hook = tf.train.StopAtStepHook(last_step=20)
    hooks = [stop_hook]
    sess = tf.train.MonitoredTrainingSession(hooks=hooks)
    while not sess.should_stop():
        _, r, gs = sess.run([opt, loss, global_step])
        print("step {}, loss {}".format(gs, r))
    sess.close()

if __name__ == '__main__':
    main()
