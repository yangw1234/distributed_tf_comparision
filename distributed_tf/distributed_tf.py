from __future__ import print_function
import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
import time

FLAGS = None
log_dir = '/logdir'
REPLICAS_TO_AGGREGATE = 2


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
    # Configure

    # Server Setup
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    num_task = len(worker_hosts)

    global_batch_size = 256

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    if FLAGS.job_name == 'ps':  # checks if parameter server
        server = tf.train.Server(cluster,
                                 job_name="ps",
                                 task_index=FLAGS.task_index)
        server.join()
    else:  # it must be a worker server
        is_chief = (FLAGS.task_index == 0)  # checks if this is the chief node
        server = tf.train.Server(cluster,
                                 job_name="worker",
                                 task_index=FLAGS.task_index)

        # Graph
        worker_device = "/job:%s/task:%d/cpu:0" % (FLAGS.job_name, FLAGS.task_index)
        with tf.device(tf.train.replica_device_setter(ps_tasks=1,
                                                      worker_device=worker_device)):

            global_step = tf.train.get_or_create_global_step()
            loss = create_model(create_dataset(FLAGS.task_index, num_task, global_batch_size))

            # create an optimizer then wrap it with SynceReplicasOptimizer
            optimizer = tf.train.GradientDescentOptimizer(.0001)
            optimizer = tf.train.SyncReplicasOptimizer(optimizer,
                                                        replicas_to_aggregate=REPLICAS_TO_AGGREGATE,
                                                        total_num_replicas=2)

            opt = optimizer.minimize(loss, global_step=global_step)

        # Session
        sync_replicas_hook = optimizer.make_session_run_hook(is_chief)
        stop_hook = tf.train.StopAtStepHook(last_step=20)
        hooks = [sync_replicas_hook, stop_hook]

        # Monitored Training Session
        sess = tf.train.MonitoredTrainingSession(master=server.target,
                                                 is_chief=is_chief,
                                                 hooks=hooks,
                                                 stop_grace_period_secs=10)

        print('Starting training on worker %d' % FLAGS.task_index)
        while not sess.should_stop():
            _, r, gs = sess.run([opt, loss, global_step])
            print(r, 'step: ', gs, 'worker: ', FLAGS.task_index)
        sess.close()
        print('Session from worker %d closed cleanly' % FLAGS.task_index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )

    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS.task_index)
    main()
