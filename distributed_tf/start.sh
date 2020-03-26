#!/bin/bash

python distributed_tf.py --ps_hosts "localhost:2222" --worker_hosts "localhost:2223,localhost:2224" --job_name "ps" --task_index 0 &
python distributed_tf.py --ps_hosts "localhost:2222" --worker_hosts "localhost:2223,localhost:2224" --job_name "worker" --task_index 0 &
python distributed_tf.py --ps_hosts "localhost:2222" --worker_hosts "localhost:2223,localhost:2224" --job_name "worker" --task_index 1 &
