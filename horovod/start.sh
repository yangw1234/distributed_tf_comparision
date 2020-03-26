#!/bin/bash
horovodrun -np 2 -H localhost:2 python horovod_tf.py 
