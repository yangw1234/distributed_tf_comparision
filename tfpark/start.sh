#!/bin/bash

bash $ANALYTICS_ZOO_HOME/bin/spark-submit-python-with-zoo.sh --master local[2] --driver-memory 2g tfpark.py
