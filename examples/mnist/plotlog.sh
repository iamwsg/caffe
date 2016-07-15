#!/usr/bin/env sh

#rm -f ./siamese_train_s.log.train
#rm -f ./siamese_train_s.log.test

~/caffe/tools/extra/parse_log.py ~/caffe/examples/mnist/lenet_train_s.log .
python ~/caffe/examples/mnist/plot_lenet.py
