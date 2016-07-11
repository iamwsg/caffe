#!/usr/bin/env sh

rm -f ./siamese_train_s.log.train
rm -f ./siamese_train_s.log.test

~/mywork/caffe/tools/extra/parse_log.py lenet_train_fine3.log .
python plot_lenet.py
