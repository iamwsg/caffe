#!/usr/bin/env sh

rm -f ./siamese_train.log.train
rm -f ./siamese_train.log.test

./tools/extra/parse_log.py siamese_train.log .
python plot_siamese.py
