#!/usr/bin/env sh

rm -f ./siamese_train_s.log.train
rm -f ./siamese_train_s.log.test

./tools/extra/parse_log.py siamese_train_s.log .
python plot_siamese.py
