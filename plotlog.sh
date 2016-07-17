#!/usr/bin/env sh

rm -f ./siamese_train_s.log.train
rm -f ./siamese_train_s.log.test

F="siamese_train_06_test_89_feat_2_sim.log"
echo $F

./tools/extra/parse_log.py $F .
python plot_siamese.py $F
