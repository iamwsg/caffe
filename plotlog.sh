#!/usr/bin/env sh

rm -f ./siamese_train_s.log.train
rm -f ./siamese_train_s.log.test

F="scene_multi.log"
echo $F

./tools/extra/parse_log.py $F .
python plot_siamese.py $F
