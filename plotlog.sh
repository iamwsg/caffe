#!/usr/bin/env sh

rm -f ./scene_multi.log.train
rm -f ./scene_multi.log.test

F="siamese_train_06_test_89_feat_2_sim.log"
#F="scene_multi.log"
#F="met.log"
echo $F

./tools/extra/parse_log.py $F .
python plot_siamese.py $F
