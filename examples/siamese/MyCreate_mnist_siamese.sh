#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

EXAMPLES=./build/examples/siamese
DATA=./data/mnist

echo "Creating leveldb..."

#rm -rf ./examples/siamese/mnist_siamese_train_leveldb_0to2
#rm -rf ./examples/siamese/mnist_siamese_test_leveldb_0to2

$EXAMPLES/MyConvert_mnist_siamese_data.bin \
    $DATA/train-images-idx3-ubyte \
    $DATA/train-labels-idx1-ubyte \
    ./examples/siamese/mnist_siamese_train_leveldb_0to3_2anorm

#$EXAMPLES/MyConvert_mnist_siamese_data.bin \
#    $DATA/t10k-images-idx3-ubyte \
#    $DATA/t10k-labels-idx1-ubyte \
#    ./examples/siamese/mnist_siamese_test_leveldb_0to3_2anorm

#$EXAMPLES/MyConvert_mnist_siamese_data_test.bin \
#    $DATA/t10k-images-idx3-ubyte \
#    $DATA/t10k-labels-idx1-ubyte \
#    ./examples/siamese/mnist_siamese_test_leveldb_89

echo "Done."
