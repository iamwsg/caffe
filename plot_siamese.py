#./tools/extra/parse_log.py siamese_train.log .
import sys
import pandas as pd
import matplotlib.pyplot as plt

#print sys.argv[1]
train_file = str(sys.argv[1])+".train";
test_file = str(sys.argv[1])+".test";
#print train_file

train_log = pd.read_csv(train_file)
test_log = pd.read_csv(test_file)
_, ax1 = plt.subplots(figsize=(15, 10))
ax2 = ax1.twinx()
ax1.plot(train_log["NumIters"], train_log["loss"], alpha=0.4)
ax1.plot(test_log["NumIters"], test_log["loss"], 'g')
ax2.plot(test_log["NumIters"], test_log["accuracy"], 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('accuracy')
plt.grid()
plt.show()
