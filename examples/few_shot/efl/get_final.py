import os
import sys
import numpy as np

if __name__ == "__main__":
    result_file = sys.argv[1]

    dev_accs = []
    test_accs = []
    strategys = []
    epochs = []
    strategys = []

    dev_num = 0
    test_num = 0

    with open(result_file) as f:
        for line in f:
            #strategy_name, epoch, dev_acc, test_acc, dev_num, test_num = line.strip().split("\t")
            strategy_name, epoch, dev_acc, dev_num = line.strip().split("\t")

            strategys.append(strategy_name)
            dev_accs.append(float(dev_acc))
            #test_accs.append(float(test_acc))
            epochs.append(epoch)

    print("dev_accs:{}".format(dev_accs))
    #print("test_accs:{}".format(test_accs))

    max_index = np.argmax(dev_accs)
    max_dev_acc = dev_accs[max_index]
    #test_acc = test_accs[max_index]

    strategy = strategys[max_index]
    epoch = epochs[max_index]

    #print("{}\t{}\t{}\t{}\t{}\t{}".format(strategy, epoch, max_dev_acc, test_acc, dev_num, test_num))
    print("{}\t{}\t{}\t{}".format(strategy, epoch, max_dev_acc, dev_num))
