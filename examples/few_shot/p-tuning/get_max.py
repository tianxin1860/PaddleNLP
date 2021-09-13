import os
import sys
import numpy as np

if __name__=="__main__":

    strategy = sys.argv[1]

    dev_accs = []
    test_accs = []
    epochs = []

    dev_num = 0
    test_num = 0

    for line in sys.stdin:
        epoch, dev_acc, dev_num_and_epoch, test_acc, test_num = line.strip().split(",")

        epoch = int(epoch.split(":")[1])
        dev_acc = float(dev_acc.split(":")[1])
        dev_num = int(dev_num_and_epoch.split("\t")[0].split(":")[1])
        test_acc = float(test_acc.split(":")[1])
        test_num = int(test_num.split(":")[1])


        epochs.append(epoch)
        dev_accs.append(dev_acc)
        test_accs.append(test_acc)

        print("{}\t{}\t{}\t{}\t{}\t{}".format(strategy, epoch, dev_acc, test_acc, dev_num, test_num), file=sys.stderr)

    max_dev_index = np.argmax(dev_accs)

    max_dev_acc = dev_accs[max_dev_index]
    max_test_acc = test_accs[max_dev_index]
    max_epoch = epochs[max_dev_index]

    print("{}\t{}\t{}\t{}\t{}\t{}".format(strategy, max_epoch, max_dev_acc, dev_num, max_test_acc, test_num))
    print("***********************************************************", file=sys.stderr)
    print("{}\t{}\t{}\t{}\t{}\t{}".format(strategy, max_epoch, max_dev_acc, dev_num, max_test_acc, test_num), file=sys.stderr)