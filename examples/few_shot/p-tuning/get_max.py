import os
import sys
import numpy as np

if __name__=="__main__":

    strategy = sys.argv[1]

    dev_accs = []
    #test_accs = []
    epochs = []
    iters = []
    steps = []


    dev_num = 0
    #test_num = 0

    for line in sys.stdin:
        #epoch, dev_acc, dev_num, _, test_acc, test_num = line.strip().split("\t")
        if not line.startswith("epoch"):
            index = line.index("epoch")
            line = line[index:]
        epoch, global_step, dev_acc, dev_num, iter_num, last_train = line.strip().split(",")

        epoch = int(epoch.split(":")[1])
        dev_acc = float(dev_acc.split(":")[1])
        dev_num = int(dev_num.split(":")[1])

        global_step = int(global_step.split(":")[1])
        iter_num = int(iter_num.split(":")[1])

        epochs.append(epoch)
        dev_accs.append(dev_acc)
        iters.append(iter_num)
        steps.append(global_step)
        #test_accs.append(test_acc)

        #print("{}\t{}\t{}\t{}\t{}\t{}".format(strategy, epoch, dev_acc, test_acc, dev_num, test_num), file=sys.stderr)
        print("{}\t{}\t{}\t{}\t{}\t{}".format(strategy, epoch, dev_acc, dev_num, iter_num, global_step), file=sys.stderr)

    max_dev_index = np.argmax(dev_accs)
    
    max_dev_acc = dev_accs[max_dev_index]
    #test_acc = test_accs[max_dev_index]
    epoch = epochs[max_dev_index]
    max_iter = iters[max_dev_index]
    max_step = steps[max_dev_index]
    
    print("{}\t{}\t{}\t{}\t{}\t{}".format(strategy, epoch, max_dev_acc, dev_num, max_iter, max_step))
    print("***********************************************************", file=sys.stderr)
    print("{}\t{}\t{}\t{}\t{}\t{}".format(strategy, epoch, max_dev_acc, dev_num, max_iter, max_step), file=sys.stderr)
