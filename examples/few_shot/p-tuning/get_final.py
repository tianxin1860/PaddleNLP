import os
import sys
import numpy as np

from shutil import copyfile

if __name__=="__main__":
    result_file = sys.argv[1]
    output_dir = sys.argv[2]
    task_name = sys.argv[3]
    index = sys.argv[4]
    submit_result_dir = sys.argv[5]

    if not os.path.exists(submit_result_dir):
        os.mkdir(submit_result_dir)

    dev_accs = []
    #test_accs = []
    strategys = []
    epochs = []
    iters = []
    steps = []

    dev_num = 0
    #test_num = 0

    with open(result_file) as f:
        for line in f:
            strategy_name, epoch, dev_acc, dev_num, iter_num, global_step = line.strip().split("\t")

            strategys.append(strategy_name)
            dev_accs.append(float(dev_acc))
            #test_accs.append(float(test_acc))
            epochs.append(epoch)
            iters.append(iter_num)
            steps.append(global_step)

    print("dev_accs:{}".format(dev_accs))
    #print("test_accs:{}".format(test_accs))

    max_index = np.argmax(dev_accs)
    max_dev_acc = dev_accs[max_index]
    max_iter = iters[max_index]
    max_step = steps[max_index]
    #test_acc = test_accs[max_index]

    strategy = strategys[max_index]
    epoch = epochs[max_index]

    index_name = index if index != "few_all" else "all"

    if task_name not in ["eprstmt", "csldcp", "bustm"]:
        best_predict_file = "index" + index + "_" + epoch  + "epoch_" + max_iter + "iter_" + max_step + "step_" + task_name + "f_predict.json"
        std_name = task_name + "f_predict_" + index_name + ".json"
    else:
        best_predict_file = "index" + index + "_" + epoch  + "epoch_" + max_iter + "iter_" + max_step + "step_" + task_name + "_predict.json"
        std_name = task_name + "_predict_" + index_name + ".json"

    predict_file = os.path.join(output_dir, strategy, task_name, best_predict_file)
    print("best_result_file:{}".format(predict_file))

    submit_file = os.path.join(submit_result_dir, std_name)
    print("std_result_file:{}".format(submit_file))

    print("{}\t{}\t{}\t{}\t{}\t{}".format(strategy, epoch, max_dev_acc, dev_num, max_iter, max_step))

    copyfile(predict_file, submit_file)
