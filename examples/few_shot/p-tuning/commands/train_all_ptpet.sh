#!/bin/bash
set -xu

###################################################################################
# 复现说明: 以复现 eprstmt 任务为例, 执行如下命令:
# Bash ./commands/train_all_ptpet.sh eprstmt 7
# 启动之后，会在预置的超参数组合上跑多组实验，所有实验结束之后
# 在所有超参数组合的验证集上挑选效果最好的模型对应的预测结果进行提交
# Note: ocnli 任务用了额外的预训练数据, 需要用预训练之后的模型进行热启动
###################################################################################

if [[ $! != 2 ]]; then
	echo "Bash $0 task_name gpu_id"
	exit 1
fi

export name="ptpet_self_train"
export pretrained_model="macbert-large-chinese"
task_name=$1
gpus=$2

indexs=(0 1 2 3 4 few_all)

for index in ${indexs[@]}; do
	bash commands/train_macbert_ptpet.sh ${task_name} ${gpus} ${index} > ${name}_gpu${gpus}_train_${pretrained_model}_${task_name}_index${index}.log 2>&1
done
