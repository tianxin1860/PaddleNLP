#!/bin/bash
set -xu

#export pretrained_model="ernie1p0"
#export pretrained_model="macbert-base-chinese"

export name="ptpet_self_train"
export pretrained_model="macbert-large-chinese"
task_name=$1
gpus=$2

indexs=(0 1 2 3 4 few_all)

for index in ${indexs[@]}; do
	if [[ ${pretrained_model} == "ernie1p0" ]]; then
		bash commands/train.sh ${task_name} ${gpus} ${index} > ${name}_gpu${gpus}_train_${pretrained_model}_${task_name}_index${index}.log 2>&1
	else
		bash commands/train_macbert_ptpet.sh ${task_name} ${gpus} ${index} > ${name}_gpu${gpus}_train_${pretrained_model}_${task_name}_index${index}.log 2>&1
	fi
done
