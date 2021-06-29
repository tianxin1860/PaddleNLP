#!/bin/bash

#task_names=(iflytek tnews eprstmt bustm ocnli csl)
#all_gpu=(0 1 2 3 4 5)

#task_name="cluewsc"
#gpus=0

#task_name="tnews"
#gpus=1

#task_name="eprstmt"
#gpus=2

#task_name="chid"
#gpus=3

#task_name="bustm"
#gpus=4

#task_name="ocnli"
#gpus=5


#task_name="csldcp"
#gpus=7

#task_name="iflytek"
#gpus=5

#task_name="csl"
#gpus=7

#export pretrained_model="ernie1p0"
#export pretrained_model="macbert-base-chinese"
export name="ptpet"
export pretrained_model="macbert-large-chinese"
task_name=$1
gpus=$2

indexs=(0 1 2 3 4 few_all)
#indexs=(0)
#indexs=(3)
#indexs=(4)
#indexs=(few_all)
#indexs=(0 few_all)

for index in ${indexs[@]}; do
	if [[ ${pretrained_model} == "ernie1p0" ]]; then
		bash commands/train.sh ${task_name} ${gpus} ${index} > gpu${gpus}_train_${pretrained_model}_${task_name}_index${index}.log 2>&1
	else
		bash commands/get_single_result.sh ${task_name} ${gpus} ${index} > get_result_${pretrained_model}_${task_name}_index${index}.log 2>&1
	fi
done
