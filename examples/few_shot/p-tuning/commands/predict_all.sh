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

task_name="iflytek"
gpus=5

#task_name="csl"
#gpus=0

bash commands/predict.sh ${task_name} ${gpus} > predict_${task_name}.log 2>&1
