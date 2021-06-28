#!/bin/bash

set -xu

source "${HOME}/share/env.sh"

PYTHON_BIN="/usr/local/bin/python3.7"
export PYTHONPATH="/home/tianxin04/develop/PaddleNLP"
export FLAGS_fast_eager_deletion_mode=1

task_data_dir="/home/tianxin04/develop/FewCLUE/datasets/"
local_log_path="${PPNLP_DATA_PATH}/examples/few_shot/${name}_${pretrained_model}/"
submit_dir="${PPNLP_DATA_PATH}/examples/few_shot/${name}_${pretrained_model}/submit"
#predict_output_dir="${local_log_path}/predict_output/"

#task_name="iflytek"
#task_name="tnews"
#task_name="eprstmt"
#task_name="bustm"
#task_name="ocnli"
#task_name="csl"

#task_name="csldcp"
#gpus=6

#task_name="cluewsc"
#gpus=7

task_name=$1
gpus=$2
index=$3

max_seq_len=512

learning_rate=(5E-5 1E-4 5E-4)
p_embedding_num=(1 8 16)

max_not_better_num=20
eval_steps=50
confidence=1.0
min_pred_prob=0.92
epoch=10


if [[ ${task_name} == "iflytek" || ${task_name} == "csl" || ${task_name} == "csldcp" ]]; then
	batch_size=(8 16)
else
	batch_size=(8 16 32)
fi

if [[ ${task_name} == "csl" || ${task_name} == "csldcp" || ${task_name} == "iflytek" ]]; then
    max_seq_len=500
else
    max_seq_len=512
fi

if [[ ${task_name} == "tnews" || ${task_name} == "eprstmt" || ${} == "iflytek" ]]; then
    learning_rate=(1E-5 2.5E-5 5E-5)
fi

if [[ ${task_name} == "csldcp" ]]; then
    learning_rate=(2E-5 5E-5)
    p_embedding_num=(1 8)
fi

if [[ ${task_name} == "iflytek" ]]; then
    batch_size=(8 16)
    p_embedding_num=(1 4)
    learning_rate=(2E-5 5E-5)
fi

if [[ ${task_name} == "ocnli" ]]; then
    batch_size=(8 16 32)
    p_embedding_num=(1 8)
    learning_rate=(2E-5 5E-5)
fi

if [[ ${task_name} == "eprstmt" ]]; then
    pattern_ids=(1)
    batch_size=(8 16)
    learning_rate=(2E-5 5E-5)
    p_embedding_num=(8 16)
    predict_batch_size=8
    min_pred_prob=0.92
fi

if [[ ${task_name} == "tnews" ]]; then
    pattern_ids=(1 0)
    batch_size=(8 16)
    learning_rate=(2E-5 5E-5)
    p_embedding_num=(1 8 16)
    predict_batch_size=8
    min_pred_prob=0.96
fi

if [[ ${task_name} == "bustm" ]]; then
    pattern_ids=(1 0)
    batch_size=(8 16)
    learning_rate=(2E-5 5E-5)
    p_embedding_num=(1 8 16)
    predict_batch_size=8
    min_pred_prob=0.90
    eval_steps=10
fi

if [[ ${task_name} == "iflytek" ]]; then
    pattern_ids=(0)
    batch_size=(8 16)
    learning_rate=(2E-5 5E-5)
    p_embedding_num=(1)
    predict_batch_size=8
    min_pred_prob=0.92
    eval_steps=40
fi

# for debug
#batch_size=(4)
#learning_rate=(1E-4)
#p_embedding_num=(3)
#p_embedding_num=(32)

#epoch=10

function train() {
	local task_name=$1
	local lr=$2
	local bs=$3
	local p_num=$4
    local pt_id=$5

	strategy="bs${bs}_lr${lr}_pnum${p_num}_maxlen${max_seq_len}_ptid${pt_id}"

	save_checkpoint_dir="${local_log_path}/checkpoints/${strategy}/${task_name}"
	output_dir="${local_log_path}/output/${strategy}/${task_name}"
	log_dir="${local_log_path}/log/${strategy}/${task_name}"
	log_file="${log_dir}/index${index}_log"

	mkdir -p ${save_checkpoint_dir}
	mkdir -p ${log_dir}
	mkdir -p ${output_dir}
	
	train_script="ptuning.py"

	cmd="${PYTHON_BIN} -u -m paddle.distributed.launch --gpus "${gpus}" --log_dir launch_log/${strategy}/${task_name} \
		${train_script} \
		--task_name ${task_name} \
		--language_model ${pretrained_model} \
		--device gpu \
        --predict_batch_size ${predict_batch_size} \
        --pattern_id ${pt_id} \
		--p_embedding_num ${p_num} \
        --max_not_better_num ${max_not_better_num} \
        --eval_steps ${eval_steps} \
		--save_dir ${save_checkpoint_dir} \
		--index ${index} \
		--output_dir ${output_dir} \
        --confidence ${confidence} \
        --min_prob ${min_pred_prob} \
		--batch_size ${bs} \
		--learning_rate ${lr} \
		--epochs ${epoch} \
		--max_seq_length ${max_seq_len} \
		> ${log_file} 2>/dev/null"

	echo $cmd
	eval $cmd
}


function train_wrapper() {
	for lr in ${learning_rate[@]}; do
	for bs in ${batch_size[@]}; do
	for p_num in ${p_embedding_num[@]}; do
	for pt_id in ${pattern_ids[@]}; do
		echo "[strat training] ${task_name} ${lr} ${bs} p_num:${p_num} pattern_id:${pt_id}"
		train ${task_name} ${lr} ${bs} ${p_num} ${pt_id}
	done
	done
	done
	done
}

function get_max_result() {

	:> "${local_log_path}/output/index${index}_${task_name}_result"
	:> "${local_log_path}/output/index${index}_${task_name}_result_all"


	for lr in ${learning_rate[@]}; do
	for bs in ${batch_size[@]}; do
	for p_num in ${p_embedding_num[@]}; do
	for pt_id in ${pattern_ids[@]}; do

		strategy="bs${bs}_lr${lr}_pnum${p_num}_maxlen${max_seq_len}_ptid${pt_id}"
		
		output_dir="${local_log_path}/output/${strategy}/${task_name}"

		log_dir="${local_log_path}/log/${strategy}/${task_name}"
		log_file="${log_dir}/index${index}_log"

		grep "dev_accuracy" ${log_file} > ${output_dir}/index${index}_dev_acc
		# only used dev_set to select model
		# grep "test_accuracy" ${log_file} > ${output_dir}/test_acc
		cat ${output_dir}/index${index}_dev_acc | ${PYTHON_BIN} get_max.py ${strategy} 1>> "${local_log_path}/output/index${index}_${task_name}_result" 2>> "${local_log_path}/output/index${index}_${task_name}_result_all"
		#paste ${output_dir}/dev_acc ${output_dir}/test_acc | ${PYTHON_BIN} get_max.py ${strategy} 1>> "${local_log_path}/output/index${index}_${task_name}_result" 2>> "${local_log_path}/output/index${index}_${task_name}_result_all"

	done
	done
	done
	done

	${PYTHON_BIN} get_final.py ${local_log_path}/output/index${index}_${task_name}_result \
	"${local_log_path}/output" \
	${task_name} \
	${index} \
	${submit_dir} \
	> ${local_log_path}/output/index${index}_${task_name}_result_final
}

train_wrapper
get_max_result
