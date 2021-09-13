#!/bin/bash

set -ux

source "${HOME}/share/env.sh"
PYTHON_BIN="/usr/local/bin/python3.7"
export PYTHONPATH="/home/tianxin04/PaddleNLP"

local_log_path="${PPNLP_DATA_PATH}/fewclue_paper/p-tuning_${pretrained_model}/"
#submit_dir="${PPNLP_DATA_PATH}/fewclue_paper/p-tuning_${pretrained_model}/submit"
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


batch_size=(16)
p_embedding_num=(9)
learning_rate=(3E-5)
epoch=20


# for debug
#batch_size=(4)
#learning_rate=(1E-4)
#p_embedding_num=(3)
#epoch=10

function train() {
	local task_name=$1
	local lr=$2
	local bs=$3
	local p_num=$4

	strategy="bs${bs}_lr${lr}_pnum${p_num}"

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
		--device gpu \
		--p_embedding_num ${p_num} \
		--save_dir ${save_checkpoint_dir} \
		--index ${index} \
		--output_dir ${output_dir} \
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
		echo "[strat training] ${task_name} ${lr} ${bs} ${p_num}"
		train ${task_name} ${lr} ${bs} ${p_num}
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

		strategy="bs${bs}_lr${lr}_pnum${p_num}"
		
		output_dir="${local_log_path}/output/${strategy}/${task_name}"

		log_dir="${local_log_path}/log/${strategy}/${task_name}"
		log_file="${log_dir}/index${index}_log"

		grep "dev_accuracy" ${log_file} > ${output_dir}/index${index}_dev_acc
		grep "test_accuracy" ${log_file} > ${output_dir}/index${index}_test_acc
		paste ${output_dir}/index${index}_dev_acc ${output_dir}/index${index}_test_acc | ${PYTHON_BIN} get_max.py ${strategy} 1>> "${local_log_path}/output/index${index}_${task_name}_result" 2>> "${local_log_path}/output/index${index}_${task_name}_result_all"

	done
	done
	done

	${PYTHON_BIN} get_final.py ${local_log_path}/output/index${index}_${task_name}_result \
	"${local_log_path}/output" \
	${task_name} \
	${index} \
	> ${local_log_path}/output/index${index}_${task_name}_result_final
}

#train_wrapper
get_max_result
