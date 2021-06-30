#!/bin/bash

set -xu

PYTHON_BIN="/usr/local/bin/python3.7"
export PYTHONPATH="../../../"
export FLAGS_fast_eager_deletion_mode=1

PPNLP_DATA_PATH="."
local_log_path="${PPNLP_DATA_PATH}/examples/few_shot/${name}_${pretrained_model}/"
submit_dir="${PPNLP_DATA_PATH}/examples/few_shot/${name}_${pretrained_model}/submit"

task_name=$1
gpus=$2
index=$3
max_seq_len=512

batch_size=(4 8 16 32)
learning_rate=(2E-5 5E-5 1E-4 5E-4)
p_embedding_num=(1 4 8 16 32)
confidences=(1.0 0.9 0.8)
eval_steps=50
max_not_better_num=40
predict_batch_size=8

epoch=10

if [[ ${task_name} == "ocnli" ]]; then
    pattern_ids=(0)
    min_pred_prob=0.90
fi

if [[ ${task_name} == "eprstmt" ]]; then
    pattern_ids=(0 1)
    min_pred_prob=0.92
fi

if [[ ${task_name} == "tnews" ]]; then
    pattern_ids=(0 1 2)
    min_pred_prob=0.96
fi

if [[ ${task_name} == "bustm" ]]; then
    pattern_ids=(0 1)
    min_pred_prob=0.92
    eval_steps=10
fi

if [[ ${task_name} == "iflytek" ]]; then
    pattern_ids=(0)
    min_pred_prob=0.92
    eval_steps=40
fi

if [[ ${task_name} == "csldcp" ]]; then
    pattern_ids=(0)
    min_pred_prob=0.92
    eval_steps=40
    max_seq_len=500
fi

if [[ ${task_name} == "csl" ]]; then
    pattern_ids=(0 1)
    min_pred_prob=0.92
    eval_steps=40
    max_seq_len=500
fi

if [[ ${task_name} == "chid" ]]; then
    pattern_ids=(0)
    min_pred_prob=0.92
    eval_steps=40
    max_seq_len=500
fi

if [[ ${task_name} == "cluewsc" ]]; then
    pattern_ids=(0)
    min_pred_prob=0.92
    eval_steps=20
fi

# for debug
pattern_ids=(0)
batch_size=(4)
learning_rate=(2E-4)
p_embedding_num=(3)
confidences=(1.0)
epoch=2

function train() {
	local task_name=$1
	local lr=$2
	local bs=$3
	local p_num=$4
    local pt_id=$5
	local confidence=${confidence}

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
	for confidence in ${confidences[@]}; do
	for lr in ${learning_rate[@]}; do
	for bs in ${batch_size[@]}; do
	for p_num in ${p_embedding_num[@]}; do
	for pt_id in ${pattern_ids[@]}; do
		echo "[strat training] task:${task_name} lr:${lr} bs:${bs} p_num:${p_num} pattern_id:${pt_id} confidence:${confidence}"
		train ${task_name} ${lr} ${bs} ${p_num} ${pt_id} ${confidence}
	done
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
