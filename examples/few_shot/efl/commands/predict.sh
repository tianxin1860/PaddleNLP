#!/bin/bash

source "${HOME}/share/env.sh"

PYTHON_BIN="/usr/local/bin/python3.7"
export PYTHONPATH="/home/tianxin04/develop/PaddleNLP:${PYTHONPATH}"

task_data_dir="/home/tianxin04/develop/FewCLUE/datasets/"
local_log_path="${PPNLP_DATA_PATH}/examples/few_shot/p-tuning/"

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

max_seq_len=512


if [[ ${task_name} == "iflytek" || ${task_name} == "csl" || ${task_name} == "csldcp" ]]; then
	batch_size=(8 16)
else
	batch_size=(8 16 32)
fi

# iflytek
p_embedding_num=(1)
learning_rate=(1E-4)
batch_size=(8)

# bustm
# bs8_lr1E-4_pnum1    5   65.625  62.472  32  1772
#batch_size=(8)
#learning_rate=(1E-4)
#p_embedding_num=(1)

# csldcp
# bs16_lr5E-5_pnum1   9   58.209  60.594  536 1784
#batch_size=(16)
#learning_rate=(5E-5)
#p_embedding_num=(1)

# csldcp
# bs16_lr5E-5_pnum1   9   58.209  60.594  536 1784
#batch_size=(32)
#learning_rate=(5E-5)
#p_embedding_num=(1)

# eprstme
# bs32_lr5E-4_pnum4   9   78.125  77.541  32  610
#batch_size=(32)
#learning_rate=(5E-4)
#p_embedding_num=(4)

# bs8_lr5E-4_pnum1    4   50.0    35.437  32  2520
# batch_size=(8)
# learning_rate=(5E-4)
# p_embedding_num=(1)

# chid
# bs32_lr5E-5_pnum8   10  28.571  40.909  42  2002
#batch_size=(32)
#learning_rate=(5E-5)
#p_embedding_num=(8)

# csl
#batch_size=(8)
#learning_rate=(5E-4)
#p_embedding_num=(1)

function train() {
	local task_name=$1
	local lr=$2
	local bs=$3
	local p_num=$4

	strategy="bs${bs}_lr${lr}_pnum${p_num}"

	log_dir="${local_log_path}/predict_log/${strategy}/${task_name}"
	#output_dir="${local_log_path}/predict_output/${strategy}/${task_name}"
	#output_dir="${local_log_path}/predict_output/${strategy}/"
	output_dir="${local_log_path}/predict_output/"
	save_checkpoint_dir="${local_log_path}/checkpoints/${strategy}/${task_name}"

	#params_file="${save_checkpoint_dir}/model_348/model_state.pdparams"
	#params_file="${save_checkpoint_dir}/model_16/model_state.pdparams"

	#p-tuning/checkpoints/bs16_lr5E-5_pnum1/csldcp/model_238/
	#params_file="${save_checkpoint_dir}/model_238/model_state.pdparams"

	# checkpoints/bs32_lr5E-5_pnum8/tnews/model_80/
	# params_file="${save_checkpoint_dir}/model_80/model_state.pdparams"

	# checkpoints/bs32_lr5E-5_pnum1/cluewsc/model_10/
	# params_file="${save_checkpoint_dir}/model_10/model_state.pdparams"

	#checkpoints/bs32_lr5E-4_pnum4/eprstmt/model_9/
	#params_file="${save_checkpoint_dir}/model_9/model_state.pdparams"

	# checkpoints/bs8_lr5E-4_pnum1/ocnli/model_40/
	# params_file="${save_checkpoint_dir}/model_40/model_state.pdparams"
	
	# checkpoints/bs32_lr5E-5_pnum8/chid/model_16/
	#params_file="${save_checkpoint_dir}/model_16/model_state.pdparams"

	# checkpoints/bs8_lr5E-4_pnum1/csl/model_40/
	# params_file="${save_checkpoint_dir}/model_40/model_state.pdparams"

	# checkpoints/bs8_lr1E-4_pnum1/iflytek/model_696/
	params_file="${save_checkpoint_dir}/model_696/model_state.pdparams"

	echo "params_file:${params_file}"
	mkdir -p ${log_dir}
	mkdir -p ${output_dir}
	

	cmd="${PYTHON_BIN} -u -m paddle.distributed.launch --gpus "${gpus}" --log_dir launch_log/${strategy}/${task_name} \
		predict.py \
		--task_name ${task_name} \
		--device gpu \
		--init_from_ckpt ${params_file} \
		--p_embedding_num ${p_num} \
		--output_dir ${output_dir} \
		--batch_size ${bs} \
		--max_seq_length ${max_seq_len} \
		> ${log_dir}/log 2>/dev/null"

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

	:> "${local_log_path}/output/${task_name}_result"
	:> "${local_log_path}/output/${task_name}_result_all"

	for lr in ${learning_rate[@]}; do
	for bs in ${batch_size[@]}; do
	for p_num in ${p_embedding_num[@]}; do

		strategy="bs${bs}_lr${lr}_pnum${p_num}"
		log_file="${local_log_path}/log/${strategy}/${task_name}/log"
		output_dir="${local_log_path}/output/${strategy}/${task_name}"

		grep "dev_accuracy" ${log_file} > ${output_dir}/dev_acc
		grep "test_accuracy" ${log_file} > ${output_dir}/test_acc
		paste ${output_dir}/dev_acc ${output_dir}/test_acc | ${PYTHON_BIN} get_max.py ${strategy} 1>> ${local_log_path}/output/${task_name}_result 2>> ${local_log_path}/output/${task_name}_result_all

	done
	done
	done

	${PYTHON_BIN} get_final.py ${local_log_path}/output/${task_name}_result > ${local_log_path}/output/${task_name}_result_final
}

train_wrapper
#get_max_result
