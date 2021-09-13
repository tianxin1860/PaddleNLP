
task_name=$1

indexs=(0 1 2 3 4 few_all)
for index in ${indexs[@]}; do
    result="${HOME}/ppnlp_data/fewclue_paper/p-tuning_ernie1p0/output/index${index}_${task_name}_result_final"
    cat ${result} | tail -n1 | awk -F"\t" 'BEGIN{OFS="\t"}{print $3,$5}'
done
