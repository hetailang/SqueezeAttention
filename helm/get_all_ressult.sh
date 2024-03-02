for i in `seq 25 32`
do

input_path=./benchmark_output/runs/xsum_llama7B_${i}_result/groups/latex/core_scenarios_accuracy.tex
output_path=./result_llama7B
model_name=GPT-NeoX

python get_result.py \
	--input_path ${input_path} \
	--output_path ${output_path} \
	--model_name ${model_name}

echo done with ${input_path}
done
