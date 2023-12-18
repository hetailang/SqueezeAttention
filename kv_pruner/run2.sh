#!/bin/bash
#modes=("sp1" "sp1_" "sp2" "sp2_" "total")
#for mode in "${modes[@]}"; do
#	python pred_new.py --model llama2-13B --pred helm --mode ${mode} --model_arch llama
#done

for i in $(seq 0.1 0.05 1.0)
do
	python pred.py --model Mistral --pred Long --hiddlayer --model Mistral --ini_size 2500 --percent ${i} --device cuda:1 --exp 1
done
