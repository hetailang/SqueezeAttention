## Step 3: Evaluate the performance of generated text (refer helm/command/eval.sh)
cd helm
sam_num=100
model=llama2-13B
ini_size=0.2
percent=50
TASK=xsum
JSONL=../pred_helm/${model}/${ini_size}/${percent}/xsum.jsonl
OUTPUT=xsum_${model}_${ini_size}_${percent}_result
ARCH=llama
pre_path=/home/user/wangzihao/kv_pruner/helm/benchmark_output/runs
mkdir ${pre_path}/${OUTPUT}
mkdir ${pre_path}/${OUTPUT}/eval_cache
cp benchmark_output/runs/eval_cache/* ${pre_path}/${OUTPUT}/eval_cache

python scripts/offline_eval/import_results.py together ${JSONL} --cache-dir prod_env/cache
helm-run --conf src/helm/benchmark/presentation/${TASK}/run_specs_${ARCH}.conf --local --max-eval-instances ${sam_num} --num-train-trials=1 --suite ${OUTPUT} -n 1
helm-summarize --suite ${OUTPUT}
cd ../
## The results are writted into a tex file that can be found in benchmark_output/runs/xsum_llama7b_result/groups/latex/
