python pred.py --model llama2-7b-32k --pred xsum --enable_squeeze --model_arch llama --device cuda:0 --ini_size 0.4 --KV_class3 0.4 --sample_num 300 --start_size 4
sh script/helm.sh
python pred.py --model llama2-7b-32k --pred xsum --enable_squeeze --model_arch llama --device cuda:0 --ini_size 0.4 --KV_class3 0.25 --sample_num 300 --start_size 4
sh script/helm.sh