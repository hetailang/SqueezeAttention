python pred.py --model Mistral --pred Long --enable_squeeze --model_arch Mistral --device cuda:0 --ini_size 0.21 --KV_class3 0.21

python eval.py --model Mistral --pred Long --ini_size 0.21 --KV_class3 0.21

python pred.py --model Mistral --pred Long --enable_squeeze --model_arch Mistral --device cuda:0 --ini_size 0.21 --KV_class3 0.08

python eval.py --model Mistral --pred Long --ini_size 0.21 --KV_class3 0.08
