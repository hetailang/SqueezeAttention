for i in $(seq 0.1 0.05 1.0)
do
	python eval.py --model Mistral --pred Long --ini_size 2500 --percent ${i} --exp 1
done

#python eval.py --model Mistral --pred exp --mode spec --spec_size 8128
