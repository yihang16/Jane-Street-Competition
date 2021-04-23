export CUDA_VISIBLE_DEVICES=7
for i in {1..3}
do
	for j in {1..3} 
	do
		for k in {1..3}
		do
			python toby.py --weights $i $j $k 1 1
		done
	done
done
