for ey in {1980, 1981, 1984}
do
	python3 train.py -ey $ey --baseline
	for seed in {1..5}
	do
		python3 train.py -ey ${ey} -na -fw --seed ${seed}
		python3 train.py -ey ${ey} -fw --seed ${seed}
		python3 train.py -ey ${ey} -na --seed ${seed}
		python3 train.py -ey ${ey} --seed ${seed}
	done
done