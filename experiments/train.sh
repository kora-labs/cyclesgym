for i in {1..3}
do 
	bsub -W 24:00 -n 4 -R "rusage[mem=8000]" "python3 train.py -ey 1980 -na --seed $i"
	bsub -W 24:00 -n 4 -R "rusage[mem=8000, ngpus_excl_p=1]" "python3 train.py -ey 1980 --seed $i"
	bsub -W 24:00 -n 4 -R "rusage[mem=8000]" "python3 train.py -ey 1981 -na --seed $i"
	bsub -W 24:00 -n 4 -R "rusage[mem=8000, ngpus_excl_p=1]" "python3 train.py -ey 1981 --seed $i"
	bsub -W 24:00 -n 4 -R "rusage[mem=8000]" "python3 train.py -ey 1984 -na --seed $i"
	bsub -W 24:00 -n 4 -R "rusage[mem=8000, ngpus_excl_p=1]" "python3 train.py -ey 1984 --seed $i"

	bsub -W 24:00 -n 4 -R "rusage[mem=8000]" "python3 train.py -ey 1980 -na -fw --seed $i"
	bsub -W 24:00 -n 4 -R "rusage[mem=8000, ngpus_excl_p=1]" "python3 train.py -ey 1980 -fw --seed $i"
	bsub -W 24:00 -n 4 -R "rusage[mem=8000]" "python3 train.py -ey 1981 -na -fw --seed $i"
	bsub -W 24:00 -n 4 -R "rusage[mem=8000, ngpus_excl_p=1]" "python3 train.py -ey 1981 -fw --seed $i"
	bsub -W 24:00 -n 4 -R "rusage[mem=8000]" "python3 train.py -ey 1980 -na -fw --seed $i"
	bsub -W 24:00 -n 4 -R "rusage[mem=8000, ngpus_excl_p=1]" "python3 train.py -ey 1984 -fw --seed $i"
done