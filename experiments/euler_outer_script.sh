for i in {3..5}
do
	bsub -W 72:00 -n 8 -R "rusage[mem=8000, scratch=2000]" "bash euler_script.sh -f True -s $i"
	bsub -W 72:00 -n 8 -R "rusage[mem=8000, scratch=2000]" "bash euler_script.sh -f False -s $i"
done
