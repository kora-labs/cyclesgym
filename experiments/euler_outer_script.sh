for i in {1..5}
do
  bsub -W 72:00 -n 8 -R "rusage[mem=8000, scratch=2000]" "bash experiments/euler_script.sh -f True -n True -s $i"
done
