#bsub -W 24:00 -n 4 -R "rusage[mem=8000, scratch=2000]" "bash subrun.sh -n no -s 1 -f no -y 1980 -b yes"
#bsub -W 24:00 -n 4 -R "rusage[mem=8000, scratch=2000]" "bash subrun.sh -n no -s 1 -f no -y 1981 -b yes"
#bsub -W 24:00 -n 4 -R "rusage[mem=8000, scratch=2000]" "bash subrun.sh -n no -s 1 -f no -y 1984 -b yes"
#bsub -W 24:00 -n 4 -R "rusage[mem=8000, scratch=2000]" "bash subrun.sh -n no -s 1 -f no -y 1989 -b yes"
for i in {1..5}
do 
	bsub -W 72:00 -n 4 -R "rusage[mem=8000, scratch=2000]" "bash subrun.sh -n yes -s $i -f no -y 1980 -b no" 
	bsub -W 72:00 -n 4 -R "rusage[mem=8000, scratch=2000, ngpus_excl_p=1]" "bash subrun.sh -n no -s $i -f no -y 1980 -b no"
	bsub -W 72:00 -n 4 -R "rusage[mem=8000, scratch=2000]" "bash subrun.sh -n yes -s $i -f no -y 1981 -b no"
	bsub -W 72:00 -n 4 -R "rusage[mem=8000, scratch=2000, ngpus_excl_p=1]" "bash subrun.sh -n no -s $i -f no -y 1981 -b no"
	bsub -W 72:00 -n 4 -R "rusage[mem=8000, scratch=2000]" "bash subrun.sh -n yes -s $i -f no -y 1984 -b no"
	bsub -W 72:00 -n 4 -R "rusage[mem=8000, scratch=2000, ngpus_excl_p=1]" "bash subrun.sh -n no -s $i -f no -y 1984 -b no"
	bsub -W 72:00 -n 4 -R "rusage[mem=8000, scratch=2000]" "bash subrun.sh -n yes -s $i -f no -y 1989 -b no"
	bsub -W 72:00 -n 4 -R "rusage[mem=8000, scratch=2000, ngpus_excl_p=1]" "bash subrun.sh -n no -s $i -f no -y 1989 -b no"

	bsub -W 72:00 -n 4 -R "rusage[mem=8000, scratch=2000]" "bash subrun.sh -n yes -s $i -f yes -y 1980 -b no" 
	bsub -W 72:00 -n 4 -R "rusage[mem=8000, scratch=2000, ngpus_excl_p=1]" "bash subrun.sh -n no -s $i -f yes -y 1980 -b no"
	bsub -W 72:00 -n 4 -R "rusage[mem=8000, scratch=2000]" "bash subrun.sh -n yes -s $i -f yes -y 1981 -b no"
	bsub -W 72:00 -n 4 -R "rusage[mem=8000, scratch=2000, ngpus_excl_p=1]" "bash subrun.sh -n no -s $i -f yes -y 1981 -b no"
	bsub -W 72:00 -n 4 -R "rusage[mem=8000, scratch=2000]" "bash subrun.sh -n yes -s $i -f yes -y 1984 -b no"
	bsub -W 72:00 -n 4 -R "rusage[mem=8000, scratch=2000, ngpus_excl_p=1]" "bash subrun.sh -n no -s $i -f yes -y 1984 -b no"
	bsub -W 72:00 -n 4 -R "rusage[mem=8000, scratch=2000]" "bash subrun.sh -n yes -s $i -f yes -y 1989 -b no"
	bsub -W 72:00 -n 4 -R "rusage[mem=8000, scratch=2000, ngpus_excl_p=1]" "bash subrun.sh -n no -s $i -f yes -y 1989 -b no"	
done