#!/bin/bash
while getopts n:s:f:y:b: flag
do
    case "${flag}" in
        n) na=${OPTARG};;
        s) seed=${OPTARG};;
        f) fw=${OPTARG};;
		y) ey=${OPTARG};;
		b) baseline=${OPTARG};;
    esac
done

rsync -aq -r ../cycles.zip ${TMPDIR}
cd $TMPDIR
unzip cycles.zip
cd $LS_SUBCWD
if [[ "$baseline" == "yes" ]]
then 
	python3 train.py -ey $ey --seed ${seed} --baseline
else
	if [[ "$fw" == "yes" ]]
	then
		if [[ "$na" == "yes" ]]
		then
			python3 train.py -ey ${ey} -na -fw --seed ${seed}
		else
			python3 train.py -ey ${ey} -fw --seed ${seed}
		fi
	else
		if [[ "$na" == "yes" ]]
		then
			python3 train.py -ey ${ey} -na --seed ${seed}
		else
			python3 train.py -ey ${ey} --seed ${seed}
		fi
	fi
fi