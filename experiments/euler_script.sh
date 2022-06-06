#!/bin/bash
while getopts f:s:n: flag
do
    case "${flag}" in
        f) fw=${OPTARG};;
        s) seed=${OPTARG};;
        n) na=${OPTARG};;
    esac
done

rsync -aq -r $PWD/cycles.zip ${TMPDIR}
cd $TMPDIR
unzip cycles.zip
cd $LS_SUBCWD

python experiments/train.py -fw ${fw} -na ${na} --seed ${seed}