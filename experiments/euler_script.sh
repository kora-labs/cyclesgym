#!/bin/bash
while getopts f:s: flag
do
    case "${flag}" in
        f) fw=${OPTARG};;
        s) seed=${OPTARG};;
    esac
done

rsync -aq -r $PWD/cycles.zip ${TMPDIR}
cd $TMPDIR
unzip cycles.zip
cd $LS_SUBCWD

echo $TMPDIR
echo $LS_SUBCWD
echo experiments/train.py -fw ${fw} --seed ${seed}

python experiments/train.py -fw ${fw} --seed ${seed}