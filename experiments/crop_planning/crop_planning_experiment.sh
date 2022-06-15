for i in {1..5}
do
  python train.py --fixed_weather True --non_adaptive True --seed $i
  python train.py --fixed_weather False --non_adaptive True --seed $i
  python train.py --fixed_weather True --non_adaptive False --seed $i
  python train.py --fixed_weather True --non_adaptive False --seed $i
done
