#!/bin/bash

echo "HELLO MARCO"

for i in {0..9}
do

    python main_EMPIRICAL_ESTIMATES_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [100,100,100] -e 1000 -bs all --id_gpu $(($i%4)) --perc_gpu 0.3 -nu 0.02 -tr_size 5000 -val_size 500 -ts_size 5000 -tr_iter $i -mn model_all &

done

wait

for i in {0..9}
do

    python main_EMPIRICAL_ESTIMATES_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [100,100,100] -e 1000 -bs all --id_gpu $(($i%4)) --perc_gpu 0.3 -nu 0.02 -tr_size 10000 -val_size 1000 -ts_size 5000 -tr_iter $i -mn model_all &

done

wait

for i in {0..9}
do

    python main_EMPIRICAL_ESTIMATES_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [100,100,100] -e 1000 -bs all --id_gpu $(($i%4)) --perc_gpu 0.3 -nu 0.02 -tr_size 20000 -val_size 2000 -ts_size 5000 -tr_iter $i -mn model_all &

done

wait

echo "All done"
