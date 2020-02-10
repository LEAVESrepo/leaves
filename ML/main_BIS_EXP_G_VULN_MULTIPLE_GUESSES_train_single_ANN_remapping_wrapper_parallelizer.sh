#!/bin/bash

echo "HELLO MARCO"

for i in {0..4}
do

    python main_BIS_EXP_G_VULN_MULTIPLE_GUESSES_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [100,100,100] -e 500 -bs 1000 --id_gpu $(($i%4)) --perc_gpu 0.3 -nu 0.02 -tr_size 10000 -val_size 1000 -ts_size 50000 -tr_iter $i -mn model_1000_ep500 &

done

wait

for i in {0..4}
do

    python main_BIS_EXP_G_VULN_MULTIPLE_GUESSES_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [100,100,100] -e 500 -bs 1000 --id_gpu $(($i%4)) --perc_gpu 0.3 -nu 0.02 -tr_size 30000 -val_size 3000 -ts_size 50000 -tr_iter $i -mn model_1000_ep500 &

done

wait

for i in {0..4}
do

    python main_BIS_EXP_G_VULN_MULTIPLE_GUESSES_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [100,100,100] -e 500 -bs 1000 --id_gpu $(($i%4)) --perc_gpu 0.3 -nu 0.02 -tr_size 50000 -val_size 5000 -ts_size 50000 -tr_iter $i -mn model_1000_ep500 &

done

wait

echo "All done"
