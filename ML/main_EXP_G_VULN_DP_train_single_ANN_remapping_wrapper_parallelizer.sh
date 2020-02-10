#!/bin/bash

echo "HELLO MARCO"

for i in {0..4}
do

    python main_EXP_G_VULN_DP_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [100,100,100] -e 700 -bs 200  --id_gpu $(($i%4)) --perc_gpu 0.3 -nu 0.02 -tr_size 10000 -val_size 1000 -ts_size 50000 -tr_iter $i -mn model_200_ep700 &

done

wait

for i in {0..4}
do

    python main_EXP_G_VULN_DP_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [100,100,100] -e 700 -bs 200  --id_gpu $(($i%4)) --perc_gpu 0.3 -nu 0.02 -tr_size 30000 -val_size 3000 -ts_size 50000 -tr_iter $i -mn model_200_ep700 &

done

wait

for i in {0..4}
do

    python main_EXP_G_VULN_DP_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [100,100,100] -e 700 -bs 200 --id_gpu $(($i%4)) --perc_gpu 0.3 -nu 0.02 -tr_size 50000 -val_size 5000 -ts_size 50000 -tr_iter $i -mn model_200_ep700 &

done

wait

echo "All done"
