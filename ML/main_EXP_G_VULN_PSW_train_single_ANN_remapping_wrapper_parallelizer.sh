#!/bin/bash

echo "HELLO MARCO!!!"

for i in {0..2}
do

    python main_EXP_G_VULN_PSW_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [100,100,100] -e 1000 -bs 1000 --id_gpu $(($i%2)) --perc_gpu 0.2 -nu 0.02 -tr_size 10000 -val_size 1000 -ts_size 10000 -tr_iter $i -mn model_3 &

done

wait

for i in {0..2}
do

    python main_EXP_G_VULN_PSW_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [100,100,100] -e 1000 -bs 1000 --id_gpu $(($i%2)) --perc_gpu 0.2 -nu 0.02 -tr_size 30000 -val_size 3000 -ts_size 30000 -tr_iter $i -mn model_3 &

done

wait

for i in {0..2}
do

    python main_EXP_G_VULN_PSW_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [100,100,100] -e 1000 -bs 1000 --id_gpu $(($i%2)) --perc_gpu 0.2 -nu 0.02 -tr_size 50000 -val_size 5000 -ts_size 50000 -tr_iter $i -mn model_3 &

done

wait

echo "All done"
