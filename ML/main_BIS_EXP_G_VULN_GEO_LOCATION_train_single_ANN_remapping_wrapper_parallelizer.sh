#!/bin/bash

echo "HELLO MARCO"

#for i in {0..4}
#do

#    python main_BIS_EXP_G_VULN_GEO_LOCATION_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [500,500,500] -e 2000 -bs 100 --id_gpu $(($i%4)) --perc_gpu 0.3 -nu 0.02 -tr_size 100 -val_size 10 -ts_size 50000 -tr_iter $i -mn model_100_300_500 &

#done

#wait

for i in {0..4}
do

    python main_BIS_EXP_G_VULN_GEO_LOCATION_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [500,500,500] -e 3000 -bs 200 --id_gpu $(($i%4)) --perc_gpu 0.3 -nu 0.02 -tr_size 1000 -val_size 100 -ts_size 50000 -tr_iter $i -mn model_100_300_500 &

done

wait

#for i in {0..4}
#do
#    python main_BIS_EXP_G_VULN_GEO_LOCATION_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [500,500,500] -e 3000 -bs 500 --id_gpu $(($i%4)) --perc_gpu 0.3 -nu 0.02 -tr_size 10000 -val_size 1000 -ts_size 50000 -tr_iter $i -mn model_100_300_500 &

#done

#wait

echo "All done"
