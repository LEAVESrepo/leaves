#!/bin/bash

echo "HELLO MARCO"

python main_BIS_EXP_G_VULN_DP_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [100,100,100] -e 1 -bs 200 --id_gpu 0 --perc_gpu 0.2 -nu 0.02 -tr_size 50000 -val_size 5000 -ts_size 50000 -tr_iter 0 -mn toy_model_500 && python main_BIS_EXP_G_VULN_DP_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn toy_model_500 --tr_size [50000] --ts_size [50000] --val_size [5000] --bg_test_it 0 --end_test_it 10 &

wait

echo "All done"