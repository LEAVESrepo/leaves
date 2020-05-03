#!/bin/bash

echo "HELLO MARCO"

python main_EXP_G_VULN_PSW_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 300 -hnc [100,100,100] -e 700 -bs 2000 --id_gpu 0 --perc_gpu 0.2 -nu 0.02 -tr_size 50000 -val_size 5000 -ts_size 50000 -tr_iter 0 -mn toy_model_1000 && python main_EXP_G_VULN_PSW_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn toy_model_1000 --tr_size [50000] --ts_size [50000] --val_size [5000] --bg_test_it 0 --end_test_it 1 &

wait

echo "All done"