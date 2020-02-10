#!/bin/bash

echo "HELLO MARCO"

python main_BIS_EXP_G_VULN_MULTIPLE_GUESSES_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [100,100,100] -e 1000 -bs 1000 --id_gpu 0 --perc_gpu 0.3 -nu 0.02 -tr_size 10000 -val_size 1000 -ts_size 50000 -tr_iter 0 -mn model_1000 && python main_BIS_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn model_1000 --tr_size [10000] --ts_size [50000] --val_size [1000] --bg_test_it 0 --end_test_it 25 &

wait

echo "All done"