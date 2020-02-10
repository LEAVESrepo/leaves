#!/bin/bash

echo "HELLO MARCO BID GEO"

python main_BIS_EXP_G_VULN_GEO_LOCATION_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [500,500,500] -e 3000 -bs 300 --id_gpu 0 --perc_gpu 0.2 -nu 0.02 -tr_size 1000 -val_size 100 -ts_size 50000 -tr_iter 2 -mn model_200 python && python main_BIS_EXP_G_VULN_GEO_LOCATION_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn model_200 --tr_size [1000] --ts_size [50000] --val_size [100] --bg_test_it 0 --end_test_it 25 &

wait

echo "All done"