#!/bin/bash

echo "HELLO MARCO"

python main_EXP_G_VULN_GEO_LOCATION_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [500,500,500] -e 700 -bs 200 --id_gpu 0 --perc_gpu 0.2 -nu 0.02 -tr_size 100 -val_size 10 -ts_size 50000 -tr_iter 2 -mn model_200 python && python main_EXP_G_VULN_GEO_LOCATION_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn model_200 --tr_size [100] --ts_size [50000] --val_size [10] --bg_test_it 0 --end_test_it 25 &
#python main_EXP_G_VULN_GEO_LOCATION_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [500,500,500] -e 1000 -bs 1000 --id_gpu 0 --perc_gpu 0.2 -nu 0.02 -tr_size 17665 -val_size 1763 -ts_size 17665 -tr_iter 2 -mn model_1000 && python main_EXP_G_VULN_GEO_LOCATION_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn model_1000 --tr_size [17665] --ts_size [17665] --val_size [1763] --bg_test_it 0 --end_test_it 25 &
#python main_EXP_G_VULN_GEO_LOCATION_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [500,500,500] -e 1000 -bs 4000 --id_gpu 0 --perc_gpu 0.2 -nu 0.02 -tr_size 17665 -val_size 1763 -ts_size 17665 -tr_iter 2 -mn model_4000 && python main_EXP_G_VULN_GEO_LOCATION_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn model_4000 --tr_size [17665] --ts_size [17665] --val_size [1763] --bg_test_it 0 --end_test_it 25 &
#python main_EXP_G_VULN_GEO_LOCATION_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [500,500,500] -e 1000 -bs 10000 --id_gpu 0 --perc_gpu 0.2 -nu 0.02 -tr_size 17665 -val_size 1763 -ts_size 17665 -tr_iter 2 -mn model_10000 && python main_EXP_G_VULN_GEO_LOCATION_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn model_10000 --tr_size [17665] --ts_size [17665] --val_size [1763] --bg_test_it 0 --end_test_it 25 &

#python main_EXP_G_VULN_GEO_LOCATION_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [500,500,500] -e 1000 -bs 512 --id_gpu 1 --perc_gpu 0.2 -nu 0.02 -tr_size 17665 -val_size 1763 -ts_size 17665 -tr_iter 2 -mn model_512_b && python main_EXP_G_VULN_GEO_LOCATION_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn model_512_b --tr_size [17665] --ts_size [17665] --val_size [1763] --bg_test_it 0 --end_test_it 25 &
#python main_EXP_G_VULN_GEO_LOCATION_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [500,500,500] -e 1000 -bs 1000 --id_gpu 1 --perc_gpu 0.2 -nu 0.02 -tr_size 17665 -val_size 1763 -ts_size 17665 -tr_iter 2 -mn model_1000_b && python main_EXP_G_VULN_GEO_LOCATION_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn model_1000_b --tr_size [17665] --ts_size [17665] --val_size [1763] --bg_test_it 0 --end_test_it 25 &
#python main_EXP_G_VULN_GEO_LOCATION_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [500,500,500] -e 1000 -bs 4000 --id_gpu 1 --perc_gpu 0.2 -nu 0.02 -tr_size 17665 -val_size 1763 -ts_size 17665 -tr_iter 2 -mn model_4000_b && python main_EXP_G_VULN_GEO_LOCATION_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn model_4000_b --tr_size [17665] --ts_size [17665] --val_size [1763] --bg_test_it 0 --end_test_it 25 &
#python main_EXP_G_VULN_GEO_LOCATION_train_single_ANN_remapping_wrapper.py -lr 0.001 -hlc 3 -hnc [500,500,500] -e 1000 -bs 10000 --id_gpu 1 --perc_gpu 0.2 -nu 0.02 -tr_size 17665 -val_size 1763 -ts_size 17665 -tr_iter 2 -mn model_10000_b && python main_EXP_G_VULN_GEO_LOCATION_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn model_10000_b --tr_size [17665] --ts_size [17665] --val_size [1763] --bg_test_it 0 --end_test_it 25 &

#python main_EXP_G_VULN_GEO_LOCATION_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn model_512 --tr_size [17665] --ts_size [17665] --val_size [1763] --bg_test_it 0 --end_test_it 25 &
#python main_EXP_G_VULN_GEO_LOCATION_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn model_1000 --tr_size [17665] --ts_size [17665] --val_size [1763] --bg_test_it 0 --end_test_it 25 &
#python main_EXP_G_VULN_GEO_LOCATION_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn model_4000 --tr_size [17665] --ts_size [17665] --val_size [1763] --bg_test_it 0 --end_test_it 25 &
#python main_EXP_G_VULN_GEO_LOCATION_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn model_10000 --tr_size [17665] --ts_size [17665] --val_size [1763] --bg_test_it 0 --end_test_it 25 &

#python main_EXP_G_VULN_GEO_LOCATION_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn model_512_b --tr_size [17665] --ts_size [17665] --val_size [1763] --bg_test_it 0 --end_test_it 25 &
#python main_EXP_G_VULN_GEO_LOCATION_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn model_1000_b --tr_size [17665] --ts_size [17665] --val_size [1763] --bg_test_it 0 --end_test_it 25 &
#python main_EXP_G_VULN_GEO_LOCATION_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn model_4000_b --tr_size [17665] --ts_size [17665] --val_size [1763] --bg_test_it 0 --end_test_it 25 &
#python main_EXP_G_VULN_GEO_LOCATION_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn model_10000_b --tr_size [17665] --ts_size [17665] --val_size [1763] --bg_test_it 0 --end_test_it 25 &

wait

echo "All done"