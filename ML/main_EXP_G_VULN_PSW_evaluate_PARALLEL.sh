#!/bin/bash

echo "HELLO MARCO"
############################################################################################################################

############################################################################################################################
######################################################  KNN  ###############################################################
############################################################################################################################

#python main_EXP_G_VULN_PSW_evaluate_KNN_remapping_wrapper.py --tr_size [10000] --ts_size [50000] --val_size [1000] --bg_test_it 0 --end_test_it 5 &
#python main_EXP_G_VULN_PSW_evaluate_KNN_remapping_wrapper.py --tr_size [10000] --ts_size [50000] --val_size [1000] --bg_test_it 5 --end_test_it 10 &

#python main_EXP_G_VULN_PSW_evaluate_KNN_remapping_wrapper.py --tr_size [30000] --ts_size [50000] --val_size [3000] --bg_test_it 0 --end_test_it 3 &
#python main_EXP_G_VULN_PSW_evaluate_KNN_remapping_wrapper.py --tr_size [30000] --ts_size [50000] --val_size [3000] --bg_test_it 3 --end_test_it 6 &
#python main_EXP_G_VULN_PSW_evaluate_KNN_remapping_wrapper.py --tr_size [30000] --ts_size [50000] --val_size [3000] --bg_test_it 6 --end_test_it 10 &

#python main_EXP_G_VULN_PSW_evaluate_KNN_remapping_wrapper.py --tr_size [50000] --ts_size [50000] --val_size [5000] --bg_test_it 0 --end_test_it 2 &
#python main_EXP_G_VULN_PSW_evaluate_KNN_remapping_wrapper.py --tr_size [50000] --ts_size [50000] --val_size [5000] --bg_test_it 2 --end_test_it 4 &
#python main_EXP_G_VULN_PSW_evaluate_KNN_remapping_wrapper.py --tr_size [50000] --ts_size [50000] --val_size [5000] --bg_test_it 4 --end_test_it 6 &
#python main_EXP_G_VULN_PSW_evaluate_KNN_remapping_wrapper.py --tr_size [50000] --ts_size [50000] --val_size [5000] --bg_test_it 6 --end_test_it 8 &
#python main_EXP_G_VULN_PSW_evaluate_KNN_remapping_wrapper.py --tr_size [50000] --ts_size [50000] --val_size [5000] --bg_test_it 8 --end_test_it 10 &

############################################################################################################################
######################################################  ANN  ###############################################################
############################################################################################################################

python main_EXP_G_VULN_PSW_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn model_3 --tr_size [10000] --ts_size [50000] --val_size [1000] --bg_test_it 0 --end_test_it 10 &
python main_EXP_G_VULN_PSW_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn model_3 --tr_size [30000] --ts_size [50000] --val_size [3000] --bg_test_it 0 --end_test_it 10 &
python main_EXP_G_VULN_PSW_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn model_3 --tr_size [50000] --ts_size [50000] --val_size [5000] --bg_test_it 0 --end_test_it 10 &

############################################################################################################################
######################################################  FREQ  ###############################################################
############################################################################################################################

#python main_EXP_G_VULN_EXP_PSW_evaluate_FREQUENTIST_wrapper.py --tr_size [10000] --ts_size [50000] --val_size [1000] --bg_test_it 0 --end_test_it 5 &
#python main_EXP_G_VULN_EXP_PSW_evaluate_FREQUENTIST_wrapper.py --tr_size [10000] --ts_size [50000] --val_size [1000] --bg_test_it 5 --end_test_it 10 &

#python main_EXP_G_VULN_EXP_PSW_evaluate_FREQUENTIST_wrapper.py --tr_size [30000] --ts_size [50000] --val_size [3000] --bg_test_it 0 --end_test_it 3 &
#python main_EXP_G_VULN_EXP_PSW_evaluate_FREQUENTIST_wrapper.py --tr_size [30000] --ts_size [50000] --val_size [3000] --bg_test_it 3 --end_test_it 6 &
#python main_EXP_G_VULN_EXP_PSW_evaluate_FREQUENTIST_wrapper.py --tr_size [30000] --ts_size [50000] --val_size [3000] --bg_test_it 6 --end_test_it 10 &

#python main_EXP_G_VULN_EXP_PSW_evaluate_FREQUENTIST_wrapper.py --tr_size [50000] --ts_size [50000] --val_size [5000] --bg_test_it 0 --end_test_it 2 &
#python main_EXP_G_VULN_EXP_PSW_evaluate_FREQUENTIST_wrapper.py --tr_size [50000] --ts_size [50000] --val_size [5000] --bg_test_it 2 --end_test_it 4 &
#python main_EXP_G_VULN_EXP_PSW_evaluate_FREQUENTIST_wrapper.py --tr_size [50000] --ts_size [50000] --val_size [5000] --bg_test_it 4 --end_test_it 6 &
#python main_EXP_G_VULN_EXP_PSW_evaluate_FREQUENTIST_wrapper.py --tr_size [50000] --ts_size [50000] --val_size [5000] --bg_test_it 6 --end_test_it 8 &
#python main_EXP_G_VULN_EXP_PSW_evaluate_FREQUENTIST_wrapper.py --tr_size [50000] --ts_size [50000] --val_size [5000] --bg_test_it 8 --end_test_it 10 &

############################################################################################################################

wait
echo "All done"