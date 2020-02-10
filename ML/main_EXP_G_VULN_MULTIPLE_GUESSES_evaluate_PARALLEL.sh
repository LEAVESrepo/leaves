#!/bin/bash

echo "HELLO MARCO"
############################################################################################################################


############################################################################################################################
######################################################  FREQ  ##############################################################
############################################################################################################################

python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_FREQ_no_remapping_wrapper.py --tr_size [10000] --ts_size [50000] --val_size [1000] --bg_test_it 0 --end_test_it 25 &
python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_FREQ_no_remapping_wrapper.py --tr_size [10000] --ts_size [50000] --val_size [1000] --bg_test_it 25 --end_test_it 50 &
##python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_FREQ_no_remapping_wrapper.py --tr_size [10000] --ts_size [10000] --val_size [1000] --bg_test_it 50 --end_test_it 75 &
##python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_FREQ_no_remapping_wrapper.py --tr_size [10000] --ts_size [10000] --val_size [1000] --bg_test_it 75 --end_test_it 100 &

python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_FREQ_no_remapping_wrapper.py --tr_size [30000] --ts_size [50000] --val_size [3000] --bg_test_it 0 --end_test_it 25 &
python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_FREQ_no_remapping_wrapper.py --tr_size [30000] --ts_size [50000] --val_size [3000] --bg_test_it 25 --end_test_it 50 &
##python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_FREQ_no_remapping_wrapper.py --tr_size [30000] --ts_size [30000] --val_size [3000] --bg_test_it 50 --end_test_it 75 &
##python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_FREQ_no_remapping_wrapper.py --tr_size [30000] --ts_size [30000] --val_size [3000] --bg_test_it 75 --end_test_it 100 &

python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_FREQ_no_remapping_wrapper.py --tr_size [50000] --ts_size [50000] --val_size [5000] --bg_test_it 0 --end_test_it 25 &
python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_FREQ_no_remapping_wrapper.py --tr_size [50000] --ts_size [50000] --val_size [5000] --bg_test_it 25 --end_test_it 50 &
##python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_FREQ_no_remapping_wrapper.py --tr_size [50000] --ts_size [50000] --val_size [5000] --bg_test_it 50 --end_test_it 75 &
##python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_FREQ_no_remapping_wrapper.py --tr_size [50000] --ts_size [50000] --val_size [5000] --bg_test_it 75 --end_test_it 100 &

#wait

############################################################################################################################
######################################################  KNN  ###############################################################
############################################################################################################################

#python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_KNN_remapping_wrapper.py --tr_size [10000] --ts_size [50000] --val_size [1000] --bg_test_it 0 --end_test_it 25 &
#python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_KNN_remapping_wrapper.py --tr_size [10000] --ts_size [50000] --val_size [1000] --bg_test_it 25 --end_test_it 50 &
##python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_KNN_remapping_wrapper.py --tr_size [10000] --ts_size [10000] --val_size [1000] --bg_test_it 50 --end_test_it 75 &
##python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_KNN_remapping_wrapper.py --tr_size [10000] --ts_size [10000] --val_size [1000] --bg_test_it 75 --end_test_it 100 &

#python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_KNN_remapping_wrapper.py --tr_size [30000] --ts_size [50000] --val_size [3000] --bg_test_it 0 --end_test_it 25 &
#python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_KNN_remapping_wrapper.py --tr_size [30000] --ts_size [50000] --val_size [3000] --bg_test_it 25 --end_test_it 50 &
##python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_KNN_remapping_wrapper.py --tr_size [30000] --ts_size [30000] --val_size [3000] --bg_test_it 50 --end_test_it 75 &
##python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_KNN_remapping_wrapper.py --tr_size [30000] --ts_size [30000] --val_size [3000] --bg_test_it 75 --end_test_it 100 &

#python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_KNN_remapping_wrapper.py --tr_size [50000] --ts_size [50000] --val_size [5000] --bg_test_it 0 --end_test_it 25 &
#python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_KNN_remapping_wrapper.py --tr_size [50000] --ts_size [50000] --val_size [5000] --bg_test_it 25 --end_test_it 50 &
##python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_KNN_remapping_wrapper.py --tr_size [50000] --ts_size [50000] --val_size [5000] --bg_test_it 50 --end_test_it 75 &
##python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_KNN_remapping_wrapper.py --tr_size [50000] --ts_size [50000] --val_size [5000] --bg_test_it 75 --end_test_it 100 &

#wait

############################################################################################################################
######################################################  ANN  ###############################################################
############################################################################################################################

#python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn model_1000 --tr_size [10000] --ts_size [50000] --val_size [1000] --bg_test_it 0 --end_test_it 25 &
#python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn model_1000 --tr_size [10000] --ts_size [50000] --val_size [1000] --bg_test_it 25 --end_test_it 50 &
##python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn model_1000_bis --tr_size [10000] --ts_size [10000] --val_size [1000] --bg_test_it 50 --end_test_it 75 &
##python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_ANN_remapping_wrapper.py --id_gpu 0 --perc_gpu 0.2 -mn model_1000_bis --tr_size [10000] --ts_size [10000] --val_size [1000] --bg_test_it 75 --end_test_it 100 &

#python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_ANN_remapping_wrapper.py --id_gpu 1 --perc_gpu 0.2 -mn model_1000 --tr_size [30000] --ts_size [50000] --val_size [3000] --bg_test_it 0 --end_test_it 25 &
#python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_ANN_remapping_wrapper.py --id_gpu 1 --perc_gpu 0.2 -mn model_1000 --tr_size [30000] --ts_size [50000] --val_size [3000] --bg_test_it 25 --end_test_it 50 &
##python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_ANN_remapping_wrapper.py --id_gpu 1 --perc_gpu 0.2 -mn model_1000_bis --tr_size [30000] --ts_size [30000] --val_size [3000] --bg_test_it 50 --end_test_it 75 &
##python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_ANN_remapping_wrapper.py --id_gpu 1 --perc_gpu 0.2 -mn model_1000_bis --tr_size [30000] --ts_size [30000] --val_size [3000] --bg_test_it 75 --end_test_it 100 &

#python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_ANN_remapping_wrapper.py --id_gpu 2 --perc_gpu 0.2 -mn model_1000 --tr_size [50000] --ts_size [50000] --val_size [5000] --bg_test_it 0 --end_test_it 25 &
#python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_ANN_remapping_wrapper.py --id_gpu 2 --perc_gpu 0.2 -mn model_1000 --tr_size [50000] --ts_size [50000] --val_size [5000] --bg_test_it 25 --end_test_it 50 &
##python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_ANN_remapping_wrapper.py --id_gpu 2 --perc_gpu 0.2 -mn model_1000_bis --tr_size [50000] --ts_size [50000] --val_size [5000] --bg_test_it 50 --end_test_it 75 &
##python main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_ANN_remapping_wrapper.py --id_gpu 2 --perc_gpu 0.2 -mn model_1000_bis --tr_size [50000] --ts_size [50000] --val_size [5000] --bg_test_it 75 --end_test_it 100 &

wait

############################################################################################################################

echo "All done"