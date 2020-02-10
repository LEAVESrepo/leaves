#!/bin/bash

echo "HELLO MARCO"

python main_EXP_G_VULN_MULTIPLE_GUESSES_create_g_preprocessed_data_wrapper.py --tr_size [10000] --ts_size [10000] --val_size [1000] &
python main_EXP_G_VULN_MULTIPLE_GUESSES_create_g_preprocessed_data_wrapper.py --tr_size [30000] --ts_size [30000] --val_size [3000] &
python main_EXP_G_VULN_MULTIPLE_GUESSES_create_g_preprocessed_data_wrapper.py --tr_size [50000] --ts_size [50000] --val_size [5000] &

wait

echo "All done"
