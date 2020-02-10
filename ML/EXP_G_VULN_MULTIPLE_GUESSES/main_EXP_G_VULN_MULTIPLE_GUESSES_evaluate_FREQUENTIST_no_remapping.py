import numpy as np
import pandas as pn
import sys
from utilities_pckg import g_vuln_freq_computation
from utilities_pckg.runtime_error_handler import exception_call as excpt_cll
from utilities_pckg import utilities
from utilities_pckg import g_vuln_computation
import time
from collections import Counter

APPROACHES = ['FREQUENTIST']
TRAINING_SET_SIZE = [10000, 30000, 50000]
# TRAINING_SET_SIZE = [int(sys.argv[1])]
TEST_SET_SIZE = [50000]
# TEST_SET_SIZE = [int(sys.argv[3])]
VALIDATION_SET_SIZE = [1000, 3000, 5000]
# VALIDATION_SET_SIZE = [int(sys.argv[2])]
# TEST_ITERATIONS = 1  # 100
TEST_ITERATIONS_BEG = 0
TEST_ITERATIONS_END = 50
TRAIN_ITERATIONS = 5
SECRETS_CARD = 10
NUMBER_OF_GUESSES = 2
ENHANCED = False

RESULT_FOLDER = "/home/comete/mromanel/MILES_EXP/EXP_G_VULN_MULTIPLE_GUESSES/RESULT_FOLDER/"
utilities.createFolder(RESULT_FOLDER)

RESULT_FOLDER = "/home/comete/mromanel/MILES_EXP/EXP_G_VULN_MULTIPLE_GUESSES/RESULT_FOLDER/FREQUENTIST/"
utilities.createFolder(RESULT_FOLDER)

DATA_FOLDER = "/home/comete/mromanel/MILES_EXP/EXP_G_VULN_MULTIPLE_GUESSES/DATA_FOLDER/"

G_MATRIX_PATH = '/home/comete/mromanel/MILES_EXP/EXP_G_VULN_MULTIPLE_GUESSES/G_MAT_FOLDER/g_matrix_10_secrets_' + \
                str(NUMBER_OF_GUESSES) + '_guesses.pkl'
G_MATRIX_ROWS_PATH = '/home/comete/mromanel/MILES_EXP/EXP_G_VULN_MULTIPLE_GUESSES/G_MAT_FOLDER/g_matrix_10_secrets_' + \
                     str(NUMBER_OF_GUESSES) + \
                     '_guesses_rows.pkl'
G_MATRIX_COLS_PATH = '/home/comete/mromanel/MILES_EXP/EXP_G_VULN_MULTIPLE_GUESSES/G_MAT_FOLDER/g_matrix_10_secrets_' + \
                     str(NUMBER_OF_GUESSES) + '_guesses_cols.pkl'
G_MATRIX_ALL_GUESSES = '/home/comete/mromanel/MILES_EXP/EXP_G_VULN_MULTIPLE_GUESSES/G_MAT_FOLDER/g_matrix_10_secrets_' + \
                       str(NUMBER_OF_GUESSES) + '_guesses_all_possible_guesses_dic.pkl'


def read_command_line_options():
    thismodule = sys.modules[__name__]
    for idx, key_val in enumerate(sys.argv, 0):

        if key_val in ['--tr_size'] and len(sys.argv) > idx + 1:
            #   it must be something like -hnc [1,2,3]
            string_to_be_adapted = sys.argv[idx + 1].strip()
            print string_to_be_adapted
            if string_to_be_adapted[0] != '[' or string_to_be_adapted[-1] != ']':
                excpt_cll(idx=idx, key_val=key_val)
            else:
                strip_string_to_be_adapted = string_to_be_adapted[1:-1]
                split_list = strip_string_to_be_adapted.split(',')
                TRAINING_SET_SIZE_TMP = []
                for item in split_list:
                    try:
                        TRAINING_SET_SIZE_TMP.append(int(item))
                    except ValueError as val_err:
                        excpt_cll(idx=idx, key_val=key_val)
                thismodule.TRAINING_SET_SIZE = TRAINING_SET_SIZE_TMP

        if key_val in ['--ts_size'] and len(sys.argv) > idx + 1:
            #   it must be something like -hnc [1,2,3]
            string_to_be_adapted = sys.argv[idx + 1].strip()
            print string_to_be_adapted
            if string_to_be_adapted[0] != '[' or string_to_be_adapted[-1] != ']':
                excpt_cll(idx=idx, key_val=key_val)
            else:
                strip_string_to_be_adapted = string_to_be_adapted[1:-1]
                split_list = strip_string_to_be_adapted.split(',')
                TEST_SET_SIZE_TMP = []
                for item in split_list:
                    try:
                        TEST_SET_SIZE_TMP.append(int(item))
                    except ValueError as val_err:
                        excpt_cll(idx=idx, key_val=key_val)
                thismodule.TEST_SET_SIZE = TEST_SET_SIZE_TMP

        if key_val in ['--val_size'] and len(sys.argv) > idx + 1:
            #   it must be something like -hnc [1,2,3]
            string_to_be_adapted = sys.argv[idx + 1].strip()
            print string_to_be_adapted
            if string_to_be_adapted[0] != '[' or string_to_be_adapted[-1] != ']':
                excpt_cll(idx=idx, key_val=key_val)
            else:
                strip_string_to_be_adapted = string_to_be_adapted[1:-1]
                split_list = strip_string_to_be_adapted.split(',')
                VAL_SET_SIZE_TMP = []
                for item in split_list:
                    try:
                        VAL_SET_SIZE_TMP.append(int(item))
                    except ValueError as val_err:
                        excpt_cll(idx=idx, key_val=key_val)
                thismodule.VALIDATION_SET_SIZE = VAL_SET_SIZE_TMP

        if key_val in ['--bg_test_it'] and len(sys.argv) > idx + 1:
            try:
                thismodule.TEST_ITERATIONS_BEG = int(sys.argv[idx + 1].strip())
            except ValueError as val_err:
                excpt_cll(idx=idx, key_val=key_val)

        if key_val in ['--enhanced', 'enh'] and len(sys.argv) > idx + 1:
            if sys.argv[idx + 1].strip() == 'True':
                try:
                    thismodule.ENHANCED = True
                except ValueError as val_err:
                    excpt_cll(idx=idx, key_val=key_val)
            elif sys.argv[idx + 1].strip() == 'False':
                try:
                    thismodule.ENHANCED = False
                except ValueError as val_err:
                    excpt_cll(idx=idx, key_val=key_val)
            else:
                excpt_cll(idx="invalid bool", key_val=key_val)

        if key_val in ['--end_test_it'] and len(sys.argv) > idx + 1:
            try:
                thismodule.TEST_ITERATIONS_END = int(sys.argv[idx + 1].strip())
            except ValueError as val_err:
                excpt_cll(idx=idx, key_val=key_val)


def main_EXP_G_VULN_MULTIPLE_GUESSES_evaluate_FREQ_no_remapping():
    read_command_line_options()

    if len(VALIDATION_SET_SIZE) != len(TRAINING_SET_SIZE):
        sys.exit("The set size lists must all contain the same amount of items.")

    loaded_g_matrix = pn.read_pickle(path=G_MATRIX_PATH)
    loaded_g_matrix_rows = pn.read_pickle(path=G_MATRIX_ROWS_PATH)
    loaded_g_matrix_cols = pn.read_pickle(path=G_MATRIX_COLS_PATH)
    loaded_all_guesses = pn.read_pickle(path=G_MATRIX_ALL_GUESSES)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  FREQ approach  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%$$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if 'FREQUENTIST' in APPROACHES:
        print("\n####################################################################################")
        print("###################################  FREQ approach  ##################################")
        print("####################################################################################\n")

        #   store the values of the error estimations via all the possible methods
        FREQUENTIST_Rf_values = []

        #   number of test samples
        number_of_test_samples = []

        #   number of training samples
        number_of_training_samples = []

        #   iterator over different training sets iterations
        training_iteration = []

        #   iterator over different test sets iterations
        test_iteration = []

        for size_list_iterator in range(len(TRAINING_SET_SIZE)):

            #   select the current values for the sizes (useful to keep track in the names of the
            training_set_size = TRAINING_SET_SIZE[size_list_iterator]
            validation_set_size = VALIDATION_SET_SIZE[size_list_iterator]
            test_set_size = TEST_SET_SIZE[0]

            for train_iteration in range(TRAIN_ITERATIONS):
                training_data = pn.read_pickle(DATA_FOLDER + str(training_set_size) + "_training_and_" + str(
                    validation_set_size) + "_validation_store_folder_train_iteration_" + str(
                    train_iteration) + "/training_set.pkl")
                counter_secrets = Counter(training_data[:, -1])

                print "\n\n\n\n\n\n\n\n#################################  test_size: " + str(
                    test_set_size) + " ################################"

                FREQ_g_vuln_file = open(RESULT_FOLDER + "/FREQ_" + str(training_set_size) + "_training_and_" + str(
                    validation_set_size) + "_validation_file_R_estimate_iteration_" + str(train_iteration) + "_" + str(
                    test_set_size) + "_test_set_size_test_iter_up_to_" + str(TEST_ITERATIONS_END) + ".txt", "wa")

                for test_iterator in range(TEST_ITERATIONS_BEG, TEST_ITERATIONS_END):
                    now = time.time()
                    print "\n\n\n#################################  test_set_" + str(
                        test_iterator) + " ################################"

                    FREQ_g_vuln_file.write("\n\n\n#################################  test_set_" + str(
                        test_iterator) + " ################################")

                    test_set = pn.read_pickle(
                        path=DATA_FOLDER + str(test_set_size) + "_size_test_sets/test_set_" + str(test_iterator) + ".pkl")

                    FREQ_VULN = \
                        g_vuln_freq_computation.g_vuln_freq_computation_multiple_guesses_monodimensional_observables(
                            train_data=training_data,
                            test_data=test_set,
                            g_mat=loaded_g_matrix,
                            g_mat_cols=loaded_g_matrix_cols,
                            all_possible_guesses_dic=loaded_all_guesses,
                            n_guesses=NUMBER_OF_GUESSES,
                            counter_secrets=counter_secrets)

                    print("\nFREQ_VULN = " + str(FREQ_VULN))

                    FREQ_g_vuln_file.write("\nFREQ_VULN = " + str(FREQ_VULN))
                    FREQUENTIST_Rf_values.append(FREQ_VULN)
                    number_of_test_samples.append(test_set_size)
                    number_of_training_samples.append(training_set_size)
                    training_iteration.append(train_iteration)
                    test_iteration.append(test_iterator)

                FREQ_g_vuln_file.close()

        FREQUENTIST_Rf_values = np.array(FREQUENTIST_Rf_values, dtype=np.float64)
        number_of_test_samples = np.array(number_of_test_samples, dtype=np.int32)
        number_of_training_samples = np.array(number_of_training_samples, dtype=np.int32)
        training_iteration = np.array(training_iteration, dtype=np.int32)
        test_iteration = np.array(test_iteration, dtype=np.int32)

        result_matrix = np.column_stack((FREQUENTIST_Rf_values, number_of_test_samples))
        # print result_matrix.shape

        result_matrix = np.column_stack((result_matrix, number_of_training_samples))
        # print result_matrix.shape

        result_matrix = np.column_stack((result_matrix, training_iteration))
        # print result_matrix.shape

        result_matrix = np.column_stack((result_matrix, test_iteration))
        # print result_matrix.shape

        result_df = pn.DataFrame(data=result_matrix,
                                 columns=["FREQUENTIST_Rf_values", "number_of_test_samples",
                                          "number_of_training_samples",
                                          "train_iteration",
                                          "test_iteration"])
        result_df.to_pickle(
            path=RESULT_FOLDER + "/FREQ_training_and_validation_result_df_train_size_" + str(
                TRAINING_SET_SIZE[0]) + "_up_to_test_iter_" + str(TEST_ITERATIONS_END) + ".pkl")
