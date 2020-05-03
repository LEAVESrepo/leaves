import sys
import numpy as np
import pandas as pn
from utilities_pckg import g_vuln_freq_computation, utilities
from utilities_pckg.runtime_error_handler import exception_call as excpt_cll

APPROACHES = ['FREQUENTIST']
TRAINING_SET_SIZE = [10000, 50000]
TEST_SET_SIZE = [10000, 50000]
VALIDATION_SET_SIZE = [1000, 5000]
TEST_ITERATIONS_BEG = 0
TEST_ITERATIONS_END = 50
TRAIN_ITERATIONS = 3

RESULT_FOLDER = "/home/comete/mromanel/MILES_EXP/EXP_PSW/RESULT_FOLDER/FREQUENTIST/"
utilities.createFolder(RESULT_FOLDER)

DATA_FOLDER = "/home/comete/mromanel/MILES_EXP/EXP_PSW/DATA_FOLDER/"
DATA_FOLDER_TEST = "/home/comete/mromanel/MILES_EXP/EXP_PSW/DATA_FOLDER/"

G_MATRIX_PATH = '/home/comete/mromanel/MILES_EXP/EXP_PSW/G_MAT_FOLDER/G_MAT'


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

        if key_val in ['--end_test_it'] and len(sys.argv) > idx + 1:
            try:
                thismodule.TEST_ITERATIONS_END = int(sys.argv[idx + 1].strip())
            except ValueError as val_err:
                excpt_cll(idx=idx, key_val=key_val)


def main_EXP_G_VULN_PSW_evaluate_FREQUENTIST():
    read_command_line_options()

    if len(VALIDATION_SET_SIZE) != len(TRAINING_SET_SIZE):
        sys.exit("The set size lists must all contain the same amount of items.")

    loaded_g_matrix = pn.read_pickle(path=G_MATRIX_PATH)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  FREQ approach  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%$$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if 'FREQUENTIST' in APPROACHES:
        print("\n####################################################################################")
        print("###################################  FREQ approach  ##################################")
        print("####################################################################################\n")

        #   store the values of the error estimations via all the possible methods
        FREQ_Rf_values = []

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
            test_set_size = TEST_SET_SIZE[size_list_iterator]

            for train_iteration in range(TRAIN_ITERATIONS):
                filepath = RESULT_FOLDER + str(training_set_size) + "_training_size_and_" + str(
                    validation_set_size) + "_validation_size_iteration_" + str(train_iteration) + "_"

                tr_set = pn.read_pickle(path=DATA_FOLDER + str(training_set_size) + "_training_and_" + str(
                    validation_set_size) + "_validation_store_folder_train_iteration_" + str(
                    train_iteration) + "/training_set.pkl")

                research_obs_dic = {}
                for obs in np.unique(tr_set[:, 0]):
                    research_obs_dic[obs] = np.where(tr_set[:, 0] == obs)[0]

                research_sec_dic = {}
                for sec in np.unique(np.array(tr_set[:, 1], dtype=int)):
                    research_sec_dic[sec] = np.where(np.array(tr_set[:, 1], dtype=int) == sec)[0]

                print "\n\n\n\n\n\n\n\n#################################  test_size: " + str(
                    test_set_size) + " ################################"
                FREQUENTIST_file_Rf_g_leak = open(filepath + str(training_set_size) + "_training_and_" + str(
                    validation_set_size) + "_validation_file_R_estimate_iteration_" + str(train_iteration) + "_" + str(
                    test_set_size) + "_test_set_size_test_iter_up_to_" + str(TEST_ITERATIONS_END) + ".txt", "wa")

                for test_iterator in range(TEST_ITERATIONS_BEG, TEST_ITERATIONS_END):
                    print "\n\n\n#################################  test_set_" + str(
                        test_iterator) + " ################################"

                    FREQUENTIST_file_Rf_g_leak.write("\n\n\n#################################  test_set_" + str(
                        test_iterator) + " ################################")

                    test_set = pn.read_pickle(
                        path=DATA_FOLDER + str(test_set_size) + "_size_test_sets/test_set_" + str(
                            test_iterator) + ".pkl")

                    most_freq_secret = utilities.find_most_frequent_symbol_in_array(
                        array=np.array(tr_set[:, 1], dtype=int))

                    g_mat_most_frequent_secret_idx = most_freq_secret  # index and values are the same

                    unq_obs_tr, unq_obs_tr_cnt = np.unique(tr_set[:, 0], return_counts=True)
                    unq_secr_tr = np.unique(np.array(tr_set[:, 1], dtype=int))
                    test_obs_unique = np.unique(test_set[:, 0])

                    best_mono_guess = g_vuln_freq_computation.find_best_mono_guess_general_approach_positional(
                        g_mat=loaded_g_matrix,
                        g_mat_rows=[0, 1],
                        research_obs_dic=research_obs_dic,
                        research_sec_dic=research_sec_dic,
                        g_mat_most_frequent_secret_idx=g_mat_most_frequent_secret_idx,
                        unq_obs_tr=unq_obs_tr,
                        unq_obs_tr_cnt=unq_obs_tr_cnt,
                        unq_secr_tr=unq_secr_tr,
                        test_obs_unique=test_obs_unique)

                    test_obs_unq = np.unique(test_set[:, 0])
                    test_secrets_col_int = np.array(test_set[:, 1], dtype=int)

                    Rf_FREQ_g_leak = g_vuln_freq_computation.compute_freq_g_vuln_mono_guess_general_approach_posiitional(
                        test_set=test_set,
                        best_mono_guess_res=best_mono_guess,
                        g_mat=loaded_g_matrix,
                        g_mat_rows=[0, 1],
                        test_obs_unq=test_obs_unq,
                        test_secrets_col_int=test_secrets_col_int)

                    print("\nRf_FREQ_g_leak = " + str(Rf_FREQ_g_leak))

                    FREQUENTIST_file_Rf_g_leak.write("\nFREQUENTIST_file_Rf_g_leak = " + str(Rf_FREQ_g_leak))
                    FREQ_Rf_values.append(Rf_FREQ_g_leak)
                    number_of_test_samples.append(test_set_size)
                    number_of_training_samples.append(training_set_size)
                    training_iteration.append(train_iteration)
                    test_iteration.append(test_iterator)

                FREQUENTIST_file_Rf_g_leak.close()

        FREQ_Rf_values = np.array(FREQ_Rf_values, dtype=np.float64)
        number_of_test_samples = np.array(number_of_test_samples, dtype=np.int32)
        number_of_training_samples = np.array(number_of_training_samples, dtype=np.int32)
        training_iteration = np.array(training_iteration, dtype=np.int32)
        test_iteration = np.array(test_iteration, dtype=np.int32)

        result_matrix = np.column_stack((FREQ_Rf_values, number_of_test_samples))
        # print result_matrix.shape

        result_matrix = np.column_stack((result_matrix, number_of_training_samples))
        # print result_matrix.shape

        result_matrix = np.column_stack((result_matrix, training_iteration))
        # print result_matrix.shape

        result_matrix = np.column_stack((result_matrix, test_iteration))
        # print result_matrix.shape

        result_df = pn.DataFrame(data=result_matrix,
                                 columns=["FREQ_Rf_values", "number_of_test_samples",
                                          "number_of_training_samples",
                                          "train_iteration",
                                          "test_iteration"])
        result_df.to_pickle(
            path=RESULT_FOLDER + "/FREQ_training_and_validation_result_df_train_size_" + str(
                TRAINING_SET_SIZE[0]) + "_up_to_test_iter_" + str(TEST_ITERATIONS_END) + ".pkl")
