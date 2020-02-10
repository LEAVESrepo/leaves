"""preprocess data according to our idea"""

import os
import sys
import pandas as pn
from utilities_pckg import g_function_manager
from utilities_pckg.runtime_error_handler import exception_call as excpt_cll

NU = 0.002
TRAINING_SET_SIZE = [5000, 10000, 20000]
TEST_SET_SIZE = [5000, 5000, 5000]
VALIDATION_SET_SIZE = [500, 1000, 2000]
# TRAINING_SET_SIZE = [int(sys.argv[1])]
# TEST_SET_SIZE = [int(sys.argv[2])]
# VALIDATION_SET_SIZE = [int(sys.argv[3])]
TEST_ITERATIONS = 10
TRAIN_ITERATIONS = 1
# TRAIN_ITERATIONS = int(sys.argv[4])

EMPIRICAL_ESTIMATES_FOLDER = "/home/comete/mromanel/MILES_EXP/EMPIRICAL_ESTIMATES"

DATA_FOLDER = EMPIRICAL_ESTIMATES_FOLDER + "/DATA_FOLDER/"

G_MATRIX_PATH = '/home/comete/mromanel/MILES_EXP/EMPIRICAL_ESTIMATES/G_MAT_FOLDER/g_matrix.pkl'
G_MATRIX_ROWS_PATH = '/home/comete/mromanel/MILES_EXP/EMPIRICAL_ESTIMATES/G_MAT_FOLDER/' \
                     'g_matrix_rows.pkl'
G_MATRIX_COLS_PATH = '/home/comete/mromanel/MILES_EXP/EMPIRICAL_ESTIMATES/' \
                     'G_MAT_FOLDER/g_matrix_cols.pkl'


FINAL_DATA_FOLDER = EMPIRICAL_ESTIMATES_FOLDER + "/DATA_FOLDER_AFTER_OUR_PREPROCESSING/"

if not os.path.exists(FINAL_DATA_FOLDER):
    os.makedirs(FINAL_DATA_FOLDER)


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


def main_EMPIRICAL_ESTIMATES_create_g_preprocessed_data():
    print "Loading G matrix..."

    loaded_g_matrix = pn.read_pickle(path=G_MATRIX_PATH)
    loaded_g_matrix_rows = pn.read_pickle(path=G_MATRIX_ROWS_PATH)
    loaded_g_matrix_cols = pn.read_pickle(path=G_MATRIX_COLS_PATH)
    loaded_g_matrix_K = 1

    print "Ended loading G matrix. Dimensions: " + str(loaded_g_matrix.shape[0]) + ", " + str(loaded_g_matrix.shape[1])
    for i_ter in range(len(TRAINING_SET_SIZE)):
        training_set_size = TRAINING_SET_SIZE[i_ter]
        validation_set_size = VALIDATION_SET_SIZE[i_ter]
        test_set_size = TEST_SET_SIZE[i_ter]

        for train_iter in range(TRAIN_ITERATIONS):
            read_folder = DATA_FOLDER + str(training_set_size) + "_training_and_" + str(
                validation_set_size) + "_validation_and_" + str(
                test_set_size) + "_test_store_folder_train_iteration_" + str(train_iter)

            write_folder = FINAL_DATA_FOLDER + str(training_set_size) + "_training_and_" + str(
                validation_set_size) + "_validation_and_" + str(
                test_set_size) + "_test_store_folder_train_iteration_" + str(train_iter)

            if not os.path.exists(write_folder):
                os.makedirs(write_folder)

            training_set = pn.read_pickle(read_folder + "/training_set.pkl")
            validation_set = pn.read_pickle(read_folder + "/validation_set.pkl")
            test_sets_folder = read_folder + "/" + str(test_set_size) + "_size_test_sets"

            #   X_train is the observable and y_train the secret, z_train is the remapping
            training_set_post_processing, training_set_post_processing_encoded_supervision = \
                g_function_manager.create_D_prime(
                    D=training_set,
                    colnames=['O_train', 'S_train', 'Z_train'],
                    g_matrix=loaded_g_matrix,
                    g_col_names=loaded_g_matrix_cols,
                    g_row_names=loaded_g_matrix_rows,
                    K=loaded_g_matrix_K)

            pn.to_pickle(training_set_post_processing, path=write_folder + "/training_set.pkl")
            pn.to_pickle(training_set_post_processing_encoded_supervision,
                         path=write_folder + "/training_set_encoded_supervision.pkl")

            validation_set_post_processing, validation_set_post_processing_encoded_supervision = \
                g_function_manager.create_D_prime(D=validation_set,
                                                  colnames=['O_val', 'S_val', 'Z_val'],
                                                  g_matrix=loaded_g_matrix,
                                                  g_col_names=loaded_g_matrix_cols,
                                                  g_row_names=loaded_g_matrix_rows,
                                                  K=loaded_g_matrix_K)
            pn.to_pickle(validation_set_post_processing, path=write_folder + "/validation_set.pkl")
            pn.to_pickle(validation_set_post_processing_encoded_supervision,
                         path=write_folder + "/validation_set_encoded_supervision.pkl")

            for test_iter in range(TEST_ITERATIONS):
                test_set = pn.read_pickle(test_sets_folder + "/test_set_" + str(test_iter) + ".pkl")
                test_set_post_processing, test_set_post_processing_encoded_supervision = \
                    g_function_manager.create_D_prime(D=test_set,
                                                      colnames=['O_test', 'S_test', 'Z_test'],
                                                      g_matrix=loaded_g_matrix,
                                                      g_col_names=loaded_g_matrix_cols,
                                                      g_row_names=loaded_g_matrix_rows,
                                                      K=loaded_g_matrix_K)

                if not os.path.exists(write_folder + "/" + str(test_set_size) + "_size_test_sets/"):
                    os.makedirs(write_folder + "/" + str(test_set_size) + "_size_test_sets/")

                pn.to_pickle(test_set_post_processing,
                             write_folder + "/" + str(test_set_size) + "_size_test_sets/test_set_" + str(test_iter))
                pn.to_pickle(test_set_post_processing_encoded_supervision, write_folder + "/" + str(
                    test_set_size) + "_size_test_sets/encoded_supervision_test_set_" + str(test_iter))
