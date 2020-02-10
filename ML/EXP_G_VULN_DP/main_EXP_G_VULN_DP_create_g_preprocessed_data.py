"""preprocess data according to our idea"""

from utilities_pckg import g_function_manager
import os
import pandas as pn
import numpy as np

TRAINING_SET_SIZE = [10000, 30000, 50000]
TEST_SET_SIZE = [50000]
VALIDATION_SET_SIZE = [1000, 3000, 5000]
# TRAINING_SET_SIZE = [int(sys.argv[1])]
# TEST_SET_SIZE = [int(sys.argv[2])]
# VALIDATION_SET_SIZE = [int(sys.argv[3])]
TRAIN_ITERATIONS = 5
# TRAIN_ITERATIONS = int(sys.argv[4])

EXP_G_VULN_DP_FOLDER = "/home/comete/mromanel/MILES_EXP/EXP_G_VULN_DP_FOLDER/"

DATA_FOLDER = EXP_G_VULN_DP_FOLDER + "DATA_FOLDER/"

G_MATRIX_PATH = EXP_G_VULN_DP_FOLDER + 'G_OBJ/g_mat.pkl'
G_MATRIX_ROWS_PATH = EXP_G_VULN_DP_FOLDER + 'G_OBJ/g_mat_rows.pkl'
G_MATRIX_COLS_PATH = EXP_G_VULN_DP_FOLDER + 'G_OBJ/g_mat_cols.pkl'
G_MATRIX_K_PATH = EXP_G_VULN_DP_FOLDER + 'G_OBJ/K.pkl'

FINAL_DATA_FOLDER = EXP_G_VULN_DP_FOLDER + "DATA_FOLDER_AFTER_OUR_PREPROCESSING/"

if not os.path.exists(FINAL_DATA_FOLDER):
    os.makedirs(FINAL_DATA_FOLDER)


def main_EXP_G_VULN_DP_create_g_preprocessed_data():
    print "Loading G matrix..."

    loaded_g_matrix = pn.read_pickle(path=G_MATRIX_PATH)
    loaded_g_matrix_rows = pn.read_pickle(path=G_MATRIX_ROWS_PATH)
    loaded_g_matrix_cols = pn.read_pickle(path=G_MATRIX_COLS_PATH)
    loaded_g_matrix_K = 1.  # pn.read_pickle(path=G_MATRIX_K_PATH)

    print "Ended loading G matrix. Dimensions: " + str(loaded_g_matrix.shape[0]) + ", " + str(loaded_g_matrix.shape[1])
    for i_ter in range(len(TRAINING_SET_SIZE)):
        training_set_size = TRAINING_SET_SIZE[i_ter]
        validation_set_size = VALIDATION_SET_SIZE[i_ter]
        test_set_size = TEST_SET_SIZE[0]

        for train_iter in range(TRAIN_ITERATIONS):
            read_folder = DATA_FOLDER + str(training_set_size) + "_training_and_" + str(
                validation_set_size) + "_validation_store_folder_train_iteration_" + str(train_iter)

            write_folder = FINAL_DATA_FOLDER + str(training_set_size) + "_training_and_" + str(
                validation_set_size) + "_validation_and_" + str(
                test_set_size) + "_test_store_folder_train_iteration_" + str(train_iter)

            if not os.path.exists(write_folder):
                os.makedirs(write_folder)

            training_set = pn.read_pickle(read_folder + "/training_set.pkl")
            validation_set = pn.read_pickle(read_folder + "/validation_set.pkl")

            #   X_train is the observable and y_train the secret, z_train is the remapping
            training_set_post_processing = g_function_manager.create_D_prime_multidimensional_inputs(
                D=training_set,
                colnames=['0', '1', '2', '3', '4', 'S_train', 'Z_train'],
                g_matrix=loaded_g_matrix,
                g_col_names=loaded_g_matrix_cols,
                g_row_names=loaded_g_matrix_rows,
                K=loaded_g_matrix_K)

            pn.to_pickle(training_set_post_processing, path=write_folder + "/training_set.pkl")
            # pn.to_pickle(training_set_post_processing_encoded_supervision,
            #             path=write_folder + "/training_set_encoded_supervision.pkl")

            validation_set_post_processing = g_function_manager.create_D_prime_multidimensional_inputs(
                D=validation_set,
                colnames=['0', '1', '2', '3', '4', 'S_val', 'Z_val'],
                g_matrix=loaded_g_matrix,
                g_col_names=loaded_g_matrix_cols,
                g_row_names=loaded_g_matrix_rows,
                K=loaded_g_matrix_K)
            pn.to_pickle(validation_set_post_processing, path=write_folder + "/validation_set.pkl")
            # pn.to_pickle(validation_set_post_processing_encoded_supervision,
            #             path=write_folder + "/validation_set_encoded_supervision.pkl")

            # for test_iter in range(TEST_ITERATIONS):
            #     test_set = pn.read_pickle(test_sets_folder + "/test_set_" + str(test_iter) + ".pkl")
            #     test_set_post_processing = g_function_manager.create_D_prime_multidimensional_inputs(
            #         D=test_set,
            #         colnames=['0', '1', '2', '3', '4', 'S_test', 'Z_test'],
            #         g_matrix=loaded_g_matrix,
            #         g_col_names=loaded_g_matrix_cols,
            #         g_row_names=loaded_g_matrix_rows,
            #         K=loaded_g_matrix_K)
            #
            #     if not os.path.exists(write_folder + "/" + str(test_set_size) + "_size_test_sets/"):
            #         os.makedirs(write_folder + "/" + str(test_set_size) + "_size_test_sets/")
            #
            #     pn.to_pickle(test_set_post_processing,
            #                  write_folder + "/" + str(test_set_size) + "_size_test_sets/test_set_" + str(test_iter))
            #     # pn.to_pickle(test_set_post_processing_encoded_supervision, write_folder + "/" + str(
            #     #    test_set_size) + "_size_test_sets/encoded_supervision_test_set_" + str(test_iter))
