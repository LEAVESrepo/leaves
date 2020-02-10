import numpy as np
import pandas as pn
from utilities_pckg import empirical_estimate

TRAINING_SET_SIZE = [5000, 10000, 20000]
TEST_SET_SIZE = [5000, 5000, 5000]
VALIDATION_SET_SIZE = [500, 1000, 2000]
TEST_ITERATIONS = 10
TRAIN_ITERATIONS = 1


def main_COMPUTE_ESTIMATES_compute_estimate():
    for size_id in range(len(TRAINING_SET_SIZE)):
        for tr_iter in range(TRAIN_ITERATIONS):
            print "Training set " + str(tr_iter) + ", size " + str(TRAINING_SET_SIZE[size_id])
            TRAINING_SET_PATH = "/home/comete/mromanel/MILES_EXP/EMPIRICAL_ESTIMATES/DATA_FOLDER/" + str(
                TRAINING_SET_SIZE[size_id]) + "_training_and_" + str(
                VALIDATION_SET_SIZE[size_id]) + "_validation_and_" + str(
                TEST_SET_SIZE[size_id]) + "_test_store_folder_train_iteration_" + str(tr_iter) + "/training_set.pkl"

            VALIDATION_SET_PATH = "/home/comete/mromanel/MILES_EXP/EMPIRICAL_ESTIMATES/DATA_FOLDER/" + str(
                TRAINING_SET_SIZE[size_id]) + "_training_and_" + str(
                VALIDATION_SET_SIZE[size_id]) + "_validation_and_" + str(
                TEST_SET_SIZE[size_id]) + "_test_store_folder_train_iteration_" + str(tr_iter) + "/validation_set.pkl"

            G_MAT_COLS_PATH = "/home/comete/mromanel/MILES_EXP/EMPIRICAL_ESTIMATES/G_MAT_FOLDER/" \
                              "g_matrix_cols.pkl"
            G_MAT_PATH = "/home/comete/mromanel/MILES_EXP/EMPIRICAL_ESTIMATES/G_MAT_FOLDER/g_matrix.pkl"
            G_MAT_ROWS_PATH = "/home/comete/mromanel/MILES_EXP/EMPIRICAL_ESTIMATES/G_MAT_FOLDER/" \
                              "g_matrix_rows.pkl"

            G_MAT_COLS = pn.read_pickle(path=G_MAT_COLS_PATH)
            G_MAT = pn.read_pickle(path=G_MAT_PATH)
            G_MAT_ROWS = pn.read_pickle(path=G_MAT_ROWS_PATH)

            tr_set = pn.read_pickle(path=TRAINING_SET_PATH)
            val_set = pn.read_pickle(path=VALIDATION_SET_PATH)

            for ts_iter in range(TEST_ITERATIONS):
                TEST_SET_PATH = "/home/comete/mromanel/MILES_EXP/EMPIRICAL_ESTIMATES/DATA_FOLDER/" + str(
                    TEST_SET_SIZE[size_id]) + "_size_test_sets/" + "test_set_" + str(ts_iter) + ".pkl"
                ts_set = pn.read_pickle(path=TEST_SET_PATH)

                DATA = np.concatenate((tr_set.values, ts_set.values), axis=0)
                print DATA.shape
                DATA = pn.DataFrame(DATA)

                print(
                    empirical_estimate.compute_empirical_estimate(data=DATA, g_mat_cols=G_MAT_COLS, g_mat=G_MAT,
                                                                  g_mat_rows=G_MAT_ROWS))
