import pandas as pn
from utilities_pckg import empirical_estimate

DATA_PATH = "/home/comete/mromanel/MILES_EXP/EMPIRICAL_ESTIMATES/DATA_FOLDER/" \
            "5000_training_and_500_validation_and_5000_test_store_folder_train_iteration_0/training_set.pkl"
G_MAT_COLS_PATH = "/home/comete/mromanel/MILES_EXP/EMPIRICAL_ESTIMATES/G_MAT_FOLDER/" \
                  "g_matrix_cols.pkl"
G_MAT_PATH = "/home/comete/mromanel/MILES_EXP/EMPIRICAL_ESTIMATES/G_MAT_FOLDER/g_matrix.pkl"
G_MAT_ROWS_PATH = "/home/comete/mromanel/MILES_EXP/EMPIRICAL_ESTIMATES/G_MAT_FOLDER/" \
                  "g_matrix_rows.pkl"


def main_COMPUTE_ESTIMATES_compute_estimate():
    DATA = pn.read_pickle(path=DATA_PATH)
    G_MAT_COLS = pn.read_pickle(path=G_MAT_COLS_PATH)
    G_MAT = pn.read_pickle(path=G_MAT_PATH)
    G_MAT_ROWS = pn.read_pickle(path=G_MAT_ROWS_PATH)

    print(
        empirical_estimate.compute_empirical_estimate(data=DATA, g_mat_cols=G_MAT_COLS, g_mat=G_MAT, g_mat_rows=G_MAT_ROWS))
