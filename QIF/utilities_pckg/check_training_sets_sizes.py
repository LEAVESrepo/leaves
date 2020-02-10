import pandas as pn

DATA_FOLDER = "/home/comete/mromanel/MILES_EXP/EXP_G_VULN_DP_FOLDER/DATA_FOLDER_AFTER_OUR_PREPROCESSING/"
TRAIN_ITER = 5
TRAINING_SIZES = [10000, 30000, 50000]
VAL_SIZES = [1000, 3000, 5000]
extra = "and_50000_test"


def main_check_training_sets_sizes():
    for tr_size_id in range(len(TRAINING_SIZES)):
        for train_iter in range(TRAIN_ITER):
            mid_folder = str(TRAINING_SIZES[tr_size_id]) + "_training_and_" + str(
                VAL_SIZES[tr_size_id]) + "_validation_" + extra + "_store_folder_train_iteration_" + str(
                train_iter) + "/"
            tr = pn.read_pickle(path=DATA_FOLDER + mid_folder + "training_set.pkl")
            if isinstance(tr, pn.DataFrame):
                tr = tr.values
            print("\n\n\nOriginal sample size: " + str(TRAINING_SIZES[tr_size_id]))
            print("Sample size: " + str(tr.shape[0]))
            print("#############################")
