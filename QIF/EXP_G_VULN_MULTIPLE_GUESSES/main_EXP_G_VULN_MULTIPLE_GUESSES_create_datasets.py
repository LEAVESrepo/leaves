from utilities_pckg import utilities
from utilities_pckg.runtime_error_handler import runtime_error_handler as err_hndl
import pandas as pn
import inspect
import numpy as np
from qif import channel, measure, probab

EXP_G_VULN_MULTIPLE_GUESSES_FOLDER_PATH = "/home/comete/mromanel/MILES_EXP/EXP_G_VULN_MULTIPLE_GUESSES/"
CHANNEL_PATH = EXP_G_VULN_MULTIPLE_GUESSES_FOLDER_PATH + "channel_df_norm.pkl"

DATA_FOLDER = EXP_G_VULN_MULTIPLE_GUESSES_FOLDER_PATH + "DATA_FOLDER/"
utilities.createFolder(DATA_FOLDER)

G_MAT_FOLDER = "/home/comete/mromanel/MILES_EXP/EXP_G_VULN_MULTIPLE_GUESSES/G_MAT_FOLDER/"
gain = pn.read_pickle(path=G_MAT_FOLDER + "g_matrix_10_secrets_2_guesses.pkl")

TRAINING_SET_SIZE = [10000, 30000, 50000]
TEST_SET_SIZE = [50000]
VALIDATION_SET_SIZE = [1000, 3000, 5000]
TEST_ITERATIONS = 50
TRAIN_ITERATIONS = 5

#   pi distribution
n = 10
pi = probab.uniform(n)
print("Secrets' distribution: ", pi)

CREATE_TEST_SET = True


def create_single_dataset(size, C):
    samples_list = []
    for i in range(size):
        x = probab.draw(pi)
        y = execute_C(x, C)
        samples_list.append([y, x])

    return np.array(samples_list).reshape((size, len(samples_list[0])))


# Draws (x,y) samples from pi/C. For the sample-preprocessing method
def draw_x_y():
    # pi_ = np.array(pi)
    x = probab.draw(pi)
    return x, execute_C(x)


def execute_C(x, C):  # we only have black box access to C. This function runs C under secret x and returns an output y
    return probab.draw(C[x, :])


def main_EXP_G_VULN_MULTIPLE_GUESSES_create_data():
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  geometric distribution loading  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    print("\n####################################################################################")
    print("#########################  geometric distribution loading  #########################")
    print("####################################################################################\n")

    utilities.createFolder(DATA_FOLDER)

    channel_matrix_df = pn.read_pickle(path=CHANNEL_PATH)

    #   sanity check
    for i in range(len(channel_matrix_df.columns.values) - 1):
        if channel_matrix_df.index.values[i + 1] <= channel_matrix_df.index.values[i]:
            import sys
            sys.exit("BAD CHANNEL FORMAT: cols")
    for i in range(len(channel_matrix_df.index.values) - 1):
        if channel_matrix_df.index.values[i + 1] <= channel_matrix_df.index.values[i]:
            import sys
            sys.exit("BAD CHANNEL FORMAT: rows")

    channel_matrix = np.transpose(channel_matrix_df.values)
    print(channel_matrix.shape)

    print("Vg(pi, C)", measure.g_vuln.posterior(gain, pi, C=channel_matrix))

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  create training sets  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    #   X are the observables and y are the secrets (respectively col 0 and 1), stratify wrt to secret
    #   split training and test data
    # mt = create_single_dataset(size=50000, C=channel_matrix)
    # print(mt)
    # print(len(np.unique(mt[:, 0])))

    if len(VALIDATION_SET_SIZE) != len(TRAINING_SET_SIZE):
        err_hndl(str_="array_sizes_not_matching", add=inspect.stack()[0][3])

    for size_list_iterator in range(len(TRAINING_SET_SIZE)):
        training_set_size = TRAINING_SET_SIZE[size_list_iterator]
        validation_set_size = VALIDATION_SET_SIZE[size_list_iterator]

        for train_iteration in range(TRAIN_ITERATIONS):
            training_set_mat = create_single_dataset(size=training_set_size, C=channel_matrix)

            training_and_validation_and_test_set_store_folder = DATA_FOLDER + str(
                training_set_size) + "_training_and_" + str(
                validation_set_size) + "_validation_store_folder_train_iteration_" + str(train_iteration) + "/"

            utilities.createFolder(path=training_and_validation_and_test_set_store_folder)

            training_df = pn.DataFrame(data=training_set_mat, columns=["O_train", "S_train"])
            pn.to_pickle(obj=training_df.values, path=training_and_validation_and_test_set_store_folder + "/training_set.pkl",
                         protocol=2)

            ################################################################################################################

            validation_set_mat = create_single_dataset(size=validation_set_size, C=channel_matrix)

            validation_df = pn.DataFrame(data=validation_set_mat, columns=["O_val", "S_val"])
            pn.to_pickle(obj=validation_df.values, path=training_and_validation_and_test_set_store_folder + "/validation_set.pkl",
                         protocol=2)

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  create test sets  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if CREATE_TEST_SET:
        list_unq = []
        print("\n####################################################################################")
        print("#################################  create test sets  ################################")
        print("####################################################################################\n")
        test_set_size = TEST_SET_SIZE[0]
        for test_iteration in range(TEST_ITERATIONS):
            test_set_mat = create_single_dataset(size=test_set_size, C=channel_matrix)
            list_unq.append(len(np.unique(test_set_mat[:, 0])))

            test_set_store_folder = DATA_FOLDER + str(test_set_size) + "_size_test_sets/"
            utilities.createFolder(path=test_set_store_folder)

            test_df = pn.DataFrame(data=test_set_mat, columns=["O_test", "S_test"])
            pn.to_pickle(obj=test_df.values,
                         path=test_set_store_folder + "/test_set_" + str(test_iteration) + ".pkl", protocol=2)

        print(list_unq)
