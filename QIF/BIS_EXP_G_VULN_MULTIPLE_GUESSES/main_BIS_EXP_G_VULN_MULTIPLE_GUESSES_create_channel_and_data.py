import sys
import numpy as np
import pandas as pn
from tqdm import tqdm
from utilities_pckg import utilities
from qif import channel, measure, probab

tqdm.monitor_interval = 0

TRAINING_SET_SIZE = [10000, 30000, 50000]  # [90000, 270000, 450000]
TEST_SET_SIZE = [90000]  # [90000, 270000, 450000]
VALIDATION_SET_SIZE = [1000, 3000, 5000]  # [9000, 27000, 45000]
TEST_ITERATIONS = 50
TRAIN_ITERATIONS = 5

BIS_EXP_G_VULN_MULTIPLE_GUESSES_FOLDER = "/home/comete/mromanel/MILES_EXP/BIS_EXP_G_VULN_MULTIPLE_GUESSES/"
utilities.createFolder(BIS_EXP_G_VULN_MULTIPLE_GUESSES_FOLDER)

CHANNEL_FILE = BIS_EXP_G_VULN_MULTIPLE_GUESSES_FOLDER + "channel_df_norm.pkl"
G_MAT_FILE = BIS_EXP_G_VULN_MULTIPLE_GUESSES_FOLDER + "G_MAT_FOLDER/g_matrix_10_secrets_2_guesses.pkl"

DATA_FOLDER = BIS_EXP_G_VULN_MULTIPLE_GUESSES_FOLDER + "DATA_FOLDER/"
utilities.createFolder(DATA_FOLDER)


##### draw from rho/RC, black box
def execute_C(x, C):  # we only have black box access to C. This function runs C under secret x and returns an output y
    # C_x = np.array(C[x, :])
    return probab.draw(C[x, :])


def draw_w_y_blackbox(R, rho, C):
    w, x = channel.draw(R, rho)  # draw (w,x) from rho/R
    return [execute_C(x=x, C=C), x, w]  # then execute C with x


def create_single_dataset(size, R, rho, C):
    list_ = []
    for i_ter in range(size):
        list_.append(draw_w_y_blackbox(R=R, rho=rho, C=C))
    # return pn.DataFrame(np.array(list_).reshape((size, len(list_[0]))), columns=["OBS", "SECR", "GUESS"])
    return np.array(list_).reshape((size, len(list_[0])))


def main_BIS_EXP_G_VULN_MULTIPLE_GUESSES_create_channel_and_data():
    #   pi distribution
    n = 10
    pi = probab.uniform(n)
    print(pi.shape)

    #   g matrix
    G = pn.read_pickle(path=G_MAT_FILE)
    print(G.shape)

    #   channel matrix
    C = pn.read_pickle(path=CHANNEL_FILE).values
    C = np.transpose(C)
    print(C.shape)

    # get rho, R, a, b
    (rho, R, a, b) = measure.g_vuln.g_to_bayes(G, pi)
    print("a --->" + str(a))
    print("b --->" + str(b))

    # for any C we have Vg[pi, C] = a * V[rho, RC] + b

    print("    Vg[pi, C]:     ", measure.g_vuln.posterior(G, pi, C))
    print("a * V[rho, RC] + b:", a * measure.bayes_vuln.posterior(rho, R.dot(C)) + b)

    print(len(np.unique(create_single_dataset(size=10000, R=R, rho=rho, C=C)[:, 0])))

    # so we can estimate Vg in a black-box matter by generating samples according to rho and RC !

    if len(VALIDATION_SET_SIZE) != len(TRAINING_SET_SIZE):
        sys.exit("ERROR! Different size lists' lengths.")

    for size_list_iterator in range(len(TRAINING_SET_SIZE)):
        training_set_size = TRAINING_SET_SIZE[size_list_iterator]
        validation_set_size = VALIDATION_SET_SIZE[size_list_iterator]
        # test_set_size = TEST_SET_SIZE[size_list_iterator]

        for train_iteration in range(TRAIN_ITERATIONS):
            training_and_validation_and_test_set_store_folder = DATA_FOLDER + str(
                training_set_size) + "_training_and_" + str(
                validation_set_size) + "_validation_store_folder_train_iteration_" + str(train_iteration) + "/"
            utilities.createFolder(path=training_and_validation_and_test_set_store_folder)

            tr = create_single_dataset(size=training_set_size, R=R, rho=rho, C=C)
            val = create_single_dataset(size=validation_set_size, R=R, rho=rho, C=C)

            pn.to_pickle(obj=tr, path=training_and_validation_and_test_set_store_folder + "training_set.pkl", protocol=2)
            pn.to_pickle(obj=val, path=training_and_validation_and_test_set_store_folder + "validation_set.pkl", protocol=2)

            print("\n\n\nSize " +
                  str(TRAINING_SET_SIZE[size_list_iterator]) +
                  ", train iteration " +
                  str(train_iteration))

        # test_set_store_folder = training_and_validation_and_test_set_store_folder + str(
        #     test_set_size) + "_size_test_sets/"
        # utilities.createFolder(path=test_set_store_folder)
        # for test_iteration in tqdm(range(TEST_ITERATIONS)):
        #     ts = create_single_dataset(size=test_set_size, R=R, rho=rho, C=C)
        #     pn.to_pickle(obj=ts, path=test_set_store_folder + "test_set_" + str(test_iteration) + ".pkl", protocol=2)
