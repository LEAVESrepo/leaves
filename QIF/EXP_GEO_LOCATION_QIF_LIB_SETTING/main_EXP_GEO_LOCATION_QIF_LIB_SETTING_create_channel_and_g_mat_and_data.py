import sys
import copy
import math
import numpy as np
import pandas as pn
from tqdm import tqdm
from tabulate import tabulate
from utilities_pckg import utilities
from qif import lp, mechanism, metric, point, probab, measure, utility
import inspect
from utilities_pckg.runtime_error_handler import runtime_error_handler as err_hndl

np.set_printoptions(threshold=100000000)

CREATE_TEST_SET = True

TRAINING_SET_SIZE = [100, 1000, 10000, 30000, 50000]
TEST_SET_SIZE = [50000]
VALIDATION_SET_SIZE = [10, 100, 1000, 3000, 5000]
TEST_ITERATIONS = 50
TRAIN_ITERATIONS = 5

WIDTH = 20  # in cells
HEIGHT = 20  # in cells
CELL_SIZE = 250.  # in length units (meters)
EUCLID = euclid = metric.euclidean(point)  # Euclidean distance on qif.point
MAX_GAIN = 4
ALPHA = 0.95

DATA_FOLDER = "/home/comete/mromanel/MILES_EXP/EXP_GEO_LOCATION_QIF_LIB_SETTING/DATA_FOLDER/"
utilities.createFolder(path=DATA_FOLDER)

CHANNEL_PATH = "/home/comete/mromanel/MILES_EXP/EXP_GEO_LOCATION_QIF_LIB_SETTING/channel.pkl"

G_OBJ_PATH = "/home/comete/mromanel/MILES_EXP/EXP_GEO_LOCATION_QIF_LIB_SETTING/G_OBJ/"
utilities.createFolder(path=G_OBJ_PATH)
G_MAT_PATH = G_OBJ_PATH + "g_mat.pkl"
G_MAT_ROWS_PATH = G_OBJ_PATH + "g_mat_rows.pkl"
G_MAT_COLS_PATH = G_OBJ_PATH + "g_mat_cols.pkl"

#   set solver
lp.defaults.solver = "GLOP"


# euclidean distance on cell ids
def euclid_cell(a, b):
    return CELL_SIZE * euclid(
        point.from_cell(a, WIDTH),  # maps cell-id to point in a grid
        point.from_cell(b, WIDTH)  # of cell size 1 and given 'width'
    )


#   perform checks on the neighbors of a considered cell printing the gain scores
def sanity_checks(considered_cell):
    print("\n\n\n")
    cont = 0
    for i in range(WIDTH ** 2):
        g = gain(considered_cell, i)
        if g != 0:
            print("Cell " + str(considered_cell) + " and " + str(i))
            print("gain:", g)
            cont += 1
            print("----------\n")
    print("Neighboring cells with gain > 0: " + str(cont))


#   function that defines the gain inside the gain function
def f(dist):
    return round(MAX_GAIN * math.exp(- ALPHA * dist / CELL_SIZE))


# gain function. w and x are cells. It recalls teh f function above
def gain(w, x):
    return f(euclid_cell(w, x))


#   create the gain matrix by recalling the gain function above
def create_gain_matrix():
    g_mat = np.zeros((HEIGHT ** 2, WIDTH ** 2))
    for i_ter in range(HEIGHT ** 2):
        for j_ter in range(WIDTH ** 2):
            g_mat[i_ter, j_ter] = gain(j_ter, i_ter)

    return g_mat


def create_gain_matrix2():
    g_mat = np.zeros((HEIGHT ** 2, WIDTH ** 2))
    for i_ter in range(HEIGHT ** 2):
        for j_ter in range(WIDTH ** 2):
            g_mat[i_ter, j_ter] = gain(i_ter, j_ter)

    return g_mat


def create_single_dataset(size, C, pi):
    samples_list = []
    for i in range(size):
        x = probab.draw(pi)
        y = execute_C(x, C)
        samples_list.append([y, x])

    return np.array(samples_list).reshape((size, len(samples_list[0])))


# Draws (x,y) samples from pi/C. For the sample-preprocessing method
def draw_x_y(pi):
    # pi_ = np.array(pi)
    x = probab.draw(pi)
    return x, execute_C(x)


def execute_C(x, C):  # we only have black box access to C. This function runs C under secret x and returns an output y
    return probab.draw(C[x, :])


def main_EXP_GEO_LOCATION_QIF_LIB_SETTING_create_channel_and_g_mat_and_data():
    # grid

    # diagonal of the grid
    diag = euclid(point(0, 0), point(CELL_SIZE * WIDTH, CELL_SIZE * HEIGHT))

    # loss function, just euclidean distance
    loss = euclid_cell

    # some sanity checks
    sanity_checks(considered_cell=132)

    max_vuln = f(CELL_SIZE)  # maximum allowed posterior g-vulnerability
    hard_max_loss = 2 * CELL_SIZE  # loss(x,y) > hard_max_loss => C[x,y] = 0
    n_secrets = n_outputs = n_guesses = WIDTH * HEIGHT
    pi_dic = pn.read_pickle(path="/home/comete/mromanel/MILES_EXP/EXP_GEO_LOCATION_QIF_LIB_SETTING/file_prior_distr.pkl")
    # print("\n\n\npi dictionary ---> pi[cell]:cell_probability")
    print(pi_dic)

    pi_mat = np.zeros((WIDTH, HEIGHT))
    for i_ter in range(WIDTH):
        for j_ter in range(HEIGHT):
            cell_id = WIDTH * i_ter + j_ter
            pi_mat[i_ter, j_ter] = pi_dic[cell_id]
    print("\n\n\n Table for pi where up is south, down is north, left is west, right is east.")
    headers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"]
    table_pi_mat = tabulate(pi_mat, headers, tablefmt="fancy_grid")
    print(table_pi_mat)

    # pi_mat_map = np.flip(pi_mat, 0)
    # print("\n\n\n")
    # print("\n\n\n Table for pi where up is north, down is south, left is west, right is east.")
    # table_pi_mat_map = tabulate(pi_mat_map, headers, tablefmt="fancy_grid")
    # print(table_pi_mat_map)

    pi = pi_mat.flatten()  # probab.uniform(n_secrets)  # uniform prior
    # print("\n\n\npi ---> such that pi[i] = prob_cell[i]")
    print(pi)

    ############################

    list_of_cells_probs = []
    for id_cell_ind in range(len(pi_dic)):
        list_of_cells_probs.append(pi_dic[id_cell_ind])

    # sanity check

    for i in range(len(pi)):
        if pi[i] != list_of_cells_probs[i]:
            sys.exit("ERROR in prior")

    ############################

    gmat1 = create_gain_matrix()
    gmat2 = create_gain_matrix2()

    for i in range(gmat1.shape[0]):
        for j in range(gmat1.shape[1]):
            if gmat1[i, j] != gmat2[i, j]:
                sys.exit("BAZINGAAAAAA")

    print(euclid_cell(13, 20))
    print(euclid_cell(20, 13))

    # solve
    C = mechanism.g_vuln.min_loss_given_max_vuln(pi, n_outputs, n_guesses, max_vuln, gain, loss, hard_max_loss)
    # print("\n\nC:\n", C)
    # print("\n\nmax_vuln:", max_vuln)
    # print("\n\nVg(pi, C)", measure.g_vuln.posterior(gain, pi, C))
    # print("\n\nUtility C:", utility.expected_distance(loss, pi, C))
    # print("-----------------\n")
    # """
    # # Inverse problem
    # max_loss = 300
    # C = mechanism.g_vuln.min_vuln_given_max_loss(pi, n_outputs, n_guesses, max_loss, gain, loss, hard_max_loss)
    # print("C:\n", C)
    # print("max_vuln:", max_vuln)
    # print("Vg(pi, C)", measure.g_vuln.posterior(gain, pi, C))
    # print("Utility:", utility.expected_distance(loss, pi, C))
    # """
    #
    # C_copy = copy.deepcopy(C)
    # for i_ter in tqdm(range(C.shape[0])):
    #     prob_observables_given_secret = C_copy[i_ter, :]
    #     # print(np.sum(prob_observables_given_secret))
    #     prob_observables_given_secret_norm = tuple(
    #         p / sum(prob_observables_given_secret) for p in prob_observables_given_secret)
    #     C_copy[i_ter, :] = prob_observables_given_secret_norm
    #
    # print("\n\nVg(pi, C_copy)", measure.g_vuln.posterior(gain, pi, C_copy))
    # print("\n\nUtility C_copy:", utility.expected_distance(loss, pi, C_copy))
    # print("-----------------\n")
    #
    # for i in range(C_copy.shape[0]):
    #     print(sum(C_copy[i, :]))
    #
    # for r in range(10):
    #     print("\n\n\n###########################################\n\n\n")
    #
    # C_copy_transposed = np.transpose(C_copy)
    # for j_ter in range(C_copy_transposed.shape[1]):
    #     sum_ = sum(C_copy_transposed[:, j_ter])
    #     print(sum_)
    # pn.to_pickle(obj=C_copy_transposed, path=CHANNEL_PATH, protocol=2)
    #
    # g_mat = create_gain_matrix()
    # g_mat_rows = np.arange(start=0, stop=HEIGHT ** 2, step=1)
    # g_mat_cols = np.arange(start=0, stop=WIDTH ** 2, step=1)
    # pn.to_pickle(obj=g_mat, path=G_MAT_PATH, protocol=2)
    # pn.to_pickle(obj=g_mat_rows, path=G_MAT_ROWS_PATH, protocol=2)
    # pn.to_pickle(obj=g_mat_cols, path=G_MAT_COLS_PATH, protocol=2)

    if len(VALIDATION_SET_SIZE) != len(TRAINING_SET_SIZE):
        err_hndl(str_="array_sizes_not_matching", add=inspect.stack()[0][3])

    for size_list_iterator in range(len(TRAINING_SET_SIZE)):
        training_set_size = TRAINING_SET_SIZE[size_list_iterator]
        validation_set_size = VALIDATION_SET_SIZE[size_list_iterator]

        for train_iteration in range(TRAIN_ITERATIONS):
            training_set_mat = create_single_dataset(size=training_set_size, C=C, pi=pi)

            training_and_validation_and_test_set_store_folder = DATA_FOLDER + str(
                training_set_size) + "_training_and_" + str(
                validation_set_size) + "_validation_store_folder_train_iteration_" + str(train_iteration) + "/"

            utilities.createFolder(path=training_and_validation_and_test_set_store_folder)

            training_df = pn.DataFrame(data=training_set_mat, columns=["O_train", "S_train"])
            pn.to_pickle(obj=training_df.values,
                         path=training_and_validation_and_test_set_store_folder + "/training_set.pkl",
                         protocol=2)

            ################################################################################################################

            validation_set_mat = create_single_dataset(size=validation_set_size, C=C, pi=pi)

            validation_df = pn.DataFrame(data=validation_set_mat, columns=["O_val", "S_val"])
            pn.to_pickle(obj=validation_df.values,
                         path=training_and_validation_and_test_set_store_folder + "/validation_set.pkl",
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
            test_set_mat = create_single_dataset(size=test_set_size, C=C, pi=pi)
            list_unq.append(len(np.unique(test_set_mat[:, 0])))

            test_set_store_folder = DATA_FOLDER + str(test_set_size) + "_size_test_sets/"
            utilities.createFolder(path=test_set_store_folder)

            test_df = pn.DataFrame(data=test_set_mat, columns=["O_test", "S_test"])
            pn.to_pickle(obj=test_df.values,
                         path=test_set_store_folder + "/test_set_" + str(test_iteration) + ".pkl", protocol=2)

        print(list_unq)
