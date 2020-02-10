import sys
import numpy as np
import pandas as pn
from tqdm import tqdm
from utilities_pckg import utilities
from qif import channel, measure, probab, metric, point, mechanism, lp
import math
from tabulate import tabulate

tqdm.monitor_interval = 0

TRAINING_SET_SIZE = [100, 1000, 10000, 30000, 50000]  # [90000, 270000, 450000]
# TEST_SET_SIZE = [90000]  # [90000, 270000, 450000]
VALIDATION_SET_SIZE = [10, 100, 1000, 3000, 5000]  # [9000, 27000, 45000]
TEST_ITERATIONS = 50
TRAIN_ITERATIONS = 5

BIS_EXP_GEO_LOCATION_FOLDER = "/home/comete/mromanel/MILES_EXP/BIS_EXP_GEO_LOCATION_QIF_LIB_SETTING/"
utilities.createFolder(BIS_EXP_GEO_LOCATION_FOLDER)

WIDTH = 20  # in cells
HEIGHT = 20  # in cells
CELL_SIZE = 250.  # in length units (meters)
EUCLID = euclid = metric.euclidean(point)  # Euclidean distance on qif.point
MAX_GAIN = 4
ALPHA = 0.95

DATA_FOLDER = BIS_EXP_GEO_LOCATION_FOLDER + "DATA_FOLDER/"
utilities.createFolder(DATA_FOLDER)
G_MAT_PATH = BIS_EXP_GEO_LOCATION_FOLDER + "G_OBJ/g_mat.pkl"

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


def main_BIS_EXP_GEO_LOCATION_create_channel_and_data():
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
    pi_dic = pn.read_pickle(
        path="/home/comete/mromanel/MILES_EXP/BIS_EXP_GEO_LOCATION_QIF_LIB_SETTING/file_prior_distr.pkl")
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

    G = pn.read_pickle(G_MAT_PATH)

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

    print(euclid_cell(13, 20))
    print(euclid_cell(20, 13))

    # solve
    C = mechanism.g_vuln.min_loss_given_max_vuln(pi, n_outputs, n_guesses, max_vuln, gain, loss, hard_max_loss)

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
