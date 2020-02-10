import sys
import time
import numpy as np
import pandas as pn
from tqdm import tqdm
from scipy import stats
from utilities_pckg import utilities
from qif import channel, measure, mechanism, probab

TRAINING_SET_SIZE = [10000, 30000, 50000]
TEST_SET_SIZE = [50000]
VALIDATION_SET_SIZE = [1000, 3000, 5000]
TEST_ITERATIONS = 50
TRAIN_ITERATIONS = 5

BIS_EXP_G_VULN_DP_FOLDER = "/home/comete/mromanel/MILES_EXP/BIS_EXP_G_VULN_DP_FOLDER/"
utilities.createFolder(BIS_EXP_G_VULN_DP_FOLDER)
DATA_FOLDER = BIS_EXP_G_VULN_DP_FOLDER + "DATA_FOLDER/"
utilities.createFolder(DATA_FOLDER)

DATA_FOLDER_TEST = DATA_FOLDER + str(TEST_SET_SIZE[0]) + "_size_test_set/"
utilities.createFolder(DATA_FOLDER_TEST)

G_OBJ = BIS_EXP_G_VULN_DP_FOLDER + "G_OBJ/"
utilities.createFolder(G_OBJ)

#   real counts, replace with those from the real db
#   order: 0, 1, 2, 3, 4 ---> 164  55  36  35  13
real_counts = np.array([164, 55, 36, 35, 13])  # true counts

# real_counts = np.array([40, 55, 36, 35, 13])  # fake counts for safety check

#   prior, from the db
pi = probab.normalize(real_counts)

#   Range of reported counts. Eg. if the real counts are [12, 15, 22, 21, 30] then the actual range is 12..30.
#   The reported range will have 10 extra values on each side, so it will be 2..40
#   (probability drops quickly so we don't need many extra values)
#
extra_values = 20  # how many values outside the range of counts to add
range_size = real_counts.ptp() + 1 + extra_values
range_start = real_counts.min() - int(extra_values / 2)

print(range_start)
print(range_size)
print(range_start + range_size - 1)

#   Geometric mechanism. Matrix rows/columns are 0-based, so the i-th row/column corresponds to value i + range_start
#
epsilon = 1  # smaller => more noise
Geom = mechanism.d_privacy.geometric(range_size, epsilon, first_x=range_start, first_y=range_start)
print(Geom.shape)

G = pn.read_pickle(G_OBJ + 'g_mat.pkl')

(rho, R, a, b) = measure.g_vuln.g_to_bayes(G, pi)
print("a --->" + str(a))
print("b --->" + str(b))

# which diseases are "sensitive"
sensitive = [False, False, False, True, True]


# gain function, higher gain if we guess a "sensitive" disease
def gain(w, x):
    if w != x:
        return 0
    elif sensitive[w]:
        return 2
    else:
        return 1


def create_g_mat():
    g_mat = np.zeros((len(real_counts), len(real_counts)))

    for secret_id in range(len(real_counts)):
        for guess_id in range(len(real_counts)):
            g_mat[secret_id, guess_id] = gain(guess_id, secret_id)

    return g_mat


# Creates a channel with:
#   secret:     the disease of the new user
#   observable: the noisy count of disease d
def create_channel_single_count(d):
    C = np.zeros((real_counts.size, Geom.shape[1]))

    for x in range(real_counts.size):
        # compute the x-th row of C. For this we need to find the real count for disease d,
        # which is real_counts[d], plus one if x == d, minus range_start to convert it to 0-based.
        # Then the x-th row of C is simply the cnt-th row of Geom
        #
        cnt = real_counts[d] + (1 if x == d else 0) - range_start
        C[x, :] = np.array(Geom[cnt, :])

    return C


# Creates a channel with:
#   secret:     the disease of the new user
#   observable: the noisy count of disease d
def create_channel_single_count(d):
    C = np.zeros((real_counts.size, Geom.shape[1]))

    for x in range(real_counts.size):
        # compute the x-th row of C. For this we need to find the real count for disease d,
        # which is real_counts[d], plus one if x == d, minus range_start to convert it to 0-based.
        # Then the x-th row of C is simply the cnt-th row of Geom
        #
        cnt = real_counts[d] + (1 if x == d else 0) - range_start
        C[x, :] = Geom[cnt, :]

    return C


# Creates a channel with:
#   secret:     the disease of the new user
#   observable: counts for all diseases
def create_channel():
    # We can create the channel that outputs the count for each disease, their
    # "parallel" composition is the one that outputs all counst.
    #
    C = create_channel_single_count(0)
    for d in range(1, real_counts.size):
        C = channel.compose.parallel(C, create_channel_single_count(d))
    return C


# def create_new_rndm_state():
#     tm = int(time.time() * (10 ** 6))
#     bs = int(str(int(time.time()))[0:-3]) * (10 ** 9)
#     return np.random.RandomState(seed=tm - bs)


# Black-box execution of C on input x (the real disease). Output is a vector of noisy counts
def execute_C(x):
    noisy_counts = real_counts.copy()
    # print(noisy_counts)
    # print(x)
    noisy_counts[x] = noisy_counts[x] + 1  # Add the disease to the count
    # print("///////////////")
    # print(x)
    # print(real_counts)
    # print(noisy_counts)
    # print("///////////////")

    for d in range(real_counts.size):
        cnt = noisy_counts[d] - range_start  # 0-based count for x
        # geom_cnt_row = np.array(Geom[cnt, :])
        # print(geom_cnt_row)
        noisy_counts[d] = probab.draw(
            Geom[cnt, :]) + range_start  # draw a noisy count from Geom's cnt-th row, map back to the real range

        # rndmstt = create_new_rndm_state()
        # obs = np.arange(start=range_start, stop=range_start + range_size, step=1)
        # # print(len(obs))
        # # print(len(Geom[cnt, :]))
        # sample_distr = stats.rv_discrete(name='draw', values=(obs, Geom[cnt, :]), seed=rndmstt)
        #
        # noisy_counts[d] = sample_distr.rvs(size=1, random_state=rndmstt)

    return noisy_counts


def draw_w_y_blackbox(R, rho):
    w, x = channel.draw(R, rho)  # draw (w,x) from rho/R
    return [execute_C(x=x), x, w]  # then execute C with x


def create_single_dataset(len_rows, R, rho):
    list_targets = []
    list_features = []
    list_guess = []

    for i in range(len_rows):
        ex_c, x, w = draw_w_y_blackbox(R, rho)
        list_targets.append(x)
        list_features.append(ex_c)
        list_guess.append(w)

    array_features = np.array(list_features).reshape((len_rows, len(real_counts)))
    array_targets = np.array(list_targets).reshape((len_rows, 1))
    array_guess = np.array(list_guess).reshape((len_rows, 1))

    dataset = np.column_stack((array_features, array_targets, array_guess))

    return dataset


# Same as create_channel_single_count, but approximate the channel using
# a different range for each d (essentially removing elements with negligible probability)
#
def create_channel_single_count_approx(d):
    C = np.zeros((real_counts.size, 1 + 2 * extra_values))

    # we only need two rows of the geometric, for cnt and cnt+1. The range is [cnt-extra_value .. cnt+extra_values]
    cnt = real_counts[d]
    Geom = mechanism.d_privacy.geometric(n_rows=2, n_cols=1 + 2 * extra_values, epsilon=epsilon, first_x=cnt,
                                         first_y=cnt - extra_values)

    for x in range(real_counts.size):
        C[x, :] = Geom[0 if x != d else 1, :]

    return C


def create_channel_approx():
    C = create_channel_single_count_approx(0)
    for d in range(1, real_counts.size):
        C = channel.compose.parallel(C, create_channel_single_count_approx(d))
    return C


def main_BIS_EXP_G_VULN_DP_create_channel_and_g_mat_and_data():
    # create_single_dataset(10)
    #
    if len(VALIDATION_SET_SIZE) != len(TRAINING_SET_SIZE):
        sys.exit("ERROR! Different size lists' lengths.")

    for size_list_iterator in range(len(TRAINING_SET_SIZE)):
        training_set_size = TRAINING_SET_SIZE[size_list_iterator]
        validation_set_size = VALIDATION_SET_SIZE[size_list_iterator]

        for train_iteration in tqdm(range(TRAIN_ITERATIONS)):
            training_and_validation_and_test_set_store_folder = DATA_FOLDER + str(
                training_set_size) + "_training_and_" + str(
                validation_set_size) + "_validation_store_folder_train_iteration_" + str(train_iteration) + "/"
            utilities.createFolder(path=training_and_validation_and_test_set_store_folder)

            tr = create_single_dataset(training_set_size, R=R, rho=rho)
            val = create_single_dataset(validation_set_size, R=R, rho=rho)

            pn.to_pickle(obj=tr, path=training_and_validation_and_test_set_store_folder + "training_set.pkl", protocol=2)
            pn.to_pickle(obj=val, path=training_and_validation_and_test_set_store_folder + "validation_set.pkl", protocol=2)

    print("computing the channel")

    # C = create_channel()
    # print("shape", C.shape)
    # print("computing Vg")
    # print("Vg[pi,C]:", measure.g_vuln.posterior(gain, pi, C))

    print("computing the channel approx")
    Ca = create_channel_approx();
    print("computing Vg")
    print("Vg[pi,Ca]:", measure.g_vuln.posterior(gain, pi, Ca))
