import itertools
import numpy as np
import pandas as pn
from fractions import Fraction
from utilities_pckg import utilities
import sys
from tqdm import tqdm
import os
import pickle

tqdm.monitor_interval = 0


def compute_K_from_g_func(g_func):
    print "Starting denominators computation"
    list_denom = []
    for raw_iter in tqdm(range(g_func.shape[0])):
        for col_iter in range(g_func.shape[1]):
            list_denom.append(Fraction(g_func[raw_iter, col_iter]).limit_denominator().denominator)

    print "Starting K computation"
    K = utilities.compute_lcm_for_list_of_numbers(list_of_numbers=list_denom)
    print "Ended K computation"

    return K


#   given a number of guesses and a list of unique secret symbols, it creates a matrix with a column for each secret and a
#   row for each possible combination of a number of secret equal to the number of guesses: secrets{a, b, c, d} and
#   n_guesses = 3 will give rows {a, b, c}, {a, b, d}, {a, c, d}, {b, c, d}. It is just a matrix with elements
#   0 or 1, so K=1
def create_g_function_matrix_n_guesses(list_unique_secrets, n_guesses, save_g_path):
    if n_guesses > len(list_unique_secrets):
        sys.exit("ERROR! The chosen number of guesses if greater than the number of possible secrets.")
    all_possible_guesses = set(list(itertools.combinations(list_unique_secrets, n_guesses)))

    all_possible_guesses_dic_idx = 0
    all_possible_guesses_dic = {}
    for guess in all_possible_guesses:
        all_possible_guesses_dic[all_possible_guesses_dic_idx] = guess
        all_possible_guesses_dic_idx += 1
    # print "all_possible_guesses_dic ---> ", all_possible_guesses_dic

    col_names = list_unique_secrets
    row_names = []
    for key in all_possible_guesses_dic:
        row_names.append(key)
    row_names = np.unique(row_names)
    # print "col_names ---> ", col_names
    # print "row_names ---> ", row_names

    g_matrix = np.zeros((len(row_names), len(col_names)))

    for row_iter in tqdm(range(len(row_names))):
        for col_iter in range(len(col_names)):
            if col_names[col_iter] in all_possible_guesses_dic[row_names[row_iter]]:
                g_matrix[row_iter, col_iter] = 1
            else:
                continue

    K = 1  # ]compute_K_from_g_func(g_func=g_matrix)

    # print "g_matrix ---> ", g_matrix
    # print "K ---> ", K

    if save_g_path is not None:
        g_matrix_path = save_g_path + "/g_matrix_" + str(len(list_unique_secrets)) + "_secrets_" + str(
            n_guesses) + "_guesses.pkl"
        g_matrix_rows = save_g_path + "/g_matrix_" + str(len(list_unique_secrets)) + "_secrets_" + str(
            n_guesses) + "_guesses_rows.pkl"
        g_matrix_cols = save_g_path + "/g_matrix_" + str(len(list_unique_secrets)) + "_secrets_" + str(
            n_guesses) + "_guesses_cols.pkl"
        g_matrix_all_possible_guesses_dic = save_g_path + "/g_matrix_" + str(len(list_unique_secrets)) + "_secrets_" + str(
            n_guesses) + "_guesses_all_possible_guesses_dic.pkl"
        g_matrix_K = save_g_path + "/g_matrix_" + str(len(list_unique_secrets)) + "_secrets_" + str(
            n_guesses) + "_guesses_K.pkl"

        print "\n Saving g_mat..."

        if not os.path.isfile(g_matrix_path):
            pickle.dump(g_matrix, open(g_matrix_path, 'w'))
        if not os.path.isfile(g_matrix_rows):
            pickle.dump(row_names, open(g_matrix_rows, 'w'))
        if not os.path.isfile(g_matrix_cols):
            pickle.dump(col_names, open(g_matrix_cols, 'w'))
        if not os.path.isfile(g_matrix_all_possible_guesses_dic):
            pickle.dump(all_possible_guesses_dic, open(g_matrix_all_possible_guesses_dic, 'w'))
        if not os.path.isfile(g_matrix_K):
            pickle.dump(K, open(g_matrix_K, 'w'))

    """print "g_matrix", g_matrix
    print "row_names", row_names
    print "col_names", col_names
    print "all_possible_guesses_dic", all_possible_guesses_dic
    print "K = ", K"""

    return [g_matrix, row_names, col_names, all_possible_guesses_dic, K]


def create_D_prime(D, colnames, g_matrix, g_col_names, g_row_names, K):
    print "Launch function to create dataset"
    if isinstance(D, pn.DataFrame):
        D = D.values

    """D = np.array([[1, 1], [1, 2], [1, 2]])
    g_matrix = np.array([[5, 1], [1, 2]])
    g_col_names = np.array([1, 2])
    g_row_names = np.array([1, 2])
    K = 1"""

    observables = D[:, 0]
    secrets = D[:, 1]

    y_list = []
    x_list = []
    z_list = []
    z_hot_list = []

    observables_unq = np.unique(observables)
    secrets_unq = np.unique(secrets)

    print "Starting dataset creation"
    for x_iter in tqdm(range(len(secrets_unq))):
        x = secrets_unq[x_iter]
        x_idx = np.where(secrets == x)[0]

        y_match, count_y_match = np.unique(observables[x_idx], return_counts=True)
        # print y_match, count_y_match

        for y_iter in range(len(y_match)):
            y = y_match[y_iter]
            m_xy = count_y_match[y_iter]

            g_allz_x = g_matrix[:, np.where(g_col_names == x)[0][0]]
            # print g_allz_x
            m_xy_allz = m_xy * g_allz_x * K
            # print m_xy_allz

            for iter_z in range(len(m_xy_allz)):
                if m_xy_allz[iter_z] == 0:
                    continue
                z = g_row_names[iter_z]
                z_hot = g_matrix[iter_z, :]
                """y_list_tmp = [y] * int(m_xy_allz[iter_z])
                y_list.extend(y_list_tmp)
                x_list_tmp = [x] * int(m_xy_allz[iter_z])
                x_list.extend(x_list_tmp)
                z_list_tmp = [z] * int(m_xy_allz[iter_z])
                z_list.extend(z_list_tmp)"""
                for i in range(int(m_xy_allz[iter_z])):
                    y_list.append(y)
                    x_list.append(x)
                    z_list.append(z)
                    z_hot_list.append(z_hot)

    res = np.column_stack((np.array(y_list), np.array(x_list)))
    res = np.column_stack((np.array(res), np.array(z_list)))

    print "Ended dataset creation"

    return [pn.DataFrame(data=res, columns=colnames), np.array(z_hot_list).reshape((res.shape[0], g_matrix.shape[1]))]


def create_D_prime_multidimensional_inputs(D, colnames, g_matrix, g_col_names, g_row_names, K):
    print "Launch function to create dataset"
    if isinstance(D, pn.DataFrame):
        D = D.values

    observables = D[:, 0:D.shape[1]-1]
    # print(observables)
    secrets = D[:, -1]
    # print(secrets)

    y_list = []
    x_list = []
    z_list = []

    secrets_unq = np.unique(secrets)
    # print secrets_unq

    print "Starting dataset creation"

    for x_iter in tqdm(range(len(secrets_unq))):
        x = secrets_unq[x_iter]
        x_idx = np.where(secrets == x)[0]
        considered_obs = observables[x_idx, :]
        # print "\n\n\n"
        # print considered_obs
        
        dt = np.dtype((np.void, considered_obs.dtype.itemsize * considered_obs.shape[1]))
        considered_obs_view = np.ascontiguousarray(considered_obs).view(dt)
        y_match, count_y_match = np.unique(considered_obs_view, return_counts=True)
        y_match = y_match.view(considered_obs.dtype).reshape(-1, considered_obs.shape[1])

        # print "\n\n\n"
        # print y_match
        # print count_y_match

        for y_iter in range(len(y_match)):
            y = y_match[y_iter]
            m_xy = count_y_match[y_iter]

            g_allz_x = g_matrix[:, np.where(g_col_names == x)[0][0]]
            # print g_allz_x
            m_xy_allz = m_xy * g_allz_x * K
            # print m_xy_allz
            for iter_z in range(len(m_xy_allz)):
                if m_xy_allz[iter_z] == 0:
                    continue
                z = g_row_names[iter_z]
                for i in range(int(m_xy_allz[iter_z])):
                    y_list.append(list(y))
                    x_list.append(x)
                    z_list.append(z)

        # print y_list
        # print x_list
        # print z_list

    res = np.column_stack((np.array(y_list), np.array(x_list)))
    res = np.column_stack((np.array(res), np.array(z_list)))
    print "Ended dataset creation"
    print res
    return pn.DataFrame(data=res, columns=colnames)
