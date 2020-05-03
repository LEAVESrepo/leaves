import sys
import copy
import time
import operator
import numpy as np
from utilities_pckg import utilities


def find_best_guess_multiple_guesses(o, train_data, all_possible_guesses_dic, n_guesses, counter_secrets):
    train_o = train_data[:, 0:-1]
    train_s = train_data[:, -1]
    #   train_o_unq, count_train_o_unq = np.unique(train_o, return_counts=True)
    # train_s_unq, count_train_s_unq = np.unique(train_s, return_counts=True)

    p_s_I_o_dic = {}

    #   search for the current observable in the trainin set
    tr_o_idx = np.where(train_o == o)[0]

    secrets_select = []

    #   if the current observable is known
    if len(tr_o_idx) > 0:

        #   compute the P_train(s|o) and add to dic p_s_I_o_dic
        secrets_tmp = train_s[tr_o_idx]
        secrets_tmp_unq, count_secrets_tmp_unq = np.unique(secrets_tmp, return_counts=True)
        for tr_s_ind in range(len(secrets_tmp_unq)):
            tr_s = secrets_tmp_unq[tr_s_ind]
            p_s_I_o = count_secrets_tmp_unq[tr_s_ind] / float(len(tr_o_idx))
            p_s_I_o_dic[tr_s] = p_s_I_o

        # sorted_p_s_I_o_dic = sorted(p_s_I_o_dic.items(), key=operator.itemgetter(1))
        copy_p_s_I_o_dic = copy.deepcopy(p_s_I_o_dic)
        # print copy_p_s_I_o_dic

        #   sometimes an observable might appear with only one secret
        if n_guesses <= len(copy_p_s_I_o_dic):
            rng = n_guesses
        else:
            rng = len(copy_p_s_I_o_dic)

        for i_ter in range(rng):
            #   get argmax_s P_train(s|o)
            s = max(copy_p_s_I_o_dic.iteritems(), key=operator.itemgetter(1))[0]
            secrets_select.append(s)
            #   delete current argmax
            del copy_p_s_I_o_dic[s]

        cont = 0
        while len(secrets_select) < n_guesses:
            if counter_secrets.most_common(n_guesses)[cont][0] not in secrets_select:
                secrets_select.append(counter_secrets.most_common(n_guesses)[cont][0])
            cont += 1

    else:
        #   else pick most frequent secret from training
        for i_ter in range(n_guesses):
            secrets_select.append(counter_secrets.most_common(n_guesses)[i_ter][0])

    best_guess = None

    #   loop over guesses try to find guess with n_guess amount of argmax_s P_train(s|o)
    for guess in all_possible_guesses_dic:
        tup = all_possible_guesses_dic[guess]

        check = None

        #   loop over all secrets
        for sec_sel in secrets_select:
            #   if one secret is not in the current guess stat with new guess
            if sec_sel not in tup:
                check = False
                break
            #   otherwise the check will be true it means we have found the optimal guess
            check = True

        #   store the optimal guess and exit
        if check is True:
            best_guess = guess
            break

    return best_guess


# def find_best_guess_multiple_guesses(o, train_data, all_possible_guesses_dic, n_guesses):
#     train_o = train_data[:, 0:-1]
#     train_s = train_data[:, -1]
#     #   train_o_unq, count_train_o_unq = np.unique(train_o, return_counts=True)
#     train_s_unq, count_train_s_unq = np.unique(train_s, return_counts=True)
#
#     p_s_and_o_dic = {}
#
#     #   search for the current observable in the training set
#     tr_o_idx = np.where(train_o == o)[0]
#
#     secrets_select = []
#
#     #   if the current observable is known
#     if len(tr_o_idx) > 0:
#
#         #   compute the P_train(s|o) and add to dic p_s_I_o_dic
#         secrets_tmp = train_s[tr_o_idx]
#         secrets_tmp_unq, count_secrets_tmp_unq = np.unique(secrets_tmp, return_counts=True)
#         for tr_s_ind in range(len(secrets_tmp_unq)):
#             tr_s = secrets_tmp_unq[tr_s_ind]
#             p_s_and_o = count_secrets_tmp_unq[tr_s_ind] / float(train_data.shape[0])
#             p_s_and_o_dic[tr_s] = p_s_and_o
#
#         # sorted_p_s_I_o_dic = sorted(p_s_I_o_dic.items(), key=operator.itemgetter(1))
#         copy_p_s_and_o_dic = copy.deepcopy(p_s_and_o_dic)
#         # print copy_p_s_I_o_dic
#
#         #   sometimes an observable might appear with only one secret
#         if n_guesses <= len(copy_p_s_and_o_dic):
#             rng = n_guesses
#         else:
#             rng = len(copy_p_s_and_o_dic)
#             for i_ter in range(n_guesses-len(copy_p_s_and_o_dic)):
#                 secrets_select.append(train_s_unq[len(train_s_unq) - i_ter - 1])
#
#         for i_ter in range(rng):
#             #   get argmax_s P_train(s|o)
#             s = max(copy_p_s_and_o_dic.iteritems(), key=operator.itemgetter(1))[0]
#             secrets_select.append(s)
#             #   delete current argmax
#             del copy_p_s_and_o_dic[s]
#
#     else:
#         #   else pick most frequent secret from training
#         for i_ter in range(n_guesses):
#             secrets_select.append(train_s_unq[len(train_s_unq) - i_ter - 1])
#
#     best_guess = None
#
#     #   loop over guesses try to find guess with n_guess amount of argmax_s P_train(s|o)
#     for guess in all_possible_guesses_dic:
#         tup = all_possible_guesses_dic[guess]
#
#         check = None
#
#         #   loop over all secrets
#         for sec_sel in secrets_select:
#             #   if one secret is not in the current guess stat with new guess
#             if sec_sel not in tup:
#                 check = False
#                 break
#             #   otherwise the check will be true it means we have found the optimal guess
#             check = True
#
#         #   store the optimal guess and exit
#         if check is True:
#             best_guess = guess
#             break
#
#     return best_guess


#   solve sum_o,s p_test(s, o) g(w, s) where w = {first argmax_s p_train(s|o), second argmax_s p_train(s|o), ...}
def g_vuln_freq_computation_multiple_guesses_monodimensional_observables(train_data, test_data, g_mat, g_mat_cols,
                                                                         all_possible_guesses_dic, n_guesses,
                                                                         counter_secrets):
    test_o = test_data[:, 0:-1]
    test_s = test_data[:, -1]
    # test_o_unq, count_test_o_unq = np.unique(test_o, return_counts=True)
    test_s_unq, count_test_s_unq = np.unique(test_s, return_counts=True)

    g_vuln_freq = 0

    #   loop over test secrets
    for ts_s in test_s_unq:
        ts_s_idx = np.where(test_s == ts_s)[0]

        #   pick the observables corresponding to the secrets and count their frequency
        obs_tmp = test_o[ts_s_idx]
        obs_tmp_unq, count_obs_tmp_unq = np.unique(obs_tmp, return_counts=True)

        #   loop over the observables
        for ts_o_ind in range(len(obs_tmp_unq)):
            ts_o = obs_tmp_unq[ts_o_ind]

            #   compute P_test(s, y)
            test_p_x_y = count_obs_tmp_unq[ts_o_ind] / float(test_data.shape[0])

            #   find the best guess
            best_guess = find_best_guess_multiple_guesses(train_data=train_data,
                                                          o=ts_o,
                                                          all_possible_guesses_dic=all_possible_guesses_dic,
                                                          n_guesses=n_guesses,
                                                          counter_secrets=counter_secrets)

            #   compute the gain
            g_w_s = g_mat[best_guess, np.where(g_mat_cols == ts_s)[0]]

            #   compute the g_vuln for the current test set
            g_vuln_freq += test_p_x_y * g_w_s

    return round(g_vuln_freq, 3)


def find_best_mono_guess_general_approach(test_set, training_set, g_mat, g_mat_rows, g_mat_cols, research_obs_dic,
                                          research_sec_dic, g_mat_most_frequent_secret_idx, unq_obs_tr, unq_obs_tr_cnt,
                                          unq_secr_tr, test_obs_unique):
    start_time = time.time()

    res_dic = {}

    for test_obs_idx in range(len(test_obs_unique)):
        test_obs = test_obs_unique[test_obs_idx]

        tmp_dic = {}

        for guess in g_mat_rows:
            tmp_sum = 0
            g_mat_guess_idx = np.where(g_mat_rows == guess)[0]

            if test_obs not in unq_obs_tr:
                tmp_sum += g_mat[g_mat_guess_idx, g_mat_most_frequent_secret_idx]

            else:
                obs_cnt_train = unq_obs_tr_cnt[np.where(unq_obs_tr == test_obs)[0]]
                for x in unq_secr_tr:
                    # p_xIy = len(np.where((training_set[:0] == test_obs) & (training_set[:, 1] == x))[0]) / float(
                    #     training_set.shape[0])
                    # p_xIy = len(np.where((training_set[:, 0] == test_obs) & (training_set[:, 1] == x))[0]) / float(
                    #     obs_cnt_train)

                    lst1 = research_obs_dic[test_obs]
                    lst2 = research_sec_dic[x]

                    p_xIy = len(utilities.array_intersection(lst1=lst1, lst2=lst2)) / float(obs_cnt_train)

                    g_mat_secret_idx = np.where(g_mat_cols == x)[0]
                    tmp_sum += p_xIy * g_mat[g_mat_guess_idx, g_mat_secret_idx]

            tmp_dic[guess] = tmp_sum

        res_dic[test_obs] = max(tmp_dic.iteritems(), key=operator.itemgetter(1))[0]

    print("--- %s seconds ---" % (time.time() - start_time))

    return res_dic


def compute_freq_g_vuln_mono_guess_general_approach(test_set, best_mono_guess_res, g_mat, g_mat_cols, g_mat_rows,
                                                    test_obs_unq, test_secrets_col_int, secrets_type="int"):
    start_time = time.time()

    res_sum = 0
    for test_obs in test_obs_unq:
        data_of_interest_idx = np.where(test_set[:, 0] == test_obs)[0]
        data_of_interest = test_set[data_of_interest_idx, :]
        if secrets_type == "int":
            secrets_of_interest = np.unique(np.array(data_of_interest[:, 1], dtype=int))
        elif secrets_type == "str":
            secrets_of_interest = np.unique(np.array(data_of_interest[:, 1], dtype=str))

        for sec in secrets_of_interest:
            p_xy = len(np.where((test_set[:, 0] == test_obs) & (test_secrets_col_int == sec))[0]) / float(
                test_set.shape[0])

            g_mat_guess_idx = np.where(g_mat_rows == best_mono_guess_res[test_obs])[0]
            g_mat_secret_idx = np.where(g_mat_cols == sec)[0]

            res_sum += p_xy * g_mat[g_mat_guess_idx, g_mat_secret_idx]

    print("--- %s seconds ---" % (time.time() - start_time))

    return res_sum


###########################################################################################################################

def find_best_mono_guess_general_approach_positional(g_mat, g_mat_rows,
                                                     research_obs_dic,
                                                     research_sec_dic, g_mat_most_frequent_secret_idx, unq_obs_tr,
                                                     unq_obs_tr_cnt,
                                                     unq_secr_tr, test_obs_unique):
    start_time = time.time()

    res_dic = {}

    for test_obs_idx in range(len(test_obs_unique)):
        test_obs = test_obs_unique[test_obs_idx]

        tmp_dic = {}

        for guess in g_mat_rows:
            tmp_sum = 0
            g_mat_guess_idx = guess  # guess 0 idx 0 and guess 1 idx 1

            if test_obs not in unq_obs_tr:
                tmp_sum += g_mat[g_mat_guess_idx, g_mat_most_frequent_secret_idx]

            else:
                obs_cnt_train = unq_obs_tr_cnt[np.where(unq_obs_tr == test_obs)[0]]
                for x in unq_secr_tr:
                    # p_xIy = len(np.where((training_set[:0] == test_obs) & (training_set[:, 1] == x))[0]) / float(
                    #     training_set.shape[0])
                    # p_xIy = len(np.where((training_set[:, 0] == test_obs) & (training_set[:, 1] == x))[0]) / float(
                    #     obs_cnt_train)

                    lst1 = research_obs_dic[test_obs]
                    lst2 = research_sec_dic[x]

                    p_xIy = len(utilities.array_intersection(lst1=lst1, lst2=lst2)) / float(obs_cnt_train)

                    g_mat_secret_idx = x  # the secret's value corresponds to the index
                    tmp_sum += p_xIy * g_mat[g_mat_guess_idx, g_mat_secret_idx]

            tmp_dic[guess] = tmp_sum

        res_dic[test_obs] = max(tmp_dic.iteritems(), key=operator.itemgetter(1))[0]

    print("--- %s seconds ---" % (time.time() - start_time))

    return res_dic


def compute_freq_g_vuln_mono_guess_general_approach_posiitional(test_set, best_mono_guess_res, g_mat,
                                                                g_mat_rows,
                                                                test_obs_unq, test_secrets_col_int, secrets_type="int"):
    start_time = time.time()

    res_sum = 0
    for test_obs in test_obs_unq:
        data_of_interest_idx = np.where(test_set[:, 0] == test_obs)[0]
        data_of_interest = test_set[data_of_interest_idx, :]
        if secrets_type == "int":
            secrets_of_interest = np.unique(np.array(data_of_interest[:, 1], dtype=int))
        elif secrets_type == "str":
            secrets_of_interest = np.unique(np.array(data_of_interest[:, 1], dtype=str))
        else:
            sys.exit("ERRRRRRRRRRROOOOOOOOOOOOOOORRRRRRRRRRRRRRR!!!!!!!!!!!!")

        for sec in secrets_of_interest:
            p_xy = len(np.where((test_set[:, 0] == test_obs) & (test_secrets_col_int == sec))[0]) / float(
                test_set.shape[0])

            g_mat_guess_idx = best_mono_guess_res[test_obs]  # guess 0 idx 0 and guess 1 idx 1
            g_mat_secret_idx = sec  # the secret's value corresponds to the index

            res_sum += p_xy * g_mat[g_mat_guess_idx, g_mat_secret_idx]

    print("--- %s seconds ---" % (time.time() - start_time))

    return res_sum
