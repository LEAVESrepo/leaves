import numpy as np
import pandas as pn
from tqdm import tqdm

tqdm.monitor_interval = 0


#   g_mat = (guesses, secrets)


def compute_empirical_estimate(data, g_mat_cols, g_mat, g_mat_rows):
    data = data.values[0:100, :]
    observables = data[:, 0]
    observables_unq, count_obs_unq = np.unique(observables, return_counts=True)

    secrets = data[:, -1]
    secrets_unq, count_secrets_unq = np.unique(secrets, return_counts=True)

    sum_obs = 0

    for ob_iter in tqdm(range(len(observables_unq))):
        ob = observables_unq[ob_iter]

        # max_w = -float("inf")
        max_w_list = []

        for w in g_mat_rows:

            sum_s = 0
            for s in secrets_unq:
                g_w_s = g_mat[np.where(g_mat_rows == w)[0], np.where(g_mat_cols == s)[0]]
                if len(secrets) < len(observables):
                    s_idx = np.where(secrets == s)[0]
                    ob_idx = np.where(observables[s_idx] == ob)[0]
                    p_s_o = len(ob_idx) / float(data.shape[0])

                else:
                    ob_idx = np.where(observables == ob)[0]
                    s_idx = np.where(secrets[ob_idx] == s)[0]
                    p_s_o = len(s_idx) / float(data.shape[0])

                sum_s += p_s_o * g_w_s

            # if sum_s > max_w:
            #    max_w = sum_s
            max_w_list.append(sum_s)

        # sum_obs += max_w 0.93486
        sum_obs += max(max_w_list)

    return sum_obs
