import numpy as np
import utilities_pckg.runtime_error_handler as err_hndl
import inspect
from tqdm import tqdm
import sys

tqdm.monitor_interval = 0


def compute_g_vuln_with_remapping_multidimesional_inputs(final_mat, g_mat, g_mat_rows, g_mat_cols):
    observables = final_mat[:, 0:final_mat.shape[1] - 2]
    dt = np.dtype((np.void, observables.dtype.itemsize * observables.shape[1]))
    b = np.ascontiguousarray(observables).view(dt)
    unique_obs = np.unique(b)
    unique_obs = unique_obs.view(observables.dtype).reshape(-1, observables.shape[1])

    sum_ = 0

    for j_ter in range(unique_obs.shape[0]):
        unique_ob = unique_obs[j_ter, :]
        #   example: np.where((O_train == ot).all(axis=1))[0]
        ob_idx = np.where((final_mat[:, 0:final_mat.shape[1] - 2] == tuple(unique_ob)).all(axis=1))[0]
        unique_secr, occurr = np.unique(final_mat[ob_idx, -2], return_counts=True)
        remap = final_mat[ob_idx, -1]
        # print remap
        if len(np.unique(remap)) != 1:
            err_hndl.runtime_error_handler(str_="not_determ", add=inspect.stack()[0][3])

        remap = remap[0]
        for i_ter in range(len(unique_secr)):
            #   p_so is the joint probability of a secret and observable
            p_so = occurr[i_ter] / float(final_mat.shape[0])
            g_element = g_mat[np.where(g_mat_rows == remap)[0][0], np.where(g_mat_cols == unique_secr[i_ter])[0][0]]
            sum_ += p_so * float(g_element)

    # return 1 - round(sum_, 3)
    return round(sum_, 3)


def compute_g_vuln_with_remapping(final_mat, g_mat, g_mat_rows, g_mat_cols):
    #   final mat has 3 cols: observables, secrets, remapping given by the classifier (f(observ))

    unique_obs = np.unique(final_mat[:, 0])

    sum_ = 0

    for j_ter in range(len(unique_obs)):
        unique_ob = unique_obs[j_ter]
        ob_idx = np.where(final_mat[:, 0] == unique_ob)[0]
        unique_secr, occurr = np.unique(final_mat[ob_idx, 1], return_counts=True)
        remap = final_mat[ob_idx, 2]
        if len(np.unique(remap)) != 1:
            err_hndl.runtime_error_handler(str_="not_determ", add=inspect.stack()[0][3])
        remap = remap[0]
        for i_ter in range(len(unique_secr)):
            #   p_so is the joint probability of a secret and observable
            p_so = occurr[i_ter] / float(final_mat.shape[0])
            g_element = g_mat[np.where(g_mat_rows == remap)[0][0], np.where(g_mat_cols == unique_secr[i_ter])[0][0]]
            sum_ += p_so * float(g_element)

    # return 1 - round(sum_, 3)
    return round(sum_, 3)


def compute_g_vuln_star(channel, rows, cols, secret_prior_dict, g_mat, g_mat_rows, g_mat_cols):
    sum_ = 0

    #   loop over the rows (observables) P(O_i|S_j) for every j
    for row_idx in tqdm(range(len(rows))):
        max_tmp = -float("inf")
        for remap_row in range(len(g_mat_rows)):
            sum_tmp = 0
            for secret in cols:
                prior = secret_prior_dict[secret]
                C = channel[row_idx, np.where(cols == secret)[0][0]]
                g = g_mat[remap_row, np.where(g_mat_cols == secret)[0][0]]
                sum_tmp += prior * C * g

            if sum_tmp > max_tmp:
                max_tmp = sum_tmp
        sum_ += max_tmp

    #   return 1 - sum_, round is necessary because for very small numbers approximations to zero might be slightly
    #   negative
    # return 1 - round(sum_, 3)
    return round(sum_, 3)


def compute_g_vuln_with_remapping_positional(final_mat, g_mat):
    #   final mat has 3 cols: observables, secrets, remapping given by the classifier (f(observ))

    unique_obs = np.unique(final_mat[:, 0])

    sum_ = 0

    for j_ter in range(len(unique_obs)):
        unique_ob = unique_obs[j_ter]
        ob_idx = np.where(final_mat[:, 0] == unique_ob)[0]
        unique_secr, occurr = np.unique(final_mat[ob_idx, 1], return_counts=True)
        remap = final_mat[ob_idx, 2]
        if len(np.unique(remap)) != 1:
            err_hndl.runtime_error_handler(str_="not_determ", add=inspect.stack()[0][3])
        remap = remap[0]
        for i_ter in range(len(unique_secr)):
            #   p_so is the joint probability of a secret and observable
            p_so = occurr[i_ter] / float(final_mat.shape[0])
            g_element = g_mat[remap, unique_secr[i_ter]]  # in both guess and secret field the name coincides with the idx
            sum_ += p_so * float(g_element)

    # return 1 - round(sum_, 3)
    return round(sum_, 3)
