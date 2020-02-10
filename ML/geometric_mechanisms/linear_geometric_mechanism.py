"""
Class to handle the creation of a geometric data distribution and the relative plots.

It takes to have an array of secrets S, an array of observables O, a privacy parameter nu.

It separately takes care of the case |S| <= |O| and |S| > |O|.

lambda_coeff defines the truncation

no truncation: _ lambda_coeff = (e^(nu) - 1) / (e^(nu) + 1)

truncation: _ lambda_coeff = e^(nu) / (e^(nu) + 1) if the observable is at one of the two extremes of the array of the
              observables
            _ lambda_coeff = (e^(nu) - 1) / (e^(nu) + 1) otherwise
"""
import numpy as np
from scipy import stats
import pandas as pn
from utilities_pckg import utilities
import pickle
import inspect
from utilities_pckg.runtime_error_handler import runtime_error_handler  as err_hndl


#   create lists to plot the distribution around a secret
def create_list_to_plot_secret_distribution(list_of_secrets_of_interest, channel_matrix_df_path, result_dir_path):
    channel_matrix_df = pn.read_pickle(path=channel_matrix_df_path)

    channel_matrix = channel_matrix_df.values
    rows = channel_matrix_df.index.values
    cols = channel_matrix_df.columns.values

    for secret in list_of_secrets_of_interest:
        if secret not in cols:
            err_hndl(str_="no_secret")

    list_tmp_x = []
    list_tmp_y = []
    for el in list_of_secrets_of_interest:
        secr_idx = np.where(cols == el)[0]
        if len(secr_idx) != 1:
            err_hndl(str_='not_unique_secr')
        else:
            linear_truncated_geometric_pdf = channel_matrix[:, secr_idx]

        tmp_x = []
        tmp_y = []
        #
        # plt.plot(obfuscated_secrets, final_pdf)

        #   create staircase fashioned plots
        for i in range(len(rows) - 1):
            tmp_x.append(rows[i])
            tmp_y.append(linear_truncated_geometric_pdf[i])

            tmp_x.append(rows[i])
            tmp_y.append(linear_truncated_geometric_pdf[i + 1])

            tmp_x.append(rows[i + 1])
            tmp_y.append(linear_truncated_geometric_pdf[i + 1])

        list_tmp_x.append(tmp_x)
        list_tmp_y.append(tmp_y)

    utilities.createFolder(path=result_dir_path)

    with open(result_dir_path + "/list_tmp_x.pkl", "wb") as output_file:
        pickle.dump(list_tmp_x, output_file)

    with open(result_dir_path + "/list_tmp_y.pkl", "wb") as output_file:
        pickle.dump(list_tmp_y, output_file)

    return


#   draw samples_per_secret observables for each secret
def sample_from_distribution(channel_matrix_df_path, rndmstt, samples_per_secret=None):
    channel_matrix_df = pn.read_pickle(path=channel_matrix_df_path)

    channel_matrix = channel_matrix_df.values
    rows = channel_matrix_df.index.values
    cols = channel_matrix_df.columns.values

    samples_draws = []

    if samples_per_secret is not None:
        for secret_iter in range(len(cols)):
            #   observables distribution for the currently considered secret no normalization, each col sums up to 1
            sample_distr = stats.rv_discrete(name='draw', values=(rows, channel_matrix[:, secret_iter]), seed=rndmstt)
            #   draw samples_per_secret observables from the distribution for the currently considered secret
            draw = sample_distr.rvs(size=samples_per_secret, random_state=rndmstt)

            #   create column with secret
            secret = np.array([cols[secret_iter] for i in range(samples_per_secret)])
            #   stack the two columns: observables on the left, secrets on the right
            samples_draw = np.column_stack((draw, secret))
            #   append partial dataset to list
            samples_draws.append(samples_draw)
        #   vertically concatenate the elements of the list
        samples_draws = np.concatenate(samples_draws, axis=0)
    else:
        err_hndl(str_="sample_per_secret")

    return samples_draws


class LinearGeometricMechanism:
    def __init__(self,
                 secrets,
                 observables,
                 nu,
                 truncation=False):
        #   array of secrets
        self.secrets = secrets
        #   array of observables
        self.observables = observables

        #   array of secrets symbols: it is the colnames of the channel matrix
        self.unique_secrets = np.unique(self.secrets)
        #   array of observables symbols: it is the rownames of the channel matrix
        self.unique_observables = np.unique(self.observables)

        #   privacy parameter
        self.nu = nu

        #   truncation flag
        self.truncation = truncation

        #   resulting matrix representing P(observable_i|secret_j)
        self.channel_matrix = None

    def upd_len_unique_observables(self, shift):
        len_unique_observables = len(self.unique_observables)
        if shift > 0:
            tmp = len_unique_observables - (2 * shift)
            if tmp > 0:
                len_unique_observables = tmp
            else:
                err_hndl(str_="shift_not_possible", add=inspect.stack()[0][3])
        return len_unique_observables

    def upd_max_unique_observables(self, shift):
        max_unique_observables = max(self.unique_observables)
        if shift > 0:
            tmp = max_unique_observables - (2 * shift)
            if tmp > 0:
                max_unique_observables = tmp
            else:
                err_hndl(str_="shift_not_possible", add=inspect.stack()[0][3])
        return max_unique_observables

    #   return the array of remapped secrets, both symmetric or asymmetric
    #   shift is only taken into account if > 0
    def compute_symmetry_shift(self, shift):
        max_unique_observables = self.upd_max_unique_observables(shift=shift)
        len_unique_observables = self.upd_len_unique_observables(shift=shift)
        max_rescaled_secret = max(self.unique_secrets) * len_unique_observables / float(len(self.unique_secrets))
        _shift_ = (max_unique_observables - max_rescaled_secret) / float(2.)
        return _shift_

    #   rescale the secret so that it is in the same range as the observables
    #   shift is only taken into account if > 0
    def rescale_secret_to_observables_set(self,
                                          secret_to_be_rescaled,
                                          symmetry,
                                          shift):
        #   case | S | > | O | in F-BLEU (according to us, the module operation creates collisions ask the author for
        #   further info)
        if len(self.unique_secrets) > len(self.unique_observables):
            secret_to_be_rescaled = secret_to_be_rescaled % float(len(self.unique_observables))

        if symmetry:
            len_unique_observables = self.upd_len_unique_observables(shift=shift)
            return secret_to_be_rescaled * len_unique_observables / float(
                len(self.unique_secrets)) + self.compute_symmetry_shift(shift) + shift
        else:
            #   obtain F-BLEU secrets not symmetric remapping (example: 100 secrets and 10K observables, the secrets go from
            #   0 to 99 and so from 0 to 9900 when normalized)
            return secret_to_be_rescaled * len(self.unique_observables) / float(len(self.unique_secrets))

    #   create the distribution around a point contained in the possible values set
    def linear_truncated_geometric_mechanism_pdf(self, center_of_the_distribution, delta, symmetry, shift):
        if center_of_the_distribution not in self.unique_secrets:
            err_hndl("no_secret", center_of_the_distribution)
        else:
            center_of_the_distribution += delta
            loc = self.rescale_secret_to_observables_set(secret_to_be_rescaled=center_of_the_distribution,
                                                         symmetry=symmetry, shift=shift)

            #   according to the distribution we need a lambda value for each observable since its values depend on the
            #   current observation: the following computation relies on the fact that 'unique' also sorts the elements in
            #   increasing order; if the secrets or observables are not sortable it might be a problem
            lambda_coeff = np.zeros(len(self.unique_observables))
            for i in range(0, len(lambda_coeff)):
                lambda_coeff[i] = (np.exp(self.nu) - 1) / float(np.exp(self.nu) + 1)
            #   if truncation is requested, modify the extremes of the lambda_coeff vector
            if self.truncation:
                lambda_coeff[np.argmin(self.unique_observables)] = lambda_coeff[
                    np.argmax(self.unique_observables)] = np.exp(
                    self.nu) / float(np.exp(self.nu) + 1)

            #   define pdf as P(o|s) = e^(-(nu) * |g(s) - o|)
            pdf = np.exp(-self.nu * abs(self.unique_observables - loc))
            #   define final_pdf as P(o|s) = lambda * e^(-(nu) * |g(s) - o|)
            final_pdf = np.multiply(lambda_coeff, pdf)

            return final_pdf

    #   create channel matrix: each entry is a different observation and each column is a secret, the values in the cells
    #   are P(observable|secret): it is built from the real PDF, not from frequentist observations
    def create_channel_matrix_from_known_distribution(self, shift, save_channel_matrix_path, delta=0, symmetry=True):
        channel_matrix = np.zeros((len(self.unique_observables), len(self.unique_secrets)))

        column_iterator = 0
        for secret in self.unique_secrets:
            utilities.inline_print_secret(secret)
            #   create distribution around each secret i.e. P(o|s)
            prob_observables_given_secret = self.linear_truncated_geometric_mechanism_pdf(
                center_of_the_distribution=secret,
                delta=delta,
                symmetry=symmetry,
                shift=shift)
            #   if the truncation is false there is the possibility that the sum along a column sum_o P(O|S=s) != 1, so it
            #   is as if we overlook the probability of values for the secrets outside of the observables space:
            #   renormalization needed
            if self.truncation is False:
                # print "Normalizing..."
                prob_observables_given_secret_norm = tuple(
                    p / sum(prob_observables_given_secret) for p in prob_observables_given_secret)
                # add pdf as a column of the channel matrix for current secret s
                channel_matrix[:, column_iterator] = prob_observables_given_secret_norm
                # print "End normalization."
            else:
                channel_matrix[:, column_iterator] = prob_observables_given_secret

            # print secret, " ---> ", sum(channel_matrix[:, column_iterator])
            column_iterator += 1

        self.channel_matrix = channel_matrix

        pn.to_pickle(obj=pn.DataFrame(data=self.channel_matrix, index=self.unique_observables, columns=self.unique_secrets),
                     path=save_channel_matrix_path)

        return self.channel_matrix
