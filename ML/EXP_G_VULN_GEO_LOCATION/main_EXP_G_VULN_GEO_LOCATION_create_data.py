import pandas as pn
from utilities_pckg import utilities, g_vuln_computation
from utilities_pckg.runtime_error_handler import runtime_error_handler as err_hndl
import numpy as np
from scipy import stats
from tqdm import tqdm

EXP_G_VULN_GEO_LOCATION_FOLDER_PATH = "/home/comete/mromanel/MILES_EXP/EXP_GEO_LOCATION_QIF_LIB_SETTING/"
CHANNEL_MATRIX_FILE = EXP_G_VULN_GEO_LOCATION_FOLDER_PATH + "channel.pkl"
ORIGINAL_SECRETS_OCCURRENCES_FILE = EXP_G_VULN_GEO_LOCATION_FOLDER_PATH + "cell_dict_occurences_per_cell.pkl"

MULTIPLICATIVE_FACTOR_FOR_SETS_CARDINALITY = [0.5, 1., 1.5]
VALIDATION_CARD_AS_FRACTION_OF_TR_CARD = 0.1

TRAINING_ITERATIONS = 10
TEST_ITERATIONS = 100

DATA_FOLDER = EXP_G_VULN_GEO_LOCATION_FOLDER_PATH + "DATA_FOLDER/"
utilities.createFolder(DATA_FOLDER)


def sample_from_channel(channel_matrix, rndmstt, samples_per_secret_dictionary):
    #   observ
    rows = np.arange(start=0, stop=channel_matrix.shape[0], step=1)
    #   secrets
    cols = np.arange(start=0, stop=channel_matrix.shape[1], step=1)

    samples_draws = []

    for secret in samples_per_secret_dictionary:
        occurr = samples_per_secret_dictionary[secret]

        secret_iter = np.where(cols == secret)[0][0]
        #   observables distribution for the currently considered secret no normalization, each col sums up to 1
        sample_distr = stats.rv_discrete(name='draw', values=(rows, channel_matrix[:, secret_iter]), seed=rndmstt)
        #   draw samples_per_secret observables from the distribution for the currently considered secret
        draw = sample_distr.rvs(size=occurr, random_state=rndmstt)

        #   create column with secret
        secret = np.array([cols[secret_iter] for i in range(occurr)])
        #   stack the two columns: observables on the left, secrets on the right
        samples_draw = np.column_stack((draw, secret))
        #   append partial dataset to list
        samples_draws.append(samples_draw)

    #   vertically concatenate the elements of the list
    samples_draws = np.concatenate(samples_draws, axis=0)

    return samples_draws


def main_EXP_G_VULN_GEO_LOCATION_create_data():
    channel_matrix = pn.read_pickle(path=CHANNEL_MATRIX_FILE)
    secrets_occurr_dictionary = pn.read_pickle(path=ORIGINAL_SECRETS_OCCURRENCES_FILE)
    # print "secrets_occurr_dictionary ===> ", secrets_occurr_dictionary

    tot_occurr = 0
    maxx_occurr = 0
    for secret in secrets_occurr_dictionary:
        current_secret_occurr = secrets_occurr_dictionary[secret]
        if current_secret_occurr > maxx_occurr:
            maxx_occurr = current_secret_occurr
        tot_occurr += current_secret_occurr
        # print secret, " ===> ", current_secret_occurr
    # print "maxx_occurr ===> ", maxx_occurr

    secrets_prior_dictionary = {}
    maxx_freq = 0
    for secret in secrets_occurr_dictionary:
        secrets_prior_dictionary[secret] = secrets_occurr_dictionary[secret] / float(tot_occurr)
        if secrets_prior_dictionary[secret] > maxx_freq:
            maxx_freq = secrets_prior_dictionary[secret]
    # print "secrets_prior_dictionary ===> ", secrets_prior_dictionary
    # print "maxx_freq ===> ", maxx_freq

    for mult_card in MULTIPLICATIVE_FACTOR_FOR_SETS_CARDINALITY:

        reform_secrets_occurr_dictionary_tr_ts = {}
        reform_secrets_occurr_dictionary_val = {}
        for secret in secrets_occurr_dictionary:
            reform_secrets_occurr_dictionary_tr_ts[secret] = int(
                round(secrets_occurr_dictionary[secret] * float(mult_card), 0))
            reform_secrets_occurr_dictionary_val[secret] = int(
                round(secrets_occurr_dictionary[secret] * float(mult_card) * float(VALIDATION_CARD_AS_FRACTION_OF_TR_CARD),
                      0))

        training_set_size = 0
        for keys in reform_secrets_occurr_dictionary_tr_ts:
            training_set_size += reform_secrets_occurr_dictionary_tr_ts[keys]
        test_set_size = training_set_size

        validation_set_size = 0
        for keys in reform_secrets_occurr_dictionary_val:
            validation_set_size += reform_secrets_occurr_dictionary_val[keys]

        # print "reform_secrets_occurr_dictionary_tr_ts ===> ", reform_secrets_occurr_dictionary_tr_ts

        for train_iteration in tqdm(range(TRAINING_ITERATIONS)):
            training_and_validation_and_test_set_store_folder = DATA_FOLDER + str(
                training_set_size) + "_training_and_" + str(validation_set_size) + "_validation_and_" + str(
                test_set_size) + "_test_store_folder_train_iteration_" + str(train_iteration) + "/"

            utilities.createFolder(training_and_validation_and_test_set_store_folder)

            training_set_mat = sample_from_channel(channel_matrix=channel_matrix,
                                                   rndmstt=utilities.create_new_rndm_state(),
                                                   samples_per_secret_dictionary=reform_secrets_occurr_dictionary_tr_ts)

            training_df = pn.DataFrame(data=training_set_mat, columns=["O_train", "S_train"])
            training_df.to_pickle(path=training_and_validation_and_test_set_store_folder + "/training_set.pkl")

            print training_set_mat.shape

            ################################################################################################################

            validation_set_mat = sample_from_channel(channel_matrix=channel_matrix,
                                                     rndmstt=utilities.create_new_rndm_state(),
                                                     samples_per_secret_dictionary=reform_secrets_occurr_dictionary_val)

            validation_df = pn.DataFrame(data=validation_set_mat, columns=["O_val", "S_val"])
            validation_df.to_pickle(path=training_and_validation_and_test_set_store_folder + "/validation_set.pkl")

            print validation_set_mat.shape

            ################################################################################################################

            for test_iteration in range(TEST_ITERATIONS):
                test_set_mat = sample_from_channel(channel_matrix=channel_matrix,
                                                   rndmstt=utilities.create_new_rndm_state(),
                                                   samples_per_secret_dictionary=reform_secrets_occurr_dictionary_tr_ts)

                test_set_store_folder = training_and_validation_and_test_set_store_folder + str(
                    test_set_size) + "_size_test_sets/"
                utilities.createFolder(path=test_set_store_folder)

                test_df = pn.DataFrame(data=test_set_mat, columns=["O_test", "S_test"])
                test_df.to_pickle(
                    path=test_set_store_folder + "/test_set_" + str(test_iteration) + ".pkl")
