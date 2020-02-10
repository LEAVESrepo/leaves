from geometric_mechanisms import linear_geometric_mechanism
from utilities_pckg import g_vuln_computation, utilities
from utilities_pckg.runtime_error_handler import runtime_error_handler as err_hndl
import pandas as pn
import inspect

EMPIRICAL_ESTIMATES_FOLDER_PATH = "/home/comete/mromanel/MILES_EXP/EMPIRICAL_ESTIMATES/"
CHANNEL_PATH = EMPIRICAL_ESTIMATES_FOLDER_PATH + "channel_df_norm.pkl"

DATA_FOLDER = EMPIRICAL_ESTIMATES_FOLDER_PATH + "DATA_FOLDER/"
utilities.createFolder(DATA_FOLDER)

TRAINING_SET_SIZE = [5000, 10000, 20000]
TEST_SET_SIZE = [5000, 5000, 5000]
VALIDATION_SET_SIZE = [500, 1000, 2000]
TEST_ITERATIONS = 10
TRAIN_ITERATIONS = 1

CREATE_TEST_SET = True


def main_EMPIRICAL_ESTIMATES_create_data():
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  geometric distribution loading  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    print("\n####################################################################################")
    print("#########################  geometric distribution loading  #########################")
    print("####################################################################################\n")

    utilities.createFolder(DATA_FOLDER)

    channel_matrix_df = pn.read_pickle(path=CHANNEL_PATH)

    channel_matrix = channel_matrix_df.values
    print channel_matrix.shape

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  create training sets  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    #   X are the observables and y are the secrets (respectively col 0 and 1), stratify wrt to secret
    #   split training and test data

    if len(TEST_SET_SIZE) != len(TRAINING_SET_SIZE) or len(VALIDATION_SET_SIZE) != len(TRAINING_SET_SIZE):
        err_hndl(str_="array_sizes_not_matching", add=inspect.stack()[0][3])

    for size_list_iterator in range(len(TRAINING_SET_SIZE)):
        training_set_size = TRAINING_SET_SIZE[size_list_iterator]
        validation_set_size = VALIDATION_SET_SIZE[size_list_iterator]
        test_set_size = TEST_SET_SIZE[size_list_iterator]

        for train_iteration in range(TRAIN_ITERATIONS):
            training_set_mat = linear_geometric_mechanism.sample_from_distribution(
                channel_matrix_df_path=CHANNEL_PATH,
                rndmstt=utilities.create_new_rndm_state(),
                samples_per_secret=int(
                    training_set_size / len(
                        channel_matrix_df.columns.values)))

            training_and_validation_and_test_set_store_folder = DATA_FOLDER + str(
                training_set_size) + "_training_and_" + str(validation_set_size) + "_validation_and_" + str(
                test_set_size) + "_test_store_folder_train_iteration_" + str(train_iteration) + "/"

            utilities.createFolder(path=training_and_validation_and_test_set_store_folder)

            training_df = pn.DataFrame(data=training_set_mat, columns=["O_train", "S_train"])
            training_df.to_pickle(path=training_and_validation_and_test_set_store_folder + "/training_set.pkl")

            ################################################################################################################

            validation_set_mat = linear_geometric_mechanism.sample_from_distribution(
                channel_matrix_df_path=CHANNEL_PATH,
                rndmstt=utilities.create_new_rndm_state(),
                samples_per_secret=int(
                    validation_set_size / len(
                        channel_matrix_df.columns.values)))

            validation_df = pn.DataFrame(data=validation_set_mat, columns=["O_val", "S_val"])
            validation_df.to_pickle(path=training_and_validation_and_test_set_store_folder + "/validation_set.pkl")

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  create test sets  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            if CREATE_TEST_SET:
                print("\n####################################################################################")
                print("#################################  create test sets  ################################")
                print("####################################################################################\n")
                for test_iteration in range(TEST_ITERATIONS):
                    test_set_mat = linear_geometric_mechanism.sample_from_distribution(
                        channel_matrix_df_path=CHANNEL_PATH,
                        rndmstt=utilities.create_new_rndm_state(),
                        samples_per_secret=int(
                            test_set_size / len(
                                channel_matrix_df.columns.values)))

                    test_set_store_folder = training_and_validation_and_test_set_store_folder + str(
                        test_set_size) + "_size_test_sets/"
                    utilities.createFolder(path=test_set_store_folder)

                    test_df = pn.DataFrame(data=test_set_mat, columns=["O_test", "S_test"])
                    test_df.to_pickle(
                        path=test_set_store_folder + "/test_set_" + str(test_iteration) + ".pkl")
