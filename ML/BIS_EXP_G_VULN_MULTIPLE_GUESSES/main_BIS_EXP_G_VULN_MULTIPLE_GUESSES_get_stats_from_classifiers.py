import numpy as np
from utilities_pckg import utilities, gpu_setup
from keras.utils import to_categorical
import os
import pandas as pn
import sys
from keras.models import load_model
from utilities_pckg.runtime_error_handler import exception_call as excpt_cll
from scipy import stats
from sklearn import preprocessing

# MIN_OBSERVABLE = 0
# MAX_OBSERVABLE = 15999
APPROACHES = ['ANN']
NUM_CLASSES = 45
NU = 0.02
TRAINING_SET_SIZE = [10000, 30000, 50000]
TEST_SET_SIZE = [10000, 30000, 50000]
VALIDATION_SET_SIZE = [1000, 3000, 5000]
TEST_ITERATIONS = 100
TRAIN_ITERATIONS = 10
MODEL_NAME = ""

EXP_G_VULN_MULTIPLE_GUESSES_FOLDER = "/home/comete/mromanel/MILES_EXP/EXP_G_VULN_MULTIPLE_GUESSES/"
DATA_FOLDER = EXP_G_VULN_MULTIPLE_GUESSES_FOLDER + "DATA_FOLDER_AFTER_OUR_PREPROCESSING_2_GUESSES/"


def read_command_line_options():
    thismodule = sys.modules[__name__]
    for idx, key_val in enumerate(sys.argv, 0):
        if key_val in ['--id_gpu'] and len(sys.argv) > idx + 1:
            try:
                thismodule.ID_GPU = sys.argv[idx + 1].strip()
            except ValueError as val_err:
                excpt_cll(idx=idx, key_val=key_val)

        if key_val in ['--perc_gpu'] and len(sys.argv) > idx + 1:
            try:
                thismodule.PERC_GPU = float(sys.argv[idx + 1].strip())
            except ValueError as val_err:
                excpt_cll(idx=idx, key_val=key_val)

        if key_val in ['--model_name', '-mn'] and len(sys.argv) > idx + 1:
            try:
                thismodule.MODEL_NAME = sys.argv[idx + 1].strip()
            except ValueError as val_err:
                excpt_cll(idx=idx, key_val=key_val)

        if key_val in ['--tr_size'] and len(sys.argv) > idx + 1:
            #   it must be something like -hnc [1,2,3]
            string_to_be_adapted = sys.argv[idx + 1].strip()
            print string_to_be_adapted
            if string_to_be_adapted[0] != '[' or string_to_be_adapted[-1] != ']':
                excpt_cll(idx=idx, key_val=key_val)
            else:
                strip_string_to_be_adapted = string_to_be_adapted[1:-1]
                split_list = strip_string_to_be_adapted.split(',')
                TRAINING_SET_SIZE_TMP = []
                for item in split_list:
                    try:
                        TRAINING_SET_SIZE_TMP.append(int(item))
                    except ValueError as val_err:
                        excpt_cll(idx=idx, key_val=key_val)
                thismodule.TRAINING_SET_SIZE = TRAINING_SET_SIZE_TMP

        if key_val in ['--ts_size'] and len(sys.argv) > idx + 1:
            #   it must be something like -hnc [1,2,3]
            string_to_be_adapted = sys.argv[idx + 1].strip()
            print string_to_be_adapted
            if string_to_be_adapted[0] != '[' or string_to_be_adapted[-1] != ']':
                excpt_cll(idx=idx, key_val=key_val)
            else:
                strip_string_to_be_adapted = string_to_be_adapted[1:-1]
                split_list = strip_string_to_be_adapted.split(',')
                TEST_SET_SIZE_TMP = []
                for item in split_list:
                    try:
                        TEST_SET_SIZE_TMP.append(int(item))
                    except ValueError as val_err:
                        excpt_cll(idx=idx, key_val=key_val)
                thismodule.TEST_SET_SIZE = TEST_SET_SIZE_TMP

        if key_val in ['--val_size'] and len(sys.argv) > idx + 1:
            #   it must be something like -hnc [1,2,3]
            string_to_be_adapted = sys.argv[idx + 1].strip()
            print string_to_be_adapted
            if string_to_be_adapted[0] != '[' or string_to_be_adapted[-1] != ']':
                excpt_cll(idx=idx, key_val=key_val)
            else:
                strip_string_to_be_adapted = string_to_be_adapted[1:-1]
                split_list = strip_string_to_be_adapted.split(',')
                VAL_SET_SIZE_TMP = []
                for item in split_list:
                    try:
                        VAL_SET_SIZE_TMP.append(int(item))
                    except ValueError as val_err:
                        excpt_cll(idx=idx, key_val=key_val)
                thismodule.VALIDATION_SET_SIZE = VAL_SET_SIZE_TMP
        if key_val in ['--bg_test_it'] and len(sys.argv) > idx + 1:
            try:
                thismodule.TEST_ITERATIONS_BEG = int(sys.argv[idx + 1].strip())
            except ValueError as val_err:
                excpt_cll(idx=idx, key_val=key_val)

        if key_val in ['--end_test_it'] and len(sys.argv) > idx + 1:
            try:
                thismodule.TEST_ITERATIONS_END = int(sys.argv[idx + 1].strip())
            except ValueError as val_err:
                excpt_cll(idx=idx, key_val=key_val)


def eval_on_dataset(data_set, model, min_max_scaler):
    X_data = data_set[:, 0]
    X_data_unique = np.unique(X_data)

    X_data = X_data.reshape(-1, 1)
    X_data_preprocessed = min_max_scaler.transform(X_data)

    X_data_unique = X_data_unique.reshape(-1, 1)
    X_data_unique = min_max_scaler.transform(X_data_unique)

    y_data = data_set[:, 1]

    #   this  will have an element for each element in the test set
    # X_data_preprocessed = preprocess.scaler_between_minus_one_and_one(column=X_data,
    #                                                                  min_column=MIN_OBSERVABLE,
    #                                                                 max_column=MAX_OBSERVABLE)

    #   this will have an element for each unique value in the test set
    observable_marginal_dict = utilities.compute_prior_distribution_from_array_of_symbols(X_data)
    # print observable_marginal_dict

    #   this too will have an element for each unique value in the test set
    # X_test_preprocessed_unique = preprocess.scaler_between_minus_one_and_one(column=X_data_unique,
    #                                                                      min_column=MIN_OBSERVABLE,
    #                                                                      max_column=MAX_OBSERVABLE)

    X_test_preprocessed_unique = min_max_scaler.transform(X_data_unique)
    if len(X_test_preprocessed_unique) != len(np.unique(X_test_preprocessed_unique)):
        sys.exit("The preprocessing created some collision which might affect the computation")

    if len(observable_marginal_dict) != len(X_test_preprocessed_unique):
        sys.exit("The preprocessing created some collision which might affect the computation")

    if len(observable_marginal_dict) != len(np.unique(X_data_preprocessed)):
        sys.exit("The preprocessing created some collision which might affect the computation")

    # print X_test_preprocessed_unique

    new_old_obs = {}
    for i in range(len(X_test_preprocessed_unique)):
        new_old_obs[X_test_preprocessed_unique[i][0]] = X_data_unique[i][0]

    # print new_old_obs"""

    ANN_prediction = []

    pred = model.predict(x=X_data_preprocessed)

    for row_iter in range(pred.shape[0]):
        ANN_prediction.append(np.argmax(pred[row_iter, :]))

    accuracy = utilities.compute_accuracy(y_classes=y_data, y_pred_classes=ANN_prediction)

    y_data = to_categorical(y=y_data, num_classes=NUM_CLASSES)
    print y_data.shape

    loss, acc = model.evaluate(x=X_data_preprocessed, y=y_data, batch_size=500000)

    return [loss, acc, accuracy]


def main_EXP_G_VULN_MULTIPLE_GUESSES_get_stats_from_classifiers():
    read_command_line_options()
    gpu_setup.gpu_setup(id_gpu="3", memory_percentage=0.5)

    if len(TEST_SET_SIZE) != len(TRAINING_SET_SIZE) or len(VALIDATION_SET_SIZE) != len(TRAINING_SET_SIZE):
        sys.exit("The set size lists must all contain the same amount of items.")

    RESULT_FOLDER = EXP_G_VULN_MULTIPLE_GUESSES_FOLDER + "RESULT_FOLDER_REMAPPING/" + MODEL_NAME + "/"

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ANN approach  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%$$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if 'ANN' in APPROACHES:
        print("\n####################################################################################")
        print("###################################  ANN approach  ##################################")
        print("####################################################################################\n")

        ANN_file_get_stats_from_classifiers = open(RESULT_FOLDER + "ANN_file_get_stats_from_classifiers.txt", "wa")

        for size_list_iterator in range(len(TRAINING_SET_SIZE)):

            #   select the current values for the sizes (useful to keep track in the names of the
            training_set_size = TRAINING_SET_SIZE[size_list_iterator]
            validation_set_size = VALIDATION_SET_SIZE[size_list_iterator]
            test_set_size = TEST_SET_SIZE[size_list_iterator]

            for train_iteration in range(TRAIN_ITERATIONS):
                filepath = RESULT_FOLDER + str(training_set_size) + "_training_size_and_" + str(
                    validation_set_size) + "_validation_size_iteration_" + str(train_iteration)

                if not os.path.exists(filepath):
                    continue

                else:
                    ANN_model = load_model(filepath=filepath + "/classifier_net_model")
                    # print ANN_model.summary()

                training_set = pn.read_pickle(
                    path=DATA_FOLDER + str(training_set_size) + "_training_and_" + str(
                        validation_set_size) + "_validation_and_" + str(
                        test_set_size) + "_test_store_folder_train_iteration_" + str(train_iteration) +
                         "/training_set.pkl").values

                X_train = training_set[:, 0]
                min_max_scaler = preprocessing.MinMaxScaler()

                X_train = X_train.reshape(-1, 1)
                X_train = min_max_scaler.fit_transform(X_train)

                ANN_file_get_stats_from_classifiers.write(
                    "\n\n\n#################################################################")
                ANN_file_get_stats_from_classifiers.write("\n\n\n#################################  training_set_" + str(
                    train_iteration) + " ################################")
                ANN_file_get_stats_from_classifiers.write(
                    "\n\n\n#################################################################")

                tr_loss, tr_acc, tr_myacc = eval_on_dataset(data_set=training_set, model=ANN_model,
                                                            min_max_scaler=min_max_scaler)

                print "model ---> ", MODEL_NAME

                print "\nTraining set, ", str(training_set_size), "size, iteration ", str(train_iteration)
                ANN_file_get_stats_from_classifiers.write(
                    "\nTraining set, " + str(training_set_size) + " size, iteration " + str(train_iteration))

                print "\ntraining_loss: ", round(tr_loss, 3)
                ANN_file_get_stats_from_classifiers.write("\ntraining_loss: " + str(round(tr_loss, 3)))

                print "\ntraining_accuracy: ", round(tr_acc, 3)
                ANN_file_get_stats_from_classifiers.write("\ntraining_accuracy: " + str(round(tr_acc, 3)) + "\n")

                print "\ntraining_my_accuracy", round(tr_myacc, 3)
                ANN_file_get_stats_from_classifiers.write("\ntraining_my_accuracy: " + str(round(tr_myacc, 3)) + "\n")

                ts_loss_list_for_avg = []
                ts_accuracy_list_for_avg = []
                ts_my_accuracy_list_for_avg = []

                for test_iterator in range(0, TEST_ITERATIONS):
                    print "\n\n\n#################################  test_set_" + str(
                        test_iterator) + " ################################"
                    # ANN_file_get_stats_from_classifiers.write("\n\n\n#################################  test_set_" + str(
                    #    test_iterator) + " ################################")

                    test_set = pn.read_pickle(
                        path=DATA_FOLDER + str(training_set_size) + "_training_and_" + str(
                            validation_set_size) + "_validation_and_" + str(
                            test_set_size) + "_test_store_folder_train_iteration_" + str(train_iteration) + "/" + str(
                            test_set_size) + "_size_test_sets/test_set_" + str(test_iterator)).values

                    ts_loss, ts_acc, ts_myacc = eval_on_dataset(data_set=test_set, model=ANN_model,
                                                                min_max_scaler=min_max_scaler)

                    ts_loss_list_for_avg.append(ts_loss)
                    ts_accuracy_list_for_avg.append(ts_acc)
                    ts_my_accuracy_list_for_avg.append(ts_myacc)

                ts_loss_array_for_avg = np.array(ts_loss_list_for_avg)

                test_loss_avg = round(np.mean(ts_loss_array_for_avg, axis=0), 3)
                test_loss_avg_var = round(np.var(a=ts_loss_array_for_avg, ddof=1), 3)
                test_loss_avg_standard_deviation = round(np.std(a=ts_loss_array_for_avg, ddof=1), 3)
                test_loss_avg_standard_error = round(stats.sem(a=ts_loss_array_for_avg, ddof=1), 3)

                print "test_loss_avg: " + str(test_loss_avg)
                print "test_loss_avg_var: " + str(test_loss_avg_var)
                print "test_loss_avg_standard_deviation: " + str(test_loss_avg_standard_deviation)
                print "test_loss_avg_standard_error: " + str(test_loss_avg_standard_error)

                ANN_file_get_stats_from_classifiers.write("test_loss_avg: " + str(test_loss_avg) + "\n")
                ANN_file_get_stats_from_classifiers.write("test_loss_avg_var: " + str(test_loss_avg_var) + "\n")
                ANN_file_get_stats_from_classifiers.write(
                    "test_loss_avg_standard_deviation: " + str(test_loss_avg_standard_deviation) + "\n")
                ANN_file_get_stats_from_classifiers.write(
                    "test_loss_avg_standard_error: " + str(test_loss_avg_standard_error) + "\n")

                ts_accuracy_array_for_avg = np.array(ts_accuracy_list_for_avg)

                ts_accuracy_avg = round(np.mean(ts_accuracy_array_for_avg, axis=0), 3)
                ts_accuracy_avg_var = round(np.var(a=ts_accuracy_array_for_avg, ddof=1), 3)
                ts_accuracy_avg_standard_deviation = round(np.std(a=ts_accuracy_array_for_avg, ddof=1), 3)
                ts_accuracy_avg_standard_error = round(stats.sem(a=ts_accuracy_array_for_avg, ddof=1), 3)

                print "ts_accuracy_avg: " + str(ts_accuracy_avg)
                print "ts_accuracy_avg_var: " + str(ts_accuracy_avg_var)
                print "ts_accuracy_avg_standard_deviation: " + str(ts_accuracy_avg_standard_deviation)
                print "ts_accuracy_avg_standard_error: " + str(ts_accuracy_avg_standard_error)

                ANN_file_get_stats_from_classifiers.write("ts_accuracy_avg: " + str(ts_accuracy_avg) + "\n")
                ANN_file_get_stats_from_classifiers.write("ts_accuracy_avg_var: " + str(ts_accuracy_avg_var) + "\n")
                ANN_file_get_stats_from_classifiers.write(
                    "ts_accuracy_avg_standard_deviation: " + str(ts_accuracy_avg_standard_deviation) + "\n")
                ANN_file_get_stats_from_classifiers.write(
                    "ts_accuracy_avg_standard_error: " + str(ts_accuracy_avg_standard_error) + "\n")

                ts_my_accuracy_array_for_avg = np.array(ts_my_accuracy_list_for_avg)
                ts_my_accuracy_avg = round(np.mean(ts_my_accuracy_array_for_avg, axis=0), 3)
                ts_my_accuracy_avg_var = round(np.var(a=ts_my_accuracy_array_for_avg, ddof=1), 3)
                ts_my_accuracy_avg_standard_deviation = round(np.std(a=ts_my_accuracy_array_for_avg, ddof=1), 3)
                ts_my_accuracy_avg_standard_error = round(stats.sem(a=ts_my_accuracy_array_for_avg, ddof=1), 3)

                print "ts_my_accuracy_avg: " + str(ts_my_accuracy_avg)
                print "ts_my_accuracy_avg_var: " + str(ts_my_accuracy_avg_var)
                print "ts_my_accuracy_avg_standard_deviation: " + str(ts_my_accuracy_avg_standard_deviation)
                print "ts_my_accuracy_avg_standard_error: " + str(ts_my_accuracy_avg_standard_error)

                ANN_file_get_stats_from_classifiers.write("ts_my_accuracy_avg: " + str(ts_my_accuracy_avg) + "\n")
                ANN_file_get_stats_from_classifiers.write("ts_my_accuracy_avg_var: " + str(ts_my_accuracy_avg_var) + "\n")
                ANN_file_get_stats_from_classifiers.write(
                    "ts_my_accuracy_avg_standard_deviation: " + str(ts_my_accuracy_avg_standard_deviation) + "\n")
                ANN_file_get_stats_from_classifiers.write(
                    "ts_my_accuracy_avg_standard_error: " + str(ts_my_accuracy_avg_standard_error) + "\n")

        ANN_file_get_stats_from_classifiers.close()
