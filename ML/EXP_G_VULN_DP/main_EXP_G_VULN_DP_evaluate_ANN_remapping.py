import os
import sys
import numpy as np
import pandas as pn
from sklearn import preprocessing
from keras.models import load_model
from utilities_pckg import gpu_setup
from utilities_pckg import g_vuln_computation, utilities, preprocess
from utilities_pckg.runtime_error_handler import exception_call as excpt_cll

# MIN_OBSERVABLE = 0
# MAX_OBSERVABLE = 15999
APPROACHES = ['ANN']
NU = 0.002
TRAINING_SET_SIZE = [10000, 50000]
# TRAINING_SET_SIZE = [int(sys.argv[1])]
TEST_SET_SIZE = [10000, 50000]
# TEST_SET_SIZE = [int(sys.argv[3])]
VALIDATION_SET_SIZE = [1000, 5000]
# VALIDATION_SET_SIZE = [int(sys.argv[2])]
# TEST_ITERATIONS = 1  # 100
TEST_ITERATIONS_BEG = 0
TEST_ITERATIONS_END = 50
TRAIN_ITERATIONS = 5
MODEL_NAME = ""
ID_GPU = "0"
PERC_GPU = 0.3

RESULT_FOLDER = "/home/comete/mromanel/MILES_EXP/EXP_G_VULN_DP_FOLDER/RESULT_FOLDER_REMAPPING/"

DATA_FOLDER = "/home/comete/mromanel/MILES_EXP/EXP_G_VULN_DP_FOLDER/DATA_FOLDER_AFTER_OUR_PREPROCESSING/"
DATA_FOLDER_TEST = "/home/comete/mromanel/MILES_EXP/EXP_G_VULN_DP_FOLDER/DATA_FOLDER/"

G_MATRIX_PATH = '/home/comete/mromanel/MILES_EXP/EXP_G_VULN_DP_FOLDER/G_OBJ/g_mat.pkl'
G_MATRIX_ROWS_PATH = '/home/comete/mromanel/MILES_EXP/EXP_G_VULN_DP_FOLDER/G_OBJ/g_mat_rows.pkl'
G_MATRIX_COLS_PATH = '/home/comete/mromanel/MILES_EXP/EXP_G_VULN_DP_FOLDER/G_OBJ/g_mat_cols.pkl'


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


def main_EXP_G_VULN_DP_evaluate_ANN_remapping():
    read_command_line_options()

    gpu_setup.gpu_setup(id_gpu=ID_GPU, memory_percentage=PERC_GPU)

    if len(TEST_SET_SIZE) != len(TRAINING_SET_SIZE) or len(VALIDATION_SET_SIZE) != len(TRAINING_SET_SIZE):
        sys.exit("The set size lists must all contain the same amount of items.")

    loaded_g_matrix = pn.read_pickle(path=G_MATRIX_PATH)
    loaded_g_matrix_rows = pn.read_pickle(path=G_MATRIX_ROWS_PATH)
    loaded_g_matrix_cols = pn.read_pickle(path=G_MATRIX_COLS_PATH)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ANN approach  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%$$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if 'ANN' in APPROACHES:
        print("\n####################################################################################")
        print("###################################  ANN approach  ##################################")
        print("####################################################################################\n")

        #   store the values of the error estimations via all the possible methods
        ANN_Rf_values = []

        #   number of test samples
        number_of_test_samples = []

        #   number of training samples
        number_of_training_samples = []

        #   iterator over different training sets iterations
        training_iteration = []

        #   iterator over different test sets iterations
        test_iteration = []

        for size_list_iterator in range(len(TRAINING_SET_SIZE)):

            #   select the current values for the sizes (useful to keep track in the names of the
            training_set_size = TRAINING_SET_SIZE[size_list_iterator]
            validation_set_size = VALIDATION_SET_SIZE[size_list_iterator]
            test_set_size = TEST_SET_SIZE[0]

            for train_iteration in range(TRAIN_ITERATIONS):
                filepath = RESULT_FOLDER + MODEL_NAME + "/" + str(training_set_size) + "_training_size_and_" + str(
                    validation_set_size) + "_validation_size_iteration_" + str(train_iteration)

                if not os.path.exists(filepath):
                    continue

                else:
                    ANN_model = load_model(filepath=filepath + "/classifier_net_model")
                    # print ANN_model.summary()

                training_set = pn.read_pickle(DATA_FOLDER + str(training_set_size) + "_training_and_" + str(
                    validation_set_size) + "_validation_and_" + str(
                    test_set_size) + "_test_store_folder_train_iteration_" + str(
                    train_iteration) + "/training_set.pkl").values

                min_tr = np.min(training_set[:, 0:training_set.shape[1] - 2])
                max_tr = np.max(training_set[:, 0:training_set.shape[1] - 2])

                print "\n\n\n\n\n\n\n\n#################################  test_size: " + str(
                    test_set_size) + " ################################"
                g_vuln_ANN_file = open(filepath + "/ANN_" + str(training_set_size) + "_training_and_" + str(
                    validation_set_size) + "_validation_file_R_estimate_iteration_" + str(train_iteration) + "_" + str(
                    test_set_size) + "_test_set_size_test_iter_up_to_" + str(TEST_ITERATIONS_END) + ".txt", "wa")

                for test_iterator in range(TEST_ITERATIONS_BEG, TEST_ITERATIONS_END):
                    print "\n\n\n#################################  test_set_" + str(
                        test_iterator) + " ################################"

                    g_vuln_ANN_file.write("\n\n\n#################################  test_set_" + str(
                        test_iterator) + " ################################")

                    test_set = pn.read_pickle(
                        path=DATA_FOLDER_TEST + str(test_set_size) + "_size_test_set/test_set_" + str(
                            test_iterator) + ".pkl")

                    # X_test = test_set[:, 0:test_set.shape[1] - 2]
                    # y_test = test_set[:, -2]
                    # z_test = test_set[:, -1]

                    X_test = test_set[:, 0:test_set.shape[1] - 1]
                    y_test = test_set[:, -1]

                    dt = np.dtype((np.void, X_test.dtype.itemsize * X_test.shape[1]))
                    b = np.ascontiguousarray(X_test).view(dt)
                    X_test_unique, X_test_unique_cnt = np.unique(b, return_counts=True)
                    X_test_unique = X_test_unique.view(X_test.dtype).reshape(-1, X_test.shape[1])

                    print X_test
                    print max_tr, min_tr
                    X_test_preprocessed = preprocess.scaler_zero_one_all_cols(data=X_test, max_=max_tr, min_=min_tr)
                    print X_test_preprocessed

                    X_test_preprocessed_unique = preprocess.scaler_zero_one_all_cols(data=X_test_unique,
                                                                                     max_=max_tr, min_=min_tr)

                    # if len(X_test_preprocessed_unique) != len(np.unique(X_test_preprocessed_unique)):
                    #     sys.exit("The preprocessing created some collision which might affect the computation")

                    # print X_test_preprocessed_unique

                    # new_old_obs = {}
                    # for i in range(len(X_test_preprocessed_unique)):
                    #     new_old_obs[X_test_preprocessed_unique[i][0]] = X_test_unique[i][0]
                    # # print new_old_obs

                    ###########################################################  Prediction

                    print "X_test_preprocessed: ", X_test_preprocessed.shape
                    print "y_test.shape: ", y_test.shape

                    ANN_prediction_test = []

                    pred = ANN_model.predict(x=X_test_preprocessed)

                    print pred

                    for row_iter in range(pred.shape[0]):
                        ANN_prediction_test.append(np.argmax(pred[row_iter, :]))

                    ANN_prediction_test = np.array(ANN_prediction_test).reshape(len(ANN_prediction_test), 1)

                    final_matrix = np.column_stack((X_test, y_test))
                    final_matrix = np.column_stack(
                        (final_matrix, ANN_prediction_test))

                    g_vuln_ANN = g_vuln_computation.compute_g_vuln_with_remapping_multidimesional_inputs(
                        final_mat=final_matrix,
                        g_mat=loaded_g_matrix,
                        g_mat_rows=loaded_g_matrix_rows,
                        g_mat_cols=loaded_g_matrix_cols
                    )

                    print("\ng_vuln_ANN = " + str(g_vuln_ANN))

                    g_vuln_ANN_file.write("\ng_vuln_ANN_file = " + str(g_vuln_ANN))
                    ANN_Rf_values.append(g_vuln_ANN)
                    number_of_test_samples.append(test_set_size)
                    number_of_training_samples.append(training_set_size)
                    training_iteration.append(train_iteration)
                    test_iteration.append(test_iterator)

                    # ###########################################################  Accuracy computation
                    #
                    # accuracy = round(utilities.compute_accuracy(y_classes=z_test,
                    #                                             y_pred_classes=ANN_prediction_test), 3)
                    #
                    # print "\nAccuracy ---> ", accuracy
                    # g_vuln_ANN_file.write("\nAccuracy --->" + str(accuracy))
                    #
                    # ###########################################################  Accuracy computation (tf fashion)
                    #
                    # accuracy_tf_fashion = round(utilities.compute_accuracy_tf_fashion(y_classes=z_test,
                    #                                                                   y_pred_classes=
                    #                                                                   ANN_prediction_test), 3)
                    #
                    # print "\nAccuracy tf fashion ---> ", accuracy_tf_fashion
                    #
                    # g_vuln_ANN_file.write("\nAccuracy tf fashion ---> " + str(accuracy_tf_fashion))
                    #
                    # ###########################################################  Precision computation
                    #
                    # precision = utilities.compute_precision(y_classes=z_test,
                    #                                         y_pred_classes=ANN_prediction_test)
                    #
                    # print "\nPrecision ---> ", precision
                    #
                    # g_vuln_ANN_file.write("\nPrecision ---> " + str(precision))
                    #
                    # ###########################################################  Recall computation
                    #
                    # recall = utilities.compute_recall(y_classes=z_test,
                    #                                   y_pred_classes=ANN_prediction_test)
                    #
                    # print "\nRecall ---> ", recall
                    #
                    # g_vuln_ANN_file.write("\nRecall ---> " + str(recall))
                    #
                    # ###########################################################  F1_score computation
                    #
                    # F1_score = utilities.compute_f1_score(y_classes=y_test,
                    #                                       y_pred_classes=ANN_prediction_test)
                    #
                    # print "\nF1_score ---> ", F1_score
                    #
                    # g_vuln_ANN_file.write("\nF1_score ---> " + str(F1_score))

                g_vuln_ANN_file.close()

        ANN_Rf_values = np.array(ANN_Rf_values, dtype=np.float64)
        number_of_test_samples = np.array(number_of_test_samples, dtype=np.int32)
        number_of_training_samples = np.array(number_of_training_samples, dtype=np.int32)
        training_iteration = np.array(training_iteration, dtype=np.int32)
        test_iteration = np.array(test_iteration, dtype=np.int32)

        result_matrix = np.column_stack((ANN_Rf_values, number_of_test_samples))
        # print result_matrix.shape

        result_matrix = np.column_stack((result_matrix, number_of_training_samples))
        # print result_matrix.shape

        result_matrix = np.column_stack((result_matrix, training_iteration))
        # print result_matrix.shape

        result_matrix = np.column_stack((result_matrix, test_iteration))
        # print result_matrix.shape

        result_df = pn.DataFrame(data=result_matrix,
                                 columns=["ANN_Rf_values", "number_of_test_samples",
                                          "number_of_training_samples",
                                          "train_iteration",
                                          "test_iteration"])
        result_df.to_pickle(
            path=RESULT_FOLDER + MODEL_NAME + "/ANN_training_and_validation_result_df_train_size_" + str(
                TRAINING_SET_SIZE[0]) + "_up_to_test_iter_" + str(TEST_ITERATIONS_END) + ".pkl")
