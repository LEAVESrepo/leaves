import numpy as np
import pandas as pn
import sys
from channel_estimation import KNN_express
from utilities_pckg.runtime_error_handler import exception_call as excpt_cll
from utilities_pckg import utilities
from utilities_pckg import g_vuln_computation
import time

APPROACHES = ['KNN']
TRAINING_SET_SIZE = [10000, 30000, 50000]
# TRAINING_SET_SIZE = [int(sys.argv[1])]
TEST_SET_SIZE = [50000]
# TEST_SET_SIZE = [int(sys.argv[3])]
VALIDATION_SET_SIZE = [1000, 3000, 5000]
# VALIDATION_SET_SIZE = [int(sys.argv[2])]
# TEST_ITERATIONS = 1  # 100
TEST_ITERATIONS_BEG = 0
TEST_ITERATIONS_END = 50
TRAIN_ITERATIONS = 5
SECRETS_CARD = 10
ENHANCED = False

RESULT_FOLDER = "/home/comete/mromanel/MILES_EXP/BIS_EXP_G_VULN_DP_FOLDER/RESULT_FOLDER_REMAPPING/"
utilities.createFolder(RESULT_FOLDER)

RESULT_FOLDER = "/home/comete/mromanel/MILES_EXP/BIS_EXP_G_VULN_DP_FOLDER/RESULT_FOLDER_REMAPPING/KNN/"
utilities.createFolder(RESULT_FOLDER)

DATA_FOLDER = "/home/comete/mromanel/MILES_EXP/BIS_EXP_G_VULN_DP_FOLDER/DATA_FOLDER/"
DATA_FOLDER_TEST = "/home/comete/mromanel/MILES_EXP/BIS_EXP_G_VULN_DP_FOLDER/DATA_FOLDER/" + str(
    TEST_SET_SIZE[0]) + "_size_test_set/"

G_MATRIX_PATH = '/home/comete/mromanel/MILES_EXP/BIS_EXP_G_VULN_DP_FOLDER/G_OBJ/g_mat.pkl'
G_MATRIX_ROWS_PATH = '/home/comete/mromanel/MILES_EXP/BIS_EXP_G_VULN_DP_FOLDER/G_OBJ/g_mat_rows.pkl'
G_MATRIX_COLS_PATH = '/home/comete/mromanel/MILES_EXP/BIS_EXP_G_VULN_DP_FOLDER/G_OBJ/g_mat_cols.pkl'


def read_command_line_options():
    thismodule = sys.modules[__name__]
    for idx, key_val in enumerate(sys.argv, 0):

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

        if key_val in ['--enhanced', 'enh'] and len(sys.argv) > idx + 1:
            if sys.argv[idx + 1].strip() == 'True':
                try:
                    thismodule.ENHANCED = True
                except ValueError as val_err:
                    excpt_cll(idx=idx, key_val=key_val)
            elif sys.argv[idx + 1].strip() == 'False':
                try:
                    thismodule.ENHANCED = False
                except ValueError as val_err:
                    excpt_cll(idx=idx, key_val=key_val)
            else:
                excpt_cll(idx="invalid bool", key_val=key_val)

        if key_val in ['--end_test_it'] and len(sys.argv) > idx + 1:
            try:
                thismodule.TEST_ITERATIONS_END = int(sys.argv[idx + 1].strip())
            except ValueError as val_err:
                excpt_cll(idx=idx, key_val=key_val)


def main_BIS_EXP_G_VULN_DP_evaluate_KNN_remapping():
    read_command_line_options()

    if len(TEST_SET_SIZE) != len(TRAINING_SET_SIZE) or len(VALIDATION_SET_SIZE) != len(TRAINING_SET_SIZE):
        sys.exit("The set size lists must all contain the same amount of items.")

    loaded_g_matrix = pn.read_pickle(path=G_MATRIX_PATH)
    loaded_g_matrix_rows = pn.read_pickle(path=G_MATRIX_ROWS_PATH)
    loaded_g_matrix_cols = pn.read_pickle(path=G_MATRIX_COLS_PATH)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  KNN approach  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%$$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if 'KNN' in APPROACHES:
        print("\n####################################################################################")
        print("###################################  KNN approach  ##################################")
        print("####################################################################################\n")

        #   store the values of the error estimations via all the possible methods
        KNN_Rf_values = []

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
            test_set_size = TEST_SET_SIZE[size_list_iterator]

            for train_iteration in range(TRAIN_ITERATIONS):
                training_data = pn.read_pickle(DATA_FOLDER + str(training_set_size) + "_training_and_" + str(
                    validation_set_size) + "_validation_store_folder_train_iteration_" + str(
                    train_iteration) + "/training_set.pkl")

                O_train = training_data[:, 0:training_data.shape[1] - 2]
                S_train = training_data[:, -2]
                Z_train = training_data[:, -1]

                Z_train_list = []

                dt = np.dtype((np.void, O_train.dtype.itemsize * O_train.shape[1]))
                b = np.ascontiguousarray(O_train).view(dt)
                O_train_unq = np.unique(b)
                O_train_unq = O_train_unq.view(O_train.dtype).reshape(-1, O_train.shape[1])

                print O_train_unq.shape

                for ot in O_train_unq:
                    ot = tuple(ot)
                    # print ot
                    idx = np.where((O_train == ot).all(axis=1))[0]
                    z, count_z = np.unique(Z_train[idx], return_counts=True)
                    Z_train_list.append(z[np.argmax(count_z)])

                Z_train_mod = np.array(Z_train_list).reshape((O_train_unq.shape[0], 1))

                k_neighbors = int(round(np.log(len(O_train_unq)), 0))
                print "k_neighbors ---> ", k_neighbors

                knn_classifier = KNN_express.KNN_express_EstimatorManager(
                    observed_samples=np.column_stack((O_train_unq, Z_train_mod)),
                    k_neighbors=k_neighbors)

                print "\n\n\n\n\n\n\n\n#################################  test_size: " + str(
                    test_set_size) + " ################################"

                KNN_g_vuln_file = open(RESULT_FOLDER + "/KNN_" + str(training_set_size) + "_training_and_" + str(
                    validation_set_size) + "_validation_file_R_estimate_iteration_" + str(train_iteration) + "_" + str(
                    test_set_size) + "_test_set_size_test_iter_up_to_" + str(TEST_ITERATIONS_END) + ".txt", "wa")

                for test_iterator in range(TEST_ITERATIONS_BEG, TEST_ITERATIONS_END):
                    now = time.time()
                    print "\n\n\n#################################  test_set_" + str(
                        test_iterator) + " ################################"

                    KNN_g_vuln_file.write("\n\n\n#################################  test_set_" + str(
                        test_iterator) + " ################################")

                    test_set = pn.read_pickle(
                        path=DATA_FOLDER_TEST + "test_set_" + str(
                            test_iterator) + ".pkl")

                    O_test = test_set[:, 0:test_set.shape[1] - 1]
                    S_test = test_set[:, -1]
                    # Z_test = test_set[:, -1]

                    # Z_test_list = []

                    dt = np.dtype((np.void, O_test.dtype.itemsize * O_test.shape[1]))
                    b = np.ascontiguousarray(O_test).view(dt)
                    O_test_unq = np.unique(b)
                    O_test_unq = O_test_unq.view(O_test.dtype).reshape(-1, O_test.shape[1])

                    # for ot in O_test_unq:
                    #     ot = tuple(ot)
                    #     idx = np.where((O_test == ot).all(axis=1))[0]
                    #     z, count_z = np.unique(Z_test[idx], return_counts=True)
                    #     Z_test_list.append(z[np.argmax(count_z)])
                    #
                    # Z_test_mod = np.array(Z_test_list).reshape((O_test_unq.shape[0], 1))

                    ind, pred = knn_classifier.predict_KNN_KDT(
                        observables_test=O_test_unq.reshape(O_test_unq.shape[0], O_test_unq.shape[1]),
                        enhanced=ENHANCED,
                        p_=1)

                    print ind.shape

                    pred_final = knn_classifier.new_way_prediction_faster_multidimensional(ind=ind, O_train_unq=O_train_unq,
                                                                                           O_train=O_train,
                                                                                           Z_train=Z_train, O_test=O_test,
                                                                                           O_test_unq=O_test_unq)

                    print "...done predicting..."

                    # ind, pred = knn_classifier.predict_KNN_KDT(observables_test=O_test.reshape(O_test.shape[0], 1),
                    #                                            enhanced=False)

                    # pred_final = []
                    #
                    # for ob in O_test:
                    #     idx = np.where(O_test_unq == ob)[0]
                    #     pred_final.append(pred[idx[0]])

                    final_mat = np.column_stack((O_test, S_test))
                    final_mat = np.column_stack((final_mat, np.array(pred_final).reshape((O_test.shape[0], 1))))

                    Rf_KNN_g_leak = g_vuln_computation.compute_g_vuln_with_remapping_multidimesional_inputs(
                        final_mat=final_mat,
                        g_mat=loaded_g_matrix,
                        g_mat_rows=loaded_g_matrix_rows,
                        g_mat_cols=loaded_g_matrix_cols
                    )

                    print("\nRf_KNN_g_leak = " + str(Rf_KNN_g_leak))

                    KNN_g_vuln_file.write("\nANN_file_Rf_ANN_g_leak = " + str(Rf_KNN_g_leak))
                    KNN_Rf_values.append(Rf_KNN_g_leak)
                    number_of_test_samples.append(test_set_size)
                    number_of_training_samples.append(training_set_size)
                    training_iteration.append(train_iteration)
                    test_iteration.append(test_iterator)

                    ###########################################################  Accuracy computation

                    # accuracy = round(utilities.compute_accuracy(y_classes=Z_test,
                    #                                             y_pred_classes=pred_final), 3)
                    #
                    # print "\nAccuracy ---> ", accuracy
                    # KNN_g_vuln_file.write("\nAccuracy --->" + str(accuracy))
                    #
                    # ###########################################################  Accuracy computation (tf fashion)
                    #
                    # accuracy_tf_fashion = round(utilities.compute_accuracy_tf_fashion(y_classes=Z_test,
                    #                                                                   y_pred_classes=pred_final), 3)
                    #
                    # print "\nAccuracy tf fashion ---> ", accuracy_tf_fashion
                    #
                    # KNN_g_vuln_file.write("\nAccuracy tf fashion ---> " + str(accuracy_tf_fashion))
                    #
                    # ###########################################################  Precision computation
                    #
                    # precision = utilities.compute_precision(y_classes=Z_test,
                    #                                         y_pred_classes=pred_final)
                    #
                    # print "\nPrecision ---> ", precision
                    #
                    # KNN_g_vuln_file.write("\nPrecision ---> " + str(precision))
                    #
                    # ###########################################################  Recall computation
                    #
                    # recall = utilities.compute_recall(y_classes=Z_test,
                    #                                   y_pred_classes=pred_final)
                    #
                    # print "\nRecall ---> ", recall
                    #
                    # KNN_g_vuln_file.write("\nRecall ---> " + str(recall))
                    #
                    # ###########################################################  F1_score computation
                    #
                    # F1_score = utilities.compute_f1_score(y_classes=Z_test,
                    #                                       y_pred_classes=pred_final)
                    #
                    # print "\nF1_score ---> ", F1_score
                    #
                    # KNN_g_vuln_file.write("\nF1_score ---> " + str(F1_score))

                    now2 = time.time()
                    print now2 - now

                KNN_g_vuln_file.close()

        KNN_Rf_values = np.array(KNN_Rf_values, dtype=np.float64)
        number_of_test_samples = np.array(number_of_test_samples, dtype=np.int32)
        number_of_training_samples = np.array(number_of_training_samples, dtype=np.int32)
        training_iteration = np.array(training_iteration, dtype=np.int32)
        test_iteration = np.array(test_iteration, dtype=np.int32)

        result_matrix = np.column_stack((KNN_Rf_values, number_of_test_samples))
        # print result_matrix.shape

        result_matrix = np.column_stack((result_matrix, number_of_training_samples))
        # print result_matrix.shape

        result_matrix = np.column_stack((result_matrix, training_iteration))
        # print result_matrix.shape

        result_matrix = np.column_stack((result_matrix, test_iteration))
        # print result_matrix.shape

        result_df = pn.DataFrame(data=result_matrix,
                                 columns=["KNN_Rf_values", "number_of_test_samples",
                                          "number_of_training_samples",
                                          "train_iteration",
                                          "test_iteration"])
        result_df.to_pickle(
            path=RESULT_FOLDER + "/KNN_training_and_validation_result_df_train_size_" + str(
                TRAINING_SET_SIZE[0]) + "_up_to_test_iter_" + str(TEST_ITERATIONS_END) + ".pkl")
