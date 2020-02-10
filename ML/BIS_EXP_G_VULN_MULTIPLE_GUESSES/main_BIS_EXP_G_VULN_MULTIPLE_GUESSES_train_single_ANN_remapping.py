"""
python main_BIS_EXP_G_VULN_MULTIPLE_GUESSES_train_single_classifier.py -lr 0.001 -hlc 1 -hnc [50] -e 1000 -bs 128 --id_gpu 0 --perc_gpu 0.3 -nu 0.02 -tr_size 10000 -val_size 1000 -ts_size 10000 -tr_iter 0 -mn model_name
"""
import sys
import pandas as pn
from sklearn import preprocessing
from utilities_pckg import utilities
from keras.utils import to_categorical
from ANN_estimation import secrets_classifier
from utilities_pckg.runtime_error_handler import exception_call as excpt_cll

NU = 0.02
TRAINING_SIZE = 10000
VALIDATION_SIZE = 1000
TEST_SIZE = 10000
TRAINING_ITERATION = 0
LEARNING_RATE = 0.01
HIDDEN_LAYERS_CARD = 1
HIDDEN_NEAURONS_CARD = [50]
EPOCHS = 10
BATCH_SIZE = 512
NUMBER_OF_GUESSES = 2
ID_GPU = "0"
PERC_GPU = 0.3
MODEL_NAME = ""
NUM_CLASSES = 45


def read_command_line_options():
    thismodule = sys.modules[__name__]

    for idx, kez_val in enumerate(sys.argv, 0):
        if kez_val in ['--NU', '-nu'] and len(sys.argv) > idx + 1:
            try:
                thismodule.NU = float(sys.argv[idx + 1].strip())
            except ValueError as val_err:
                excpt_cll(idx=idx, kez_val=kez_val)

        if kez_val in ['--model_name', '-mn'] and len(sys.argv) > idx + 1:
            try:
                thismodule.MODEL_NAME = sys.argv[idx + 1].strip()
            except ValueError as val_err:
                excpt_cll(idx=idx, kez_val=kez_val)

        if kez_val in ['--training_size', '-tr_size'] and len(sys.argv) > idx + 1:
            try:
                thismodule.TRAINING_SIZE = int(sys.argv[idx + 1].strip())
            except ValueError as val_err:
                excpt_cll(idx=idx, kez_val=kez_val)

        if kez_val in ['--validation_size', '-val_size'] and len(sys.argv) > idx + 1:
            try:
                thismodule.VALIDATION_SIZE = int(sys.argv[idx + 1].strip())
            except ValueError as val_err:
                excpt_cll(idx=idx, kez_val=kez_val)

        if kez_val in ['--test_size', '-ts_size'] and len(sys.argv) > idx + 1:
            try:
                thismodule.TEST_SIZE = int(sys.argv[idx + 1].strip())
            except ValueError as val_err:
                excpt_cll(idx=idx, kez_val=kez_val)

        if kez_val in ['--training_iteration', '-tr_iter'] and len(sys.argv) > idx + 1:
            try:
                thismodule.TRAINING_ITERATION = int(sys.argv[idx + 1].strip())
            except ValueError as val_err:
                excpt_cll(idx=idx, kez_val=kez_val)

        if kez_val in ['--learning_rate', '-lr'] and len(sys.argv) > idx + 1:
            try:
                thismodule.LEARNING_RATE = float(sys.argv[idx + 1].strip())
            except ValueError as val_err:
                excpt_cll(idx=idx, kez_val=kez_val)

        if kez_val in ['--hidden_layers_card', '-hlc'] and len(sys.argv) > idx + 1:
            try:
                thismodule.HIDDEN_LAYERS_CARD = int(sys.argv[idx + 1].strip())
            except ValueError as val_err:
                excpt_cll(idx=idx, kez_val=kez_val)

        if kez_val in ['--hidden_neurons_card', '-hnc'] and len(sys.argv) > idx + 1:
            #   it must be something like -hnc [1,2,3]
            string_to_be_adapted = sys.argv[idx + 1].strip()
            print string_to_be_adapted
            if string_to_be_adapted[0] != '[' or string_to_be_adapted[-1] != ']':
                excpt_cll(idx=idx, kez_val=kez_val)
            else:
                strip_string_to_be_adapted = string_to_be_adapted[1:-1]
                split_list = strip_string_to_be_adapted.split(',')
                HIDDEN_NEAURONS_CARD_ = []
                for item in split_list:
                    try:
                        HIDDEN_NEAURONS_CARD_.append(int(item))
                    except ValueError as val_err:
                        excpt_cll(idx=idx, kez_val=kez_val)
                thismodule.HIDDEN_NEAURONS_CARD = HIDDEN_NEAURONS_CARD_

        if kez_val in ['--epochs', '-e'] and len(sys.argv) > idx + 1:
            try:
                thismodule.EPOCHS = int(sys.argv[idx + 1].strip())
            except ValueError as val_err:
                excpt_cll(idx=idx, kez_val=kez_val)

        if kez_val in ['--batch_size', '-bs'] and len(sys.argv) > idx + 1:
            if sys.argv[idx + 1].strip() == 'all' or sys.argv[idx + 1].strip() == "ALL":
                thismodule.BATCH_SIZE = None
            else:
                try:
                    thismodule.BATCH_SIZE = int(sys.argv[idx + 1].strip())
                except ValueError as val_err:
                    excpt_cll(idx=idx, kez_val=kez_val)

        if kez_val in ['--id_gpu'] and len(sys.argv) > idx + 1:
            try:
                thismodule.ID_GPU = sys.argv[idx + 1].strip()
            except ValueError as val_err:
                excpt_cll(idx=idx, kez_val=kez_val)

        if kez_val in ['--perc_gpu'] and len(sys.argv) > idx + 1:
            try:
                thismodule.PERC_GPU = float(sys.argv[idx + 1].strip())
            except ValueError as val_err:
                excpt_cll(idx=idx, kez_val=kez_val)


def main_BIS_EXP_G_VULN_MULTIPLE_GUESSES_train_single_ANN_remapping():
    read_command_line_options()

    thismodule = sys.modules[__name__]

    BIS_EXP_G_VULN_MULTIPLE_GUESSES_FOLDER = "/home/comete/mromanel/MILES_EXP/BIS_EXP_G_VULN_MULTIPLE_GUESSES/"
    utilities.createFolder(BIS_EXP_G_VULN_MULTIPLE_GUESSES_FOLDER)

    RESULT_FOLDER = BIS_EXP_G_VULN_MULTIPLE_GUESSES_FOLDER + "RESULT_FOLDER_REMAPPING/"
    utilities.createFolder(RESULT_FOLDER)

    result_folder = RESULT_FOLDER + MODEL_NAME + "/"
    utilities.createFolder(result_folder)

    result_folder = result_folder + str(TRAINING_SIZE) + "_training_size_and_" + str(
        VALIDATION_SIZE) + "_validation_size_iteration_" + str(TRAINING_ITERATION) + "/"
    utilities.createFolder(result_folder)

    DATA_FOLDER = BIS_EXP_G_VULN_MULTIPLE_GUESSES_FOLDER + "DATA_FOLDER/"

    ANN_data_folder = DATA_FOLDER + str(TRAINING_SIZE) + "_training_and_" + str(
        VALIDATION_SIZE) + "_validation_store_folder_train_iteration_" + str(TRAINING_ITERATION)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  load datasets  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%$$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    print "\n\n\nDATA ARE LOADED FROM ", ANN_data_folder, "\n\n\n"

    log_file = open(result_folder + "/log_file.txt", "wa")
    log_file.write("\n\n\nDATA ARE LOADED FROM " + ANN_data_folder + "\n\n\n")
    log_file.close()

    training_set = pn.read_pickle(path=ANN_data_folder + "/training_set.pkl")
    O_train = training_set[:, 0]
    S_train = training_set[:, 1]
    Z_train = training_set[:, 2]
    Z_train_enc = to_categorical(y=Z_train, num_classes=NUM_CLASSES)

    val_set = pn.read_pickle(path=ANN_data_folder + "/validation_set.pkl")
    O_val = val_set[:, 0]
    S_val = val_set[:, 1]
    Z_val = val_set[:, 2]
    Z_val_enc = to_categorical(y=Z_val, num_classes=NUM_CLASSES)

    """O_train = preprocess.scaler_between_minus_one_and_one(column=O_train, min_column=MIN_OBSERVABLE,
                                                          max_column=MAX_OBSERVABLE)

    O_val = preprocess.scaler_between_minus_one_and_one(column=O_val, min_column=MIN_OBSERVABLE,
                                                        max_column=MAX_OBSERVABLE)"""

    min_max_scaler = preprocessing.MinMaxScaler()

    O_train = O_train.reshape(-1, 1)
    O_train = min_max_scaler.fit_transform(O_train)

    O_val = O_val.reshape(-1, 1)
    O_val = min_max_scaler.transform(O_val)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ANN: instantiate, train, evaluate %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%$$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if len(O_train.shape) == 1:
        input_x_dimension = 1
    else:
        input_x_dimension = O_train.shape[1]

    if thismodule.BATCH_SIZE is None:
        thismodule.BATCH_SIZE = O_train.shape[0]

    secrets_classifier_manager = secrets_classifier.ClassifierNetworkManager(
        number_of_classes=Z_train_enc.shape[1],
        learning_rate=LEARNING_RATE,
        hidden_layers_card=HIDDEN_LAYERS_CARD,
        hidden_neurons_card=HIDDEN_NEAURONS_CARD,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        id_gpu=ID_GPU,
        perc_gpu=PERC_GPU,
        input_x_dimension=input_x_dimension)

    secrets_classifier_manager.train_classifier_net(training_set=O_train,
                                                    training_supervision=Z_train_enc,
                                                    validation_set=O_val,
                                                    validation_supervision=Z_val_enc,
                                                    results_folder=result_folder)
