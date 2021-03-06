from utilities import plot_ANN_classifier

MILES_EXP = '/home/comete/mromanel/MILES_EXP/BIS_EXP_GEO_LOCATION_QIF_LIB_SETTING/RESULT_FOLDER_REMAPPING/'
MODEL = 'model_100_300_500/'
TR_TS_SIZE = 1000
VAL_SIZE = 100
ITER = 2


def main_plot_ANN(save_plot):
    plot_ANN_classifier.plot_standalone_classifier_network(
        remote_directory_absolute_path=MILES_EXP + MODEL + str(TR_TS_SIZE) + "_training_size_and_" + str(
            VAL_SIZE) + "_validation_size_iteration_" + str(ITER),
        local_directory_absolute_path="/Users/marcoromanelli/Desktop/results_classifier/",
        save_plot=save_plot)
