from utilities import plot_ANN_classifier

MILES_EXP = '/home/comete/mromanel/MILES_EXP/EXP_G_VULN_DP_FOLDER/RESULT_FOLDER_REMAPPING/'
MODEL = 'model_200/'
TR_TS_SIZE = 10000
VAL_SIZE = 1000
ITER = 0


def main_plot_ANN(save_plot):
    plot_ANN_classifier.plot_standalone_classifier_network(
        remote_directory_absolute_path=MILES_EXP + MODEL + str(TR_TS_SIZE) + "_training_size_and_" + str(
            VAL_SIZE) + "_validation_size_iteration_" + str(ITER),
        local_directory_absolute_path="/Users/marcoromanelli/Desktop/results_classifier/",
        save_plot=save_plot)
