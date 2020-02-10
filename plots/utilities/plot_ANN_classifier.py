from pylab import *
from utilities.load_pickled_result_file import load_pickled_result_file
import os
import shutil


def plot_standalone_classifier_network(remote_directory_absolute_path, local_directory_absolute_path, save_plot=False):
    if save_plot:
        dpi = 1000
    else:
        dpi = 150

    if not os.path.isdir(local_directory_absolute_path):
        command = 'mkdir ' + local_directory_absolute_path
        os.system(command)

    command = 'scp -r mromanel@selene.saclay.inria.fr:' + remote_directory_absolute_path + '/*.pkl ' + \
              local_directory_absolute_path

    print(command)

    os.system(command)
    print("Download is over")

    #   plot_list = [classifier_network_categ_acc, classifier_network_loss, f1_values]
    plot_list = [True, True, True]

    ####################################################################################################################
    #########################################  Plot classifier_network_categ_acc  ######################################
    ####################################################################################################################
    if plot_list[0] is True:
        classifier_network_training_accuracy_array = load_pickled_result_file(
            local_directory_absolute_path + "classifier_network_categ_acc_vec.pkl")

        classifier_network_validation_accuracy_array = load_pickled_result_file(
            local_directory_absolute_path + "classifier_network_val_categ_acc_vec.pkl")

        """classifier_network_test_accuracy_array = load_pickled_result_file(
            local_directory_absolute_path + "classifier_network_evaluation_on_test_set_accuracy_vec.pkl")"""

        classifier_network_array_epochs = load_pickled_result_file(
            local_directory_absolute_path + "classifier_network_epochs.pkl")

        x = []
        y = classifier_network_training_accuracy_array

        for i in range(1, len(y) + 1):
            x.append(i)

        figure("classifier network accuracy", dpi=dpi)
        plot(x, y, 'blue', )

        y = classifier_network_validation_accuracy_array

        plot(x, y, 'green')

        """y = classifier_network_test_accuracy_array

        plot(x, y, 'red')"""

        xlabel("# Epochs")
        ylabel("Classifier accuracy")

        legend(['Training data', 'Validation data', 'Test data'])

        cont = 0
        for i in range(0, len(classifier_network_array_epochs)):
            cont += classifier_network_array_epochs[i]
            # axvline(x=cont, linewidth=1, color='red', linestyle='dashed')
            # plt.text(cont, 0, 'iteration ' + str(i), rotation=90, size='smaller')
        xlim(1, cont)
        ylim(0, 1)
        plt.grid()

        if save_plot:
            plt.savefig("/Users/marcoromanelli/Desktop/classifier_network_accuracy.pdf")

    ####################################################################################################################
    ##########################################  Plot classifier_network_loss  ##########################################
    ####################################################################################################################
    if plot_list[1] is True:
        classifier_network_training_loss_array = load_pickled_result_file(
            local_directory_absolute_path + 'classifier_network_loss_vec.pkl')

        classifier_network_validation_loss_array = load_pickled_result_file(
            local_directory_absolute_path + 'classifier_network_val_loss_vec.pkl')

        """classifier_network_test_loss_array = load_pickled_result_file(
            local_directory_absolute_path + 'classifier_network_evaluation_on_test_set_loss_vec.pkl')"""

        classifier_network_array_epochs = load_pickled_result_file(
            local_directory_absolute_path + "classifier_network_epochs.pkl")

        y_max = -float("inf")
        y_min = float("inf")

        x = []
        y = classifier_network_training_loss_array

        for i in range(1, len(y) + 1):
            x.append(i)

        figure("classifier network loss", dpi=dpi)
        plot(x, y, 'blue')
        if max(y) > y_max:
            y_max = max(y)
        if min(y) < y_min:
            y_min = min(y)

        y = classifier_network_validation_loss_array
        plot(x, y, 'green')
        if max(y) > y_max:
            y_max = max(y)
        if min(y) < y_min:
            y_min = min(y)

        """y = classifier_network_test_loss_array
        plot(x, y, 'red')
        if max(y) > y_max:
            y_max = max(y)
        if min(y) < y_min:
            y_min = min(y)"""

        xlabel("# Epochs")
        ylabel("Classification loss: cross-entropy")
        # title("Discriminator_loss")
        legend(['Training data', 'Validation data'])
        cont = 0
        for i in range(0, len(classifier_network_array_epochs)):
            cont += classifier_network_array_epochs[i]
            # axvline(x=cont, linewidth=1, color='red', linestyle='dashed')
            # plt.text(cont, 0, 'iteration ' + str(i), rotation=90, size='smaller')
        xlim(1, cont)
        y_min = y_min - (0.10 * y_min)
        y_max = y_max + (0.10 * y_max)
        ylim(y_min, y_max)
        plt.grid()

        if save_plot:
            plt.savefig("/Users/marcoromanelli/Desktop/classifier_network_loss.pdf")

    ####################################################################################################################
    ##########################################  Plot classifier_network_f1_scores  #####################################
    ####################################################################################################################

    if plot_list[2] is True:
        classifier_network_f1_value_training_vec = load_pickled_result_file(
            local_directory_absolute_path + 'f1_value_training_vec.pkl')

        classifier_network_f1_value_validation_vec = load_pickled_result_file(
            local_directory_absolute_path + 'f1_value_validation_vec.pkl')

        """classifier_network_f1_value_test_vec = load_pickled_result_file(
            local_directory_absolute_path + 'f1_value_test_vec.pkl')"""

        classifier_network_array_epochs = load_pickled_result_file(
            local_directory_absolute_path + "classifier_network_epochs.pkl")

        y_max = -float("inf")
        y_min = float("inf")

        x = []
        y = classifier_network_f1_value_training_vec

        for i in range(1, len(y) + 1):
            x.append(i)

        figure("classifier network f1_scores", dpi=dpi)
        plot(x, y, 'blue')
        if max(y) > y_max:
            y_max = max(y)
        if min(y) < y_min:
            y_min = min(y)

        y = classifier_network_f1_value_validation_vec
        plot(x, y, 'green')
        if max(y) > y_max:
            y_max = max(y)
        if min(y) < y_min:
            y_min = min(y)

        """y = classifier_network_f1_value_test_vec
        plot(x, y, 'red')
        if max(y) > y_max:
            y_max = max(y)
        if min(y) < y_min:
            y_min = min(y)"""

        xlabel("# Epochs")
        ylabel("f1_scores")
        # title("Discriminator_loss")
        legend(['Training data', 'Validation data'])
        cont = 0
        for i in range(0, len(classifier_network_array_epochs)):
            cont += classifier_network_array_epochs[i]
            # axvline(x=cont, linewidth=1, color='red', linestyle='dashed')
            # plt.text(cont, 0, 'iteration ' + str(i), rotation=90, size='smaller')
        xlim(1, cont)
        y_min = y_min - (0.10 * y_min)
        y_max = y_max + (0.10 * y_max)
        ylim(y_min, 1)
        plt.grid()

        if save_plot:
            plt.savefig("/Users/marcoromanelli/Desktop/classifier_network_f1_score.pdf")

    if save_plot is False:
        plt.show()

    shutil.rmtree("/Users/marcoromanelli/Desktop/results_classifier")
