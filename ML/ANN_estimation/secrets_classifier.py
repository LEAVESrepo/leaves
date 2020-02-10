from keras import Input, Model, optimizers
from keras.layers import Dense
from utilities_pckg import utilities, gpu_setup
import numpy as np
import pickle
from tensorflow import set_random_seed
from utilities_pckg.runtime_error_handler import runtime_error_handler as err_hndl
import inspect

np.random.seed(1234)
set_random_seed(1234)


class ClassifierNetworkManager:

    def __init__(self, number_of_classes, learning_rate, hidden_layers_card, hidden_neurons_card, epochs, batch_size,
                 id_gpu, perc_gpu, input_x_dimension):
        #   ml hyper-parameters
        self.input_x_dimension = input_x_dimension
        self.number_of_classes = number_of_classes
        self.learning_rate = learning_rate
        self.hidden_layers_card = hidden_layers_card
        self.hidden_neurons_card = hidden_neurons_card
        self.epochs = epochs
        self.batch_size = batch_size
        self.id_gpu = id_gpu
        self.perc_gpu = perc_gpu

        #   ml store vectors
        self.classifier_network_epochs = []
        self.classifier_network_loss_vec = []
        self.classifier_network_categ_acc_vec = []
        self.classifier_network_val_loss_vec = []
        self.classifier_network_val_categ_acc_vec = []
        # self.classifier_network_evaluation_on_test_set_loss_vec = []
        # self.classifier_network_evaluation_on_test_set_accuracy_vec = []
        self.f1_value_training = []
        self.f1_value_validation = []
        # self.f1_value_test = []

        self.results_folder = None

        self.check_layers_mismatch()

    def check_layers_mismatch(self):
        if self.hidden_layers_card != len(self.hidden_neurons_card):
            err_hndl(str_="hidden_layers_mismatch", add=inspect.stack()[0][3])

    def build_classifier_network(self):
        #   inputs of dimension input_x_dimension
        input_layer = Input(shape=(self.input_x_dimension,), name='classifier_network_input')

        #   at least one hidden layer
        hidden_layer = Dense(self.hidden_neurons_card[0], activation='relu', name='classifier_network_hidden_layer_0')(
            input_layer)

        #   other hidden layers if any
        for idx in range(1, self.hidden_layers_card):
            layer_name = 'classifier_network_hidden_layer_' + str(idx)
            hidden_layer = Dense(self.hidden_neurons_card[idx], activation='relu', name=layer_name)(hidden_layer)

        #   output layer: since it is a softmax layer a neuron is needed to encode each class
        output_layer = Dense(
            self.number_of_classes, activation='softmax', name='classifier_network_output_layer')(hidden_layer)

        #   create the network model
        classifier_network_model = Model(inputs=input_layer, outputs=output_layer, name='classifier_network_model')

        if self.learning_rate == "None":
            classifier_network_model.compile(loss='categorical_crossentropy',
                                             optimizer='adam',
                                             metrics=['categorical_accuracy'])

        else:
            opt = optimizers.Adam(lr=float(self.learning_rate))
            classifier_network_model.compile(loss='categorical_crossentropy',
                                             optimizer=opt,
                                             metrics=['categorical_accuracy'])

        print(classifier_network_model.summary(line_length=100))

        return classifier_network_model

    def train_classifier_net(self,
                             results_folder,
                             training_set,
                             training_supervision,
                             validation_set,
                             validation_supervision,
                             test_set=None,
                             test_supervision=None):

        log_file = open(results_folder + "/log_file.txt", "a")

        epochs = int(self.epochs)

        batch_size = int(self.batch_size)

        perc_gpu = float(self.perc_gpu)
        gpu_setup.gpu_setup(id_gpu=self.id_gpu, memory_percentage=perc_gpu)

        classifier_net_model = self.build_classifier_network()

        self.results_folder = results_folder

        for epoch in range(epochs):
            print "\n\n\nEpoch " + str(epoch)
            log_file.write("\n\n\nEpoch " + str(epoch))
            history_classifier_net = classifier_net_model.fit(x=training_set,
                                                              y=training_supervision,
                                                              batch_size=batch_size,
                                                              epochs=1,
                                                              shuffle=True,
                                                              validation_data=(validation_set, validation_supervision))

            self.classifier_network_epochs.append(len(history_classifier_net.history.get('loss')))
            if len(history_classifier_net.history.get('loss')) != 1:
                err_hndl(str_="error_epochs_repartition", add=inspect.stack()[0][3])

            self.classifier_network_loss_vec.append(
                history_classifier_net.history.get('loss')[0])
            log_file.write("\nClassifier loss ---> " + str(history_classifier_net.history.get('loss')[0]))

            self.classifier_network_categ_acc_vec.append(
                history_classifier_net.history.get('categorical_accuracy')[0])
            log_file.write("\nClassifier categorical accuracy ---> " + str(
                history_classifier_net.history.get('categorical_accuracy')[0]))

            self.classifier_network_val_loss_vec.append(
                history_classifier_net.history.get('val_loss')[0])
            log_file.write("\nClassifier validation loss ---> " + str(
                history_classifier_net.history.get('val_loss')[0]))

            self.classifier_network_val_categ_acc_vec.append(
                history_classifier_net.history.get('val_categorical_accuracy')[0])
            log_file.write("\nClassifier validation categorical accuracy ---> " + str(
                history_classifier_net.history.get('val_categorical_accuracy')[0]))

            """#   evaluation over the test set
            test_eval = classifier_net_model.evaluate(x=test_set, y=test_supervision, batch_size=batch_size)
            self.classifier_network_evaluation_on_test_set_loss_vec.append(
                test_eval[0]
            )
            self.classifier_network_evaluation_on_test_set_accuracy_vec.append(
                test_eval[1]
            )"""

            ###########################  these operations needs prediction and argmax transformation  ##########################
            training_set_classes_supervision = np.argmax(training_supervision, axis=1)
            training_set_classes_prediction = np.argmax(
                classifier_net_model.predict(x=training_set, batch_size=batch_size), axis=1)

            validation_set_classes_supervision = np.argmax(validation_supervision, axis=1)
            validation_set_classes_prediction = np.argmax(
                classifier_net_model.predict(x=validation_set, batch_size=batch_size), axis=1)

            """test_set_classes_supervision = np.argmax(test_supervision, axis=1)
            test_set_classes_prediction = np.argmax(
                classifier_net_model.predict(x=test_set, batch_size=batch_size), axis=1)"""

            training_precision = utilities.compute_precision(y_classes=training_set_classes_supervision,
                                                             y_pred_classes=training_set_classes_prediction)
            log_file.write("\nClassifier training_precision ---> " + str(training_precision))

            training_recall = utilities.compute_recall(y_classes=training_set_classes_supervision,
                                                       y_pred_classes=training_set_classes_prediction)
            log_file.write("\nClassifier training_recall ---> " + str(training_recall))

            training_f1 = utilities.compute_f1_score(y_classes=training_set_classes_supervision,
                                                     y_pred_classes=training_set_classes_prediction)
            log_file.write("\nClassifier training_f1 ---> " + str(training_f1))

            self.f1_value_training.append(training_f1)

            # %%%%%%%%%%%%%%%%%%%%%%%%%%

            validation_precision = utilities.compute_precision(y_classes=validation_set_classes_supervision,
                                                               y_pred_classes=validation_set_classes_prediction)
            log_file.write("\nClassifier validation_precision ---> " + str(validation_precision))

            validation_recall = utilities.compute_recall(y_classes=validation_set_classes_supervision,
                                                         y_pred_classes=validation_set_classes_prediction)
            log_file.write("\nClassifier validation_recall ---> " + str(validation_recall))

            validation_f1 = utilities.compute_f1_score(y_classes=validation_set_classes_supervision,
                                                       y_pred_classes=validation_set_classes_prediction)
            log_file.write("\nClassifier validation_f1 ---> " + str(validation_f1))

            self.f1_value_validation.append(validation_f1)

            """self.f1_value_test.append(utilities.compute_f1_score(y_classes=test_set_classes_supervision,
                                                                 y_pred_classes=test_set_classes_prediction))"""

            ####################################################################################################################

            #   save all vectors
            with open(results_folder + '/classifier_network_epochs.pkl', 'wb') as f:
                pickle.dump(self.classifier_network_epochs, f)
            with open(results_folder + '/classifier_network_loss_vec.pkl', 'wb') as f:
                pickle.dump(self.classifier_network_loss_vec, f)
            with open(results_folder + '/classifier_network_categ_acc_vec.pkl', 'wb') as f:
                pickle.dump(self.classifier_network_categ_acc_vec, f)
            with open(results_folder + '/classifier_network_val_loss_vec.pkl', 'wb') as f:
                pickle.dump(self.classifier_network_val_loss_vec, f)
            with open(results_folder + '/classifier_network_val_categ_acc_vec.pkl', 'wb') as f:
                pickle.dump(self.classifier_network_val_categ_acc_vec, f)

            """#   classifier_net_model.evaluate ---> ['loss', 'categorical_accuracy']
            with open(results_folder + 'classifier_network_evaluation_on_test_set_loss_vec.pkl', 'wb') as f:
                pickle.dump(self.classifier_network_evaluation_on_test_set_loss_vec, f)
            with open(results_folder + 'classifier_network_evaluation_on_test_set_accuracy_vec.pkl', 'wb') as f:
                pickle.dump(self.classifier_network_evaluation_on_test_set_accuracy_vec, f)"""

            with open(results_folder + '/f1_value_training_vec.pkl', 'wb') as f:
                pickle.dump(self.f1_value_training, f)
            with open(results_folder + '/f1_value_validation_vec.pkl', 'wb') as f:
                pickle.dump(self.f1_value_validation, f)
            """with open(results_folder + 'f1_value_test_vec.pkl', 'wb') as f:
                pickle.dump(self.f1_value_test, f)"""

            classifier_net_model.save(filepath=results_folder + "/classifier_net_model")
            classifier_net_model.save_weights(filepath=results_folder + "/classifier_net_model_weights")

        log_file.close()
        return None
