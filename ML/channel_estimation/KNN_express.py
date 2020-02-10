import copy
import inspect
import numpy as np
from utilities_pckg import utilities
from sklearn.neighbors import NearestNeighbors
from utilities_pckg import runtime_error_handler


def fix_order_distances(distances, indices):
    indices_ = []
    distances_array = np.array(distances)
    unq_dist = np.unique(np.array(distances))

    for dist in unq_dist:
        idx = np.where(distances_array == dist)[0]
        for id in idx:
            indices_.append(indices[id])

    distances_ = copy.deepcopy(distances)
    distances_.sort()

    return [distances_, indices_]


class KNN_express_EstimatorManager:
    def __init__(self, observed_samples, k_neighbors):
        self.observed_samples = observed_samples
        self.nrows = observed_samples.shape[0]
        self.ncols = observed_samples.shape[1]
        self.k_neighbors = k_neighbors

        self.secrets = observed_samples[:, -1]
        self.observables = observed_samples[:, 0:self.ncols - 1]

    #   we know for sure that the result from scikit are ordered from the smallest distances to the biggest
    def predict_KNN_KDT(self, observables_test, enhanced=False, p_=2):
        print "...finding neighbors..."

        #   fit the training observables so we "learn" some observations
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors, algorithm='kd_tree', p=p_).fit(self.observables)

        #   retrieve distances and realtive sample indices when trying to find the neighbors of a test observation
        distances, indices = nbrs.kneighbors(observables_test)
        # print distances.shape

        # print "\n\nindices\n", indices
        # print "\n\ndistances\n", distances
        #   print observables_test.shape[0]

        k_neig = self.k_neighbors

        if enhanced:
            print "...still finding neighbors..."
            while k_neig < self.nrows:  # this is only < since the incrementation is inside
                k_neig += 1
                # print k_neig
                nbrs_ = NearestNeighbors(n_neighbors=k_neig, algorithm='kd_tree', p=p_).fit(self.observables)
                distances_, indices_ = nbrs_.kneighbors(observables_test)
                a = distances[:, -1]
                b = distances_[:, -1]

                check = True
                for i in range(len(a)):
                    if a[i] == b[i]:
                        check = False
                        break

                if check or k_neig == self.nrows:
                    distances = distances_
                    indices = indices_
                    break

        #   check if distances are ordered from the smallest to the biggest
        if utilities.check_list_order(list_to_be_checked=distances, criterion="increasing") is False:
            runtime_error_handler.runtime_error_handler(str_="wrong_order_but_right_expected", add=inspect.stack()[0][3])

        predictions = []

        print "...classifying..."
        for i in range(observables_test.shape[0]):
            indices_i_cpy = copy.deepcopy(indices[i, :])
            distances_i_cpy = copy.deepcopy(distances[i, :])
            list_distances_i_cpy = list(distances_i_cpy)

            max_i_dist = np.max(list_distances_i_cpy[0:self.k_neighbors])
            last_batch = list(indices_i_cpy[np.where(distances_i_cpy == max_i_dist)[0]])
            prev_batches = list(indices_i_cpy[np.where(distances_i_cpy < max_i_dist)[0]])

            classes_previous_batches = list(self.secrets[prev_batches])
            classes_last_batch = list(self.secrets[last_batch])

            """if i == 0:
                print "classes_previous_batches ", classes_previous_batches
                print "classes_last_batch ", classes_last_batch"""

            if len(classes_last_batch) + len(classes_previous_batches) == self.k_neighbors:
                list_classes = []
                list_classes.extend(classes_last_batch)
                list_classes.extend(classes_previous_batches)
                array_classes = np.array(list_classes)
                unq, count_unq = np.unique(array_classes, return_counts=True)
                predictions.append(unq[np.argmax(count_unq)])

            else:
                list_classes = []
                list_classes.extend(classes_previous_batches)

                array_classes_last_batch = np.array(classes_last_batch)
                unq, count_unq = np.unique(array_classes_last_batch, return_counts=True)
                most_frequent_class_in_ties = unq[np.argmax(count_unq)]

                for i_ter in range(len(classes_previous_batches), self.k_neighbors):
                    list_classes.append(most_frequent_class_in_ties)

                array_classes = np.array(list_classes)
                unq, count_unq = np.unique(array_classes, return_counts=True)
                predictions.append(unq[np.argmax(count_unq)])

                """if i == 0:
                    print unq
                    print count_unq
                    print "predictions ", predictions"""
        print "...done classifying..."
        return [indices, predictions]

    def new_way_prediction_faster(self, ind, O_train_unq, O_train, Z_train, O_test, O_test_unq):
        final_pred_u = []
        neig_lists = []
        for O_ts_u in O_test_unq:
            idx_o_test = np.where(O_test_unq == O_ts_u)[0][0]
            neig_lists.append(O_train_unq[ind[idx_o_test]])
        mapping_tr = []
        for O_tr_u in O_train_unq:
            mapping_tr.append(np.where(O_train == O_tr_u)[0])

        for neig_list in neig_lists:
            classes = []
            for el in neig_list:  # sample in original tr_set
                idx = np.where(O_train_unq == el)[0][0]
                classes.extend(Z_train[mapping_tr[idx]])

            classes_array, classes_counter = np.unique(np.array(classes), return_counts=True)
            final_pred_u.append(classes_array[np.argmax(classes_counter)])

        final_pred = []

        for O_ts in O_test:
            idx_o_test = np.where(O_test_unq == O_ts)[0][0]
            final_pred.append(final_pred_u[idx_o_test])

        return final_pred

    def new_way_prediction_faster_multidimensional(self, ind, O_train_unq, O_train, Z_train, O_test, O_test_unq):
        final_pred_u = []
        neig_lists = []
        for O_ts_u in O_test_unq:
            idx_o_test = np.where((O_test_unq == tuple(O_ts_u)).all(axis=1))[0][0]
            neig_lists.append(O_train_unq[ind[idx_o_test]])

        mapping_tr = []
        for O_tr_u in O_train_unq:
            mapping_tr.append(np.where((O_train == tuple(O_tr_u)).all(axis=1))[0])

        for neig_list in neig_lists:
            classes = []
            for el in neig_list:  # sample in original tr_set
                idx = np.where((O_train_unq == tuple(el)).all(axis=1))[0][0]
                classes.extend(Z_train[mapping_tr[idx]])

            classes_array, classes_counter = np.unique(np.array(classes), return_counts=True)
            final_pred_u.append(classes_array[np.argmax(classes_counter)])

        final_pred = []

        for O_ts in O_test:
            idx_o_test = np.where((O_test_unq == tuple(O_ts)).all(axis=1))[0][0]
            final_pred.append(final_pred_u[idx_o_test])

        return final_pred
