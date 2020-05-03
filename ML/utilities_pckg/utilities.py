import os
import sys
import copy
import time
import inspect
import numpy as np
from runtime_error_handler import runtime_error_handler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def createFolder(path):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except OSError:
            runtime_error_handler(str_="folder_creation_failed", add=inspect.stack()[0][3])
        else:
            print("\nSuccessfully created the directory %s " % path)
    else:
        print("\nDirectory %s already existing." % path)


def inline_print_secret(arg):
    sys.stdout.write("\rCreating channel for secret: " + str(arg))
    sys.stdout.flush()


def uniform_distribution_given_symbols(secrets_list):
    secret_prior = {}
    for secret in secrets_list:
        secret_prior[secret] = 1 / float(len(secrets_list))
    return secret_prior


def create_new_rndm_state():
    tm = int(time.time() * (10 ** 6))
    bs = int(str(int(time.time()))[0:-3]) * (10 ** 9)
    return np.random.RandomState(seed=tm - bs)


def check_list_order(list_to_be_checked, criterion="increasing"):
    for i_ter in range(list_to_be_checked.shape[0]):
        a = list(list_to_be_checked[i_ter, :])
        b = a[:]
        if criterion == "increasing":
            b.sort()
            if b == a:
                return True
            else:
                return False

        elif criterion == "decreasing":
            b.sort(reverse=True)
            if b == a:
                return True
            else:
                return False
        else:
            runtime_error_handler(str_="unspecified_option", add=inspect.stack()[0][3])


def compute_accuracy(y_classes, y_pred_classes):
    tmp = []
    for el in y_pred_classes:
        tmp.append(float(el))

    y_pred_classes = tmp

    a = 0
    b = 0

    # print 'y_classes', y_classes
    # print 'y_pred_classes', y_pred_classes

    for i in range(0, len(y_pred_classes)):
        if y_pred_classes[i] == y_classes[i]:
            a += 1
        b += 1

    return a / float(b)


def compute_accuracy_tf_fashion(y_classes, y_pred_classes):
    tmp = []
    for el in y_pred_classes:
        tmp.append(float(el))

    y_pred_classes = tmp

    all_classes = np.unique(y_classes)

    res = 0
    for el in all_classes:
        count_matches = 0
        list_samples_of_class_el = np.where(y_classes == el)[0]
        for sample_id in list_samples_of_class_el:
            if y_pred_classes[sample_id] == el:
                count_matches += 1
        res += count_matches / float(len(list_samples_of_class_el))
    res = res / float(len(all_classes))
    return res


def compute_precision(y_classes, y_pred_classes):
    tmp = []
    for el in y_pred_classes:
        tmp.append(float(el))

    y_pred_classes = tmp

    return precision_score(y_true=y_classes, y_pred=y_pred_classes, average=None)


def compute_recall(y_classes, y_pred_classes):
    tmp = []
    for el in y_pred_classes:
        tmp.append(float(el))

    y_pred_classes = tmp

    return recall_score(y_true=y_classes, y_pred=y_pred_classes, average=None)


def compute_f1_score(y_classes, y_pred_classes):
    tmp = []
    for el in y_pred_classes:
        tmp.append(float(el))

    y_pred_classes = tmp

    return f1_score(y_true=y_classes, y_pred=y_pred_classes, average="macro")


def compute_confusion_matrix(y_classes, y_pred_classes):
    tmp = []
    for el in y_pred_classes:
        tmp.append(float(el))

    y_pred_classes = tmp

    return confusion_matrix(y_true=y_classes, y_pred=y_pred_classes)


def compute_prior_distribution_from_array_of_symbols(array):
    #   dictionary that maps a symbol to its number of occurrences
    symbols_map = {}
    unique_symbols, count_occ_symbols = np.unique(array, return_counts=True)
    for i in range(len(unique_symbols)):
        symbols_map[unique_symbols[i]] = count_occ_symbols[i]

    copy_of_symbols_map = copy.deepcopy(symbols_map)
    for element in copy_of_symbols_map:
        copy_of_symbols_map[element] = copy_of_symbols_map[element] / float(len(array))
    return copy_of_symbols_map


def reduce_data_dimensionality(data, cvrt_to_int=False):
    new_data = []

    for i in range(data.shape[0]):
        new_record = ""

        for j in range(data.shape[1] - 1):
            if cvrt_to_int:
                new_record += str(int(data[i, j])) + "-"
            else:
                new_record += str(data[i, j]) + "-"

        new_data.append(new_record[:-1])

    new_data = np.array(new_data).reshape(data.shape[0], 1)
    new_data = np.column_stack((new_data, data[:, -1]))

    return new_data


def find_most_frequent_symbol_in_array(array):
    unq_array, unq_array_cnt = np.unique(array, return_counts=True)

    return unq_array[np.argmax(unq_array_cnt)]


def array_intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))
