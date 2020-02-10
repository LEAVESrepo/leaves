import pickle
import os


def load_pickled_result_file(input_file_absolute_path, rm_file=False):
    try:
        with open(input_file_absolute_path) as f:
            try:
                loaded_obj = pickle.load(f)

            except pickle.PickleError:
                print("Cannot load " + input_file_absolute_path)

            else:
                if rm_file is True:
                    os.remove(input_file_absolute_path)

                return loaded_obj

    except EnvironmentError:
        print("Cannot find " + input_file_absolute_path)
