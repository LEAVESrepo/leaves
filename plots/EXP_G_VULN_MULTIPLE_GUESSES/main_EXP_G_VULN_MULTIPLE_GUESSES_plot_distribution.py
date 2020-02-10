from utilities import download_from_remote
from utilities import plot_distribution_around_secrets
import pickle
import os
import shutil

RM_FILES_AFTER_PLOT = True

REMOTE_OBJ = "/home/comete/mromanel/MILES_EXP/EXP_G_VULN_MULTIPLE_GUESSES/coord_plot_distribution_s5_s6/*"


def main_EXP_G_VULN_MULTIPLE_GUESSES_plot_distribution():
    [succ, local_folder] = download_from_remote.download_directory_from_server(remote_obj=REMOTE_OBJ,
                                                                               local_folder=os.path.dirname(
                                                                                   os.path.dirname(os.path.realpath(
                                                                                       __file__))) + '/tmp_res/')

    with open(local_folder + 'list_tmp_x.pkl', 'rb') as pickle_file:
        x_list = pickle.load(pickle_file)
    with open(local_folder + 'list_tmp_y.pkl', 'rb') as pickle_file:
        y_list = pickle.load(pickle_file)

    plot_distribution_around_secrets.plot_from_lists_of_coordinates(x_list=x_list, y_list=y_list, colors=['purple', 'orange'],
                                                                    centers=[5, 6], x_lim=(8000, 10000),
                                                                    save_plot=True, dpi=3000)

    if RM_FILES_AFTER_PLOT:
        shutil.rmtree(path=local_folder)
