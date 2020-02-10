import numpy as np
import pandas as pn
from utilities_pckg import g_vuln_computation, utilities

EXP_GEO_LOCATION_FOLDER_PATH = "/home/comete/mromanel/MILES_EXP/EXP_GEO_LOCATION_QIF_LIB_SETTING/"

CHANNEL_PATH = EXP_GEO_LOCATION_FOLDER_PATH + "channel.pkl"
ORIGINAL_SECRETS_OCCURRENCES_FILE = EXP_GEO_LOCATION_FOLDER_PATH + "cell_dict_occurences_per_cell.pkl"
N_CELLS = 400

G_MATRIX_PATH = EXP_GEO_LOCATION_FOLDER_PATH + 'G_OBJ/g_mat.pkl'
G_MATRIX_ROWS_PATH = EXP_GEO_LOCATION_FOLDER_PATH + 'G_OBJ/g_mat_rows.pkl'
G_MATRIX_COLS_PATH = EXP_GEO_LOCATION_FOLDER_PATH + 'G_OBJ/g_mat_cols.pkl'
FINAL_FILE_PATH = EXP_GEO_LOCATION_FOLDER_PATH + 'R_star_with_g_matrix.txt'


def main_EXP_G_VULN_GEO_LOCATION_compute_g_vuln_star():
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  geometric distribution setup  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    channel_matrix_from_known_distribution = pn.read_pickle(path=CHANNEL_PATH)
    channel_colnames = np.arange(start=0, stop=N_CELLS, step=1)
    channel_rownames = np.arange(start=0, stop=N_CELLS, step=1)

    secrets_occurr_dictionary = pn.read_pickle(path=ORIGINAL_SECRETS_OCCURRENCES_FILE)
    # print "secrets_occurr_dictionary ===> ", secrets_occurr_dictionary

    tot_occurr = 0
    maxx_occurr = 0
    for secret in secrets_occurr_dictionary:
        current_secret_occurr = secrets_occurr_dictionary[secret]
        if current_secret_occurr > maxx_occurr:
            maxx_occurr = current_secret_occurr
        tot_occurr += current_secret_occurr
        # print secret, " ===> ", current_secret_occurr
    # print "maxx_occurr ===> ", maxx_occurr

    secrets_prior_dictionary = {}
    maxx_freq = 0
    for secret in secrets_occurr_dictionary:
        secrets_prior_dictionary[secret] = secrets_occurr_dictionary[secret] / float(tot_occurr)
        if secrets_prior_dictionary[secret] > maxx_freq:
            maxx_freq = secrets_prior_dictionary[secret]
    # print "secrets_prior_dictionary ===> ", secrets_prior_dictionary
    # print "maxx_freq ===> ", maxx_freq

    #   create the channel matrix for the distribution: an observable for each row and a secret for each column

    loaded_g_matrix = pn.read_pickle(path=G_MATRIX_PATH)
    loaded_g_matrix_rows = pn.read_pickle(path=G_MATRIX_ROWS_PATH)
    loaded_g_matrix_cols = pn.read_pickle(path=G_MATRIX_COLS_PATH)

    R_star_with_g_matrix = g_vuln_computation.compute_g_vuln_star(
        channel=channel_matrix_from_known_distribution,
        secret_prior_dict=secrets_prior_dictionary,
        rows=channel_rownames,
        cols=channel_colnames,
        g_mat=loaded_g_matrix,
        g_mat_rows=loaded_g_matrix_rows,
        g_mat_cols=loaded_g_matrix_cols
    )

    file_R_star_with_g_matrix = open(FINAL_FILE_PATH, "wa")
    file_R_star_with_g_matrix.write("R_star_with_g_matrix = " + str(R_star_with_g_matrix))
    file_R_star_with_g_matrix.close()

    print R_star_with_g_matrix
