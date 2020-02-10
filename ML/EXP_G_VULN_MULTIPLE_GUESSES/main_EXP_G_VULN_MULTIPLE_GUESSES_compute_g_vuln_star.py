from utilities_pckg import g_vuln_computation
import pandas as pn
from utilities_pckg import utilities

EXP_G_VULN_MULTIPLE_GUESSES_FOLDER_PATH = "/home/comete/mromanel/MILES_EXP/EXP_G_VULN_MULTIPLE_GUESSES/"
CHANNEL_PATH = EXP_G_VULN_MULTIPLE_GUESSES_FOLDER_PATH + "channel_df_norm.pkl"
G_MATRIX_PATH = '/home/comete/mromanel/MILES_EXP/EXP_G_VULN_MULTIPLE_GUESSES/G_MAT_FOLDER/' \
                'g_matrix_10_secrets_2_guesses.pkl'
G_MATRIX_ROWS_PATH = '/home/comete/mromanel/MILES_EXP/EXP_G_VULN_MULTIPLE_GUESSES/G_MAT_FOLDER/' \
                     'g_matrix_10_secrets_2_guesses_rows.pkl'
G_MATRIX_COLS_PATH = '/home/comete/mromanel/MILES_EXP/EXP_G_VULN_MULTIPLE_GUESSES/G_MAT_FOLDER/' \
                     'g_matrix_10_secrets_2_guesses_cols.pkl'
FINAL_FILE_PATH = '/home/comete/mromanel/MILES_EXP/EXP_G_VULN_MULTIPLE_GUESSES/R_star_with_g_matrix.txt'


def main_EXP_G_VULN_MULTIPLE_GUESSES_compute_g_vuln_star():
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  geometric distribution setup  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    channel_df_from_known_distribution = pn.read_pickle(path=CHANNEL_PATH)
    channel_colnames = channel_df_from_known_distribution.columns.values
    channel_rownames = channel_df_from_known_distribution.index.values
    channel_matrix_from_known_distribution = channel_df_from_known_distribution.values

    secret_prior = utilities.uniform_distribution_given_symbols(secrets_list=channel_colnames)
    # print(secret_prior)

    #   create the channel matrix for the distribution: an observable for each row and a secret for each column

    loaded_g_matrix = pn.read_pickle(path=G_MATRIX_PATH)
    loaded_g_matrix_rows = pn.read_pickle(path=G_MATRIX_ROWS_PATH)
    loaded_g_matrix_cols = pn.read_pickle(path=G_MATRIX_COLS_PATH)

    R_star_with_g_matrix = g_vuln_computation.compute_g_vuln_star(
        channel=channel_matrix_from_known_distribution,
        secret_prior_dict=secret_prior,
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
