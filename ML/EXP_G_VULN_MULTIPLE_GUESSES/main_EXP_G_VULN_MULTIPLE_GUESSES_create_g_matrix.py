from utilities_pckg import g_function_manager, utilities
import pandas as pn

EXP_G_VULN_MULTIPLE_GUESSES_FOLDER_PATH = "/home/comete/mromanel/MILES_EXP/EXP_G_VULN_MULTIPLE_GUESSES/"
CHANNEL_PATH = EXP_G_VULN_MULTIPLE_GUESSES_FOLDER_PATH + "channel_df_norm.pkl"
G_MAT_FOLDER = EXP_G_VULN_MULTIPLE_GUESSES_FOLDER_PATH + "G_MAT_FOLDER/"
utilities.createFolder(G_MAT_FOLDER)
N_GUESSES = 2


def main_EXP_G_VULN_MULTIPLE_GUESSES_create_g_matrix():
    channel_colnames = pn.read_pickle(path=CHANNEL_PATH).columns.values
    g_function_manager.create_g_function_matrix_n_guesses(list_unique_secrets=channel_colnames,
                                                          n_guesses=N_GUESSES,
                                                          save_g_path=G_MAT_FOLDER)
