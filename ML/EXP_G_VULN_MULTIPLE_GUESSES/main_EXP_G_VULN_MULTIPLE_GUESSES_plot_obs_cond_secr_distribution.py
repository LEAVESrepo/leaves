from geometric_mechanisms import linear_geometric_mechanism

EXP_G_VULN_MULTIPLE_GUESSES_FOLDER_PATH = "/home/comete/mromanel/MILES_EXP/EXP_G_VULN_MULTIPLE_GUESSES/"
CHANNEL_MATRIX_PATH = EXP_G_VULN_MULTIPLE_GUESSES_FOLDER_PATH + "channel_df_norm.pkl"
LIST_OIF_SECRETS_OF_INTEREST = [5, 6]
RESULT_DIR_PATH = EXP_G_VULN_MULTIPLE_GUESSES_FOLDER_PATH + "coord_plot_distribution_s5_s6/"


def main_EXP_G_VULN_MULTIPLE_GUESSES_plot_obs_cond_secr_distribution():
    linear_geometric_mechanism.create_list_to_plot_secret_distribution(
        list_of_secrets_of_interest=LIST_OIF_SECRETS_OF_INTEREST,
        channel_matrix_df_path=CHANNEL_MATRIX_PATH,
        result_dir_path=RESULT_DIR_PATH)
