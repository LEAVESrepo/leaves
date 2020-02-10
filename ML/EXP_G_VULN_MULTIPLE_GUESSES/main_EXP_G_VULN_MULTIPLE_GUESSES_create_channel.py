from geometric_mechanisms import linear_geometric_mechanism as lgm
import numpy as np
from utilities_pckg import utilities

NU = 0.002
SECRETS_CARD = 10.
OBSERVABLE_CARD = 16000.
EXP_G_VULN_MULTIPLE_GUESSES_FOLDER_PATH = "/home/comete/mromanel/MILES_EXP/EXP_G_VULN_MULTIPLE_GUESSES/"


def main_EXP_G_VULN_MULTIPLE_GUESSES_create_channel():
    utilities.createFolder(EXP_G_VULN_MULTIPLE_GUESSES_FOLDER_PATH)
    #   set of secrets
    secrets = np.arange(0., SECRETS_CARD, 1)

    #   set of observables
    observables = np.arange(0., OBSERVABLE_CARD, 1.)

    lgm_manager = lgm.LinearGeometricMechanism(secrets=secrets, observables=observables, nu=NU, truncation=False)
    lgm_manager.create_channel_matrix_from_known_distribution(
        shift=3000,
        save_channel_matrix_path=EXP_G_VULN_MULTIPLE_GUESSES_FOLDER_PATH + "channel_df_norm.pkl",
        symmetry=True)
