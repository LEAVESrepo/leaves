import os
import copy
import shutil
import matplotlib
import numpy as np
import pandas as pn
import seaborn as sns
import matplotlib.pyplot as plt
from utilities import download_from_remote

RM_FILES_AFTER_PLOT = True
g_f_star = 0.892


def plot_box_plot_BIS_EXP_G_VULN_MULTIPLE_GUESSES(df, title, y, save_fig=False):
    # print df.columns.values
    fig, ax1 = plt.subplots()
    colors = ["windows blue", "faded green"]  # ["windows blue", "faded green", "red"]

    #   plot real error baseline
    # plt.plot([-100000, 100000], [0.108, 0.108], linewidth=2, color='black', linestyle='dashed')
    plt.plot([-100000, 100000], [0.892, 0.892], linewidth=2, color='black', linestyle='dashed')
    bp = sns.boxplot(x="number_of_training_samples", y=y, hue="hue", data=df,
                     ax=ax1,
                     linewidth=.5,
                     palette=sns.xkcd_palette(colors))

    ax1.annotate('g-Vulnerability: 0.892', xy=(-0.30, 0.895))

    labels = ["ANN with preprocessing",
              "KNN with preprocessing",
              "Frequentist"]

    h, l = bp.get_legend_handles_labels()

    bp.legend(h, labels, title=title)

    plt.ylabel("g-Vulnerability estimation")
    plt.xlabel("Training set cardinality")

    if save_fig:
        fig.savefig('/Users/marcoromanelli/Desktop/bp_' + title + '.pdf', dpi=3000)

    plt.show()


def plot_box_plot_BIS_EXP_G_VULN_MULTIPLE_GUESSES_new(df, title, y, save_fig=False):
    # print df.columns.values
    fig, ax1 = plt.subplots()
    colors = ["windows blue", "faded green"]  # ["windows blue", "faded green", "red"]

    #   plot real error baseline
    # plt.plot([-100000, 100000], [0.108, 0.108], linewidth=2, color='black', linestyle='dashed')
    ax1.annotate('dispersion: 0.012', xy=(-0.445, 0.11), size=8, rotation=0)
    ax1.annotate('total error: 0.061', xy=(-0.445, 0.10), size=8)
    plt.plot([-0.395, -0.005], [0.06, 0.06], linewidth=.5, color='black')

    ax1.annotate('dispersion: 0.003', xy=(-0.095, 0.21), size=8)
    ax1.annotate('total error: 0.231', xy=(-0.095, 0.20), size=8)
    plt.plot([0.005, 0.395], [0.231, 0.231], linewidth=.5, color='black')

    ax1.annotate('dispersion: 0.016', xy=(0.505, 0.09), size=8)
    ax1.annotate('total error: 0.048', xy=(0.505, 0.08), size=8)
    plt.plot([0.605, 0.995], [0.046, 0.046], linewidth=.5, color='black')

    ax1.annotate('dispersion: 0.004', xy=(0.905, 0.18), size=8)
    ax1.annotate('total error: 0.203', xy=(0.905, 0.17), size=8)
    plt.plot([1.005, 1.395], [0.203, 0.203], linewidth=.5, color='black')

    ax1.annotate('dispersion: 0.006', xy=(1.505, 0.09), size=8)
    ax1.annotate('total error: 0.044', xy=(1.505, 0.08), size=8)
    plt.plot([1.605, 1.995], [0.043, 0.043], linewidth=.5, color='black')

    ax1.annotate('dispersion: 0.003', xy=(1.865, 0.16), size=8)
    ax1.annotate('total error: 0.186', xy=(1.865, 0.15), size=8)
    plt.plot([2.005, 2.3955], [0.186, 0.186], linewidth=.5, color='black')
    bp = sns.boxplot(x="number_of_training_samples", y=y, hue="hue", data=df,
                     ax=ax1,
                     linewidth=.5,
                     palette=sns.xkcd_palette(colors),
                     medianprops={'color': (0.2, 0.4, 0.6, 0)})
    for c in bp.get_children():
        if type(c) == matplotlib.patches.PathPatch:
            print c.get_extents()

    # ax1.annotate('g-Vulnerability: 0.892', xy=(-0.30, 0.893))

    labels = ["ANN",
              "k-NN"]

    h, l = bp.get_legend_handles_labels()

    bp.legend(h, labels, title=title)

    plt.ylabel("normalized estimation error")
    plt.xlabel("Training set size")

    if save_fig:
        fig.savefig('/Users/marcoromanelli/Desktop/bp_' + title + '.pdf', dpi=3000)

    plt.show()


def main_BIS_EXP_G_VULN_MULTIPLE_GUESSES_box_plot():
    #################################################  REMAPPING KNN  ######################################################

    success, local_folder = download_from_remote.download_directory_from_server(
        remote_obj='/home/comete/mromanel/MILES_EXP/BIS_EXP_G_VULN_MULTIPLE_GUESSES_bckp/RESULT_FOLDER_REMAPPING/KNN/KNN*.pkl'
    )
    colnames = None
    mat_final = None
    for filename in os.listdir(local_folder):
        print filename
        df = pn.read_pickle(path=local_folder + "/" + filename)
        if colnames is None:
            colnames = df.columns.values
        if mat_final is None:
            mat_final = df.values
        else:
            mat_final = np.concatenate((mat_final, df.values), axis=0)
    shutil.rmtree(path=local_folder)

    mat_final_df_KNN_REMAPPING = pn.DataFrame(data=mat_final, columns=colnames)
    print mat_final_df_KNN_REMAPPING.values.shape
    print colnames

    #################################################  REMAPPING ANN  ######################################################

    success, local_folder = download_from_remote.download_directory_from_server(
        remote_obj='/home/comete/mromanel/MILES_EXP/BIS_EXP_G_VULN_MULTIPLE_GUESSES_bckp/RESULT_FOLDER_REMAPPING/model_1000_ep500/ANN*.pkl'
    )
    colnames = None
    mat_final = None
    for filename in os.listdir(local_folder):
        print filename
        df = pn.read_pickle(path=local_folder + "/" + filename)
        if colnames is None:
            colnames = df.columns.values
        if mat_final is None:
            mat_final = df.values
        else:
            mat_final = np.concatenate((mat_final, df.values), axis=0)
    shutil.rmtree(path=local_folder)

    mat_final_df_ANN_REMAPPING = pn.DataFrame(data=mat_final, columns=colnames)
    print mat_final_df_ANN_REMAPPING.values.shape
    print colnames

    ##################################################  FREQUENTIST  #######################################################

    # success, local_folder = download_from_remote.download_directory_from_server(
    #     remote_obj='/home/comete/mromanel/MILES_EXP/EXP_G_VULN_MULTIPLE_GUESSES/RESULT_FOLDER/FREQUENTIST/FREQ*.pkl'
    # )
    # colnames = None
    # mat_final = None
    # for filename in os.listdir(local_folder):
    #     print filename
    #     df = pn.read_pickle(path=local_folder + "/" + filename)
    #     if colnames is None:
    #         colnames = df.columns.values
    #     if mat_final is None:
    #         mat_final = df.values
    #     else:
    #         mat_final = np.concatenate((mat_final, df.values), axis=0)
    # shutil.rmtree(path=local_folder)
    #
    # mat_final_df_FREQ = pn.DataFrame(data=mat_final, columns=colnames)
    # print mat_final_df_FREQ.values.shape
    # print colnames

    ################################################

    mat_final = np.concatenate([mat_final_df_ANN_REMAPPING.values, mat_final_df_KNN_REMAPPING.values])
    # mat_final = np.concatenate([mat_final, mat_final_df_FREQ.values])

    # mat_final = mat_final_df_KNN_REMAPPING.values

    list_source = []

    for i_ter in range(mat_final.shape[0] // 2):
        list_source.append(0.0)  # ANN + REMAPPING

    for i_ter in range(mat_final.shape[0] // 2, mat_final.shape[0]):
        list_source.append(1.0)  # KNN + REMAPPING

    # for i_ter in range(2 * mat_final.shape[0] // 3, mat_final.shape[0]):
    #     list_source.append(2.0)  # FREQ

    # for i_ter in range(mat_final.shape[0]):
    #     list_source.append(1.0)  # KNN

    # print len(list_source)

    mat_final = np.column_stack((mat_final, np.array(list_source).reshape(mat_final.shape[0], 1)))

    colnames = list(colnames)
    colnames.append("hue")

    """for i in range(mat_final.shape[0]):
        mat_final[i, 0] = 1 - mat_final[i, 0]"""

    mat_final_df = pn.DataFrame(data=mat_final, columns=colnames)
    mat_final_df["number_of_training_samples"] = mat_final_df["number_of_training_samples"].astype('int')
    # plot_box_plot_BIS_EXP_G_VULN_MULTIPLE_GUESSES(df=mat_final_df,
    #                                               title="",
    #                                               y="ANN_Rf_values",
    #                                               save_fig=True)

    mat_final_df_cpy = copy.deepcopy(mat_final_df)
    mat_final_cpy = mat_final_df_cpy.values
    for i in range(mat_final_cpy.shape[0]):
        mat_final_cpy[i, 0] = abs(mat_final_cpy[i, 0] - g_f_star) / float(g_f_star)

    mat_final_df_cpy = pn.DataFrame(data=mat_final_cpy, columns=colnames)
    mat_final_df_cpy["number_of_training_samples"] = mat_final_df_cpy["number_of_training_samples"].astype('int')

    for size in np.unique(mat_final_df.values[:, 2]):
        for model in np.unique(mat_final_df.values[:, -1]):
            idx_of_interest = np.where((mat_final_df.values[:, 2] == size) & (mat_final_df.values[:, -1] == model))[0]
            print("\nSize: " + str(size))
            print("Model: " + str(model))
            print("idx_of_interest: " + str(len(idx_of_interest)))
            # g_var = np.mean(mat_final_df.values[idx_of_interest, 0])
            # avg = abs(g_var - g_f_star) / float(g_f_star)
            # print("g_var: " + str(g_var))
            avg = np.mean(mat_final_cpy[idx_of_interest, 0])
            print("avg: " + str(avg))
            # dispersion_std = np.std(mat_final_df.values[idx_of_interest, 0])
            # dispersion_formula = np.sqrt(np.mean(((mat_final_df.values[idx_of_interest, 0] - g_var) ** 2)))
            dispersion_std = np.std(mat_final_df_cpy.values[idx_of_interest, 0])
            dispersion_formula = np.sqrt(np.mean((mat_final_df_cpy.values[idx_of_interest, 0] - avg)**2))
            print("dispersion_std: " + str(dispersion_std))
            print("dispersion_formula: " + str(dispersion_formula))
            # total_error = np.sqrt(np.mean(((mat_final_df.values[idx_of_interest, 0] - g_f_star) / float(g_f_star)) ** 2))
            total_error = np.sqrt(np.mean(mat_final_df_cpy.values[idx_of_interest, 0] ** 2))
            print("total_error: " + str(total_error))
            print("\n\n\n###########################")

    plot_box_plot_BIS_EXP_G_VULN_MULTIPLE_GUESSES_new(df=mat_final_df_cpy,
                                                      title="",
                                                      y="ANN_Rf_values",
                                                      save_fig=True)
