from utilities import plot_heatmap
import pandas as pn
import matplotlib.pyplot as plt
from utilities import download_from_remote
import shutil

MILES_EXP = '/home/comete/mromanel/MILES_EXP/EXP_GEO_LOCATION_QIF_LIB_SETTING/'


def main_plot_2D_ROI_and_heatmap():
    success, local_folder = download_from_remote.download_directory_from_server(
        remote_obj=MILES_EXP + 'mat_for_heatmap.pkl')

    #   coord ---> (lat, lon) ---> (y, x)
    #   matrix has lon=x, lat=y
    #   row 0 south, row 59 north, flipped
    #   row, cols ---> lat, lon
    #   cell 1, lon 1 lat 0, is therefore in (0, 1)
    #   cell 3558 is in (59, 58)

    mat_for_heatmap = pn.read_pickle(path=local_folder + "/mat_for_heatmap.pkl")

    shutil.rmtree(path=local_folder)

    success, local_folder = download_from_remote.download_directory_from_server(
        remote_obj=MILES_EXP + 'cell_dict_occurences_per_cell.pkl')

    cell_dict_occurences_per_cell = pn.read_pickle(path=local_folder + "/cell_dict_occurences_per_cell.pkl")

    max_ = 0

    cells_with_at_least_one_sample = 0

    for cell in cell_dict_occurences_per_cell:
        if cell_dict_occurences_per_cell[cell] > 0:
            cells_with_at_least_one_sample += 1
            if cell_dict_occurences_per_cell[cell] > max_:
                max_ = cell_dict_occurences_per_cell[cell]

    print "cells_with_at_least_one_sample ---> ", cells_with_at_least_one_sample

    print "max_occurence_in_a_cell ---> ", max_

    print cell_dict_occurences_per_cell[200]

    print mat_for_heatmap

    print mat_for_heatmap[18, 16]

    plot_heatmap.plot_heat_map(mat_for_heatmap=mat_for_heatmap, max_=max_, save=True)

    shutil.rmtree(path=local_folder)

    # ex cell 2784, row 2784//60 lat, col 2784-(lat * 60) lon
