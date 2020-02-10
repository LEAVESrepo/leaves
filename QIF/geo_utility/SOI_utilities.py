"""
given a central position and side length in meters find the coordinates of the vertices to delimit the area of interest
"""

from geo_utility import laplacian_noise_and_other_utilities as ln
from geo_utility import position_class as pc
import math
import pandas as pn
import numpy as np
from utilities_pckg import utilities
import matplotlib.pyplot as plt


def create_square_limit(central_position, side_length):
    #   half side length forced to be float
    half_side_length = side_length / float(2)

    #############################################  north_western limit  ################################################
    #   go to the north
    north = ln.addVectorToPos(pos=central_position, distance=half_side_length, angle=0)

    #   go to the east
    pos_tmp_north = pc.position(north[0], north[1])
    north_west = ln.addVectorToPos(pos=pos_tmp_north, distance=half_side_length, angle=-math.pi / 2)

    #   final position
    nw = pc.position(north_west[0], north_west[1])

    #############################################  north_eastern limit  ###############################################
    #   go to the east
    north_east = ln.addVectorToPos(pos=pos_tmp_north, distance=half_side_length, angle=math.pi / 2)

    #   final position
    ne = pc.position(north_east[0], north_east[1])

    #############################################  south_western limit  ################################################
    #   go to the south
    south = ln.addVectorToPos(pos=central_position, distance=half_side_length, angle=math.pi)

    #   go to the west
    pos_tmp_south = pc.position(south[0], south[1])
    south_west = ln.addVectorToPos(pos=pos_tmp_south, distance=half_side_length, angle=-math.pi / 2)

    #   final position
    sw = pc.position(south_west[0], south_west[1])

    #############################################  south_western limit  ################################################
    #   go to the east
    south_east = ln.addVectorToPos(pos=pos_tmp_south, distance=half_side_length, angle=math.pi / 2)

    #   final position
    se = pc.position(south_east[0], south_east[1])

    #   return the limits
    min_lat = se.latitude
    max_lat = nw.latitude

    min_longit = nw.longitude
    max_longit = ne.longitude

    """print("\n\n\nSquare centered in " + str((central_position.latitude, central_position.longitude)) + " and " + str(
        side_length) + " meter side length.")

    print("nw.latitude " + str(nw.latitude))
    print("nw.longitude " + str(nw.longitude))

    print("ne.latitude " + str(ne.latitude))
    print("ne.longitude " + str(ne.longitude))

    print("sw.latitude " + str(sw.latitude))
    print("sw.longitude " + str(sw.longitude))

    print("se.latitude " + str(se.latitude))
    print("se.longitude " + str(se.longitude))

    print("min_lat " + str(min_lat))
    print("min_lon " + str(min_longit))

    print("max_lat " + str(max_lat))
    print("max_lon " + str(max_longit) + "\n\n\n")"""

    return [min_lat, max_lat, min_longit, max_longit]


def retrieve_samples_in_SOI_from_db(db_path, separator, colnames, keep_cols, edges_dict, header=None):
    df_from_db = pn.read_csv(filepath_or_buffer=db_path, sep=separator, header=header)
    df_from_db.columns = colnames
    # print df_from_db.head()
    mat_from_db = df_from_db[keep_cols].values
    # print mat_from_db.dtype

    records_of_interest_idx = np.where(
        (mat_from_db[:, 0] >= edges_dict["square_min_lat"]) & (
                mat_from_db[:, 0] <= edges_dict["square_max_lat"]) & (
                mat_from_db[:, 1] >= edges_dict["square_min_lon"]) & (
                mat_from_db[:, 1] <= edges_dict["square_max_lon"]))[0]

    return mat_from_db[records_of_interest_idx, :]


def create_square_grid(edges_dict, cells_per_side, path_to_save):
    start_lat = edges_dict["square_min_lat"]
    stop_lat = edges_dict["square_max_lat"]
    step_lat = (stop_lat - start_lat) / float(cells_per_side)
    lat_arange = np.arange(start=start_lat, stop=stop_lat, step=step_lat)

    start_lon = edges_dict["square_min_lon"]
    stop_lon = edges_dict["square_max_lon"]
    step_lon = (stop_lon - start_lon) / float(cells_per_side)
    lon_arange = np.arange(start=start_lon, stop=stop_lon, step=step_lon)

    cells_dict = {}  # {cell: (lat, lon)}

    #   coord = (lat, lon) but lon=x and lat=y
    for y in range(cells_per_side):  # loop over lats
        for x in range(cells_per_side):  # loop over lons
            lon = edges_dict["square_min_lon"] + ((step_lon / 2.) + (x * step_lon))
            lat = edges_dict["square_min_lat"] + ((step_lat / 2.) + (y * step_lat))
            #   cells id grows with lon to the right and with lat upward
            cells_dict[y * cells_per_side + x] = (lat, lon)

    """print "cells_dict[0]: ", cells_dict[0]
    print "cells_dict[" + str(cells_per_side - 1) + "]: ", cells_dict[cells_per_side - 1]
    print "cells_dict[" + str((cells_per_side ** 2) - (cells_per_side - 1)) + "]: ", cells_dict[
        (cells_per_side ** 2) - (cells_per_side - 1)]
    print "cells_dict[" + str((cells_per_side ** 2) - 1) + "]: ", cells_dict[(cells_per_side ** 2) - 1]"""

    pn.to_pickle(obj=cells_dict, path=path_to_save + "cells_dict_centers.pkl")

    return cells_dict


def associate_location_to_cell(location, cells_dic):
    dist = float("inf")
    cell = None

    for cell_ in cells_dic:
        D2_euclidean_distance = utilities.D2_euclidean_distance(a=location, b=cells_dic[cell_])
        if D2_euclidean_distance < dist:
            dist = D2_euclidean_distance
            cell = cell_

    return cell


def create_heatmap(cells_list, cells_per_side, path_to_save):
    # file_prior_distr = open(path_to_save + "file_prior_distr.txt", 'w+')

    cells_array = np.array(cells_list)
    cells_array_unq, cells_array_unq_count = np.unique(cells_array, return_counts=True)

    cell_dict_occurences_per_cell = {}
    cell_dict_prior = {}

    for i_ter in range(cells_per_side ** 2):
        if i_ter in cells_array_unq:
            idx = np.where(cells_array_unq == i_ter)[0]
            cell_dict_occurences_per_cell[i_ter] = cells_array_unq_count[idx[0]]
            cell_dict_prior[i_ter] = cells_array_unq_count[idx[0]] / float(len(cells_array))
            # file_prior_distr.write(str(cell_dict_prior[i_ter]))

        else:
            cell_dict_occurences_per_cell[i_ter] = 0
            cell_dict_prior[i_ter] = 0
            # file_prior_distr.write(str(cell_dict_prior[i_ter]))

        # if i_ter != (cells_per_side ** 2) - 1:
        # file_prior_distr.write("\t")
        # else:
        # file_prior_distr.write("\n")

    pn.to_pickle(obj=cell_dict_prior, path=path_to_save + "file_prior_distr.pkl", protocol=2)

    # file_prior_distr.close()

    pn.to_pickle(obj=cell_dict_occurences_per_cell, path=path_to_save + "cell_dict_occurences_per_cell.pkl", protocol=2)

    mat_for_heatmap = np.empty((cells_per_side, cells_per_side))

    for y in range(cells_per_side):  # loop over lats
        for x in range(cells_per_side):  # loop over lons
            #   first row from 0 sx to 59 dx
            #   second row from 60 sx to 119 dx
            #   ...
            #   last row from 3499 sx to 3559 dx
            #   lat along rows (y on a map), long along cols (x on a map)
            #   #####
            #   lat grows from up downward and lon from left to right
            mat_for_heatmap[y, x] = len(np.where(cells_array == y * cells_per_side + x)[0])

    pn.to_pickle(obj=mat_for_heatmap, path=path_to_save + "mat_for_heatmap.pkl", protocol=2)

    return mat_for_heatmap
