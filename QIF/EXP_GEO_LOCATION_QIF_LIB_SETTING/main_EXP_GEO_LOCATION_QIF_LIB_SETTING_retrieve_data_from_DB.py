from geo_utility import SOI_utilities, position_class
from utilities_pckg import utilities
import tqdm
import warnings
import sys

CENTER_LAT = 37.755
CENTER_LON = -122.440

SMALL_SQUARE_SIZE = 5000

# BIG_SQUARE_SIZE = 6000

SMALL_SQUARE_CELL_SIDE_LENGTH = 250

CELLS_PER_SIDE = SMALL_SQUARE_SIZE // SMALL_SQUARE_CELL_SIDE_LENGTH

DATABASE_PATH = "/home/comete/mromanel/MILES_EXP/gowalla/loc-gowalla_totalCheckins.txt"
COLNAMES = ["user", "check_in_timestamp", "latitude", "longitude", "location_id"]
KEEP_COLS = ["latitude", "longitude"]
STORE_FOLDER = "/home/comete/mromanel/MILES_EXP/EXP_GEO_LOCATION_QIF_LIB_SETTING/"
utilities.createFolder(STORE_FOLDER)


def main_EXP_GEO_LOCATION_QIF_LIB_SETTING_retrieve_data_from_DB():
    ########################################################################################################################
    #############################################  SQUARES OF INTEREST  ####################################################
    ########################################################################################################################

    center = position_class.position(lat=CENTER_LAT, lon=CENTER_LON)

    small_square_min_lat, small_square_max_lat, small_square_min_lon, small_square_max_lon = \
        SOI_utilities.create_square_limit(
            central_position=center, side_length=SMALL_SQUARE_SIZE)

    SMALL_SQUARE = {"square_min_lat": small_square_min_lat, "square_max_lat": small_square_max_lat,
                    "square_min_lon": small_square_min_lon, "square_max_lon": small_square_max_lon}

    print(SMALL_SQUARE)
    sys.exit("EXIT")
    # big_square_min_lat, big_square_max_lat, big_square_min_lon, big_square_max_lon = \
    #    SOI_utilities.create_square_limit(
    #        central_position=center, side_length=BIG_SQUARE_SIZE)

    # BIG_SQUARE = {"square_min_lat": big_square_min_lat, "square_max_lat": big_square_max_lat,
    #              "square_min_lon": big_square_min_lon, "square_max_lon": big_square_max_lon}

    # print BIG_SQUARE

    ########################################################################################################################
    ########################################  SAMPLES IN THE SQUARES OF INTEREST  ##########################################
    ########################################################################################################################

    #   mat of coordinates (lat, lon)
    mat_from_db_records_of_interest = SOI_utilities.retrieve_samples_in_SOI_from_db(db_path=DATABASE_PATH,
                                                                                    separator='\t',
                                                                                    colnames=COLNAMES,
                                                                                    keep_cols=KEEP_COLS,
                                                                                    edges_dict=SMALL_SQUARE,
                                                                                    header=None)
    print("#checkins: " + str(mat_from_db_records_of_interest.shape[0]))
    print("head:\n ")
    print(mat_from_db_records_of_interest[0:5, :])

    ########################################################################################################################
    ####################################################  CELLS DICTIONARY  ################################################
    ########################################################################################################################

    cells_dict_small = SOI_utilities.create_square_grid(edges_dict=SMALL_SQUARE, cells_per_side=CELLS_PER_SIDE,
                                                        path_to_save=STORE_FOLDER)

    for key, val in cells_dict_small.items():
        print(key, "=>", val)

    ########################################################################################################################
    ##############################################  ASSOCIATE_LOCATIONS_TO_CELLS  ##########################################
    ########################################################################################################################

    list_cells = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", tqdm.TqdmSynchronisationWarning)
        for i_ter in tqdm.tqdm(range(mat_from_db_records_of_interest.shape[0])):
            # list_cells.append()
            lat = mat_from_db_records_of_interest[i_ter, 0]
            lon = mat_from_db_records_of_interest[i_ter, 1]
            list_cells.append(SOI_utilities.associate_location_to_cell(location=(lat, lon), cells_dic=cells_dict_small))

    heatmap = SOI_utilities.create_heatmap(cells_list=list_cells, cells_per_side=CELLS_PER_SIDE, path_to_save=STORE_FOLDER)

    print(heatmap)

    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    return
