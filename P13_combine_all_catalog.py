"""
Combines the catalog created in P8 with the azimuth results
"""
# Import packages
import glob
import pandas as pd
import os
import numpy as np
pd.options.mode.chained_assignment = None


# Functions
def combine_azi(input_list):
    """
    Combines the azimuth results into the new dataframe

    :param input_list: [vector] List of input files to add
    :return:
    """

    evid_list = []
    for input_file in input_list:
        input_file_bn = os.path.basename(input_file)
        evid_list.append(input_file_bn.split('_')[0])

        if input_file == input_list[0]:
            full_cat = pd.read_csv(input_file)
        else:
            df2 = pd.read_csv(input_file)
            frames = [full_cat, df2]
            full_cat = pd.concat(frames)
    full_cat.insert(0, "evid", evid_list, True)

    return full_cat


def combine_cats(cat1_df, cat2_df):
    """
    Combines the previously generated catalog with the one containing the azimuth results

    :param cat1_df: [pd df] Original catalog without the azimuth
    :param cat2_df: [pd df] New catalog with the azimuth results
    :return:
    """

    # Create new rows in cat1 corresponding to the cat2 columns
    cat1_df["theta_mean"] = np.nan
    cat1_df["theta_variance"] = np.nan
    cat1_df["avg_misfit"] = np.nan

    evid_list = np.unique(cat1_df['evid'].values)
    for evid in evid_list:
        # Find the indices in cat1 corresponding to that evid
        evid_ind_cat1 = np.where(cat1_df['evid'].values == evid)[0]
        # print(evid_ind_cat1)

        if len(np.where(np.array(evids_to_exclude) == evid)[0]) > 0:
            cat1_df["theta_mean"][ind_index] = 0
            cat1_df["theta_variance"][ind_index] = 0
            cat1_df["avg_misfit"][ind_index] = 0
        else:
            # Find the appropriate evid in cat2
            evid_ind_cat2 = np.where(cat2_df['evid'].values == evid)[0]
            theta_mean_val = cat2_df["theta_mean"].values[evid_ind_cat2]
            theta_variance_val = cat2_df["theta_variance"].values[evid_ind_cat2]
            avg_misfit_val = cat2_df["avg_misfit"].values[evid_ind_cat2]

            for ind_index in evid_ind_cat1:
                cat1_df["theta_mean"][ind_index] = theta_mean_val
                cat1_df["theta_variance"][ind_index] = theta_variance_val
                cat1_df["avg_misfit"][ind_index] = avg_misfit_val

    # Save the result
    cat1_df.to_csv(f'{catdir}GradeA_thermal_mq_catalog_final.csv', index=False)
    print(f'Finished saving the thermal mq catalog!')
    return


# Main
out_dir = 'C:/data/lunar_output/'
catdir = f'{out_dir}catalogs/'
input_catalog = f'{catdir}GradeA_thermal_mq_catalog.csv'
azimuth_results = f'{out_dir}results/locations/azimuth_combined/data/'

# If you want to exclude some evids from the catalog, replace below with the event IDs.
evids_to_exclude = ['999999-99-99']

# Get the
azimuth_files = sorted(glob.glob(f'{azimuth_results}*.csv'))
azi_df = combine_azi(azimuth_files)

# Load the original catalog
cat_df = pd.read_csv(input_catalog)

# Combine the two catalogs
combine_cats(cat_df, azi_df)
