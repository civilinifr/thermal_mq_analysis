"""
Combines the generated output of the finetuning and PGV into a single catalog.
This catalog is later updated using azimuth.

"""
# Import packages
import glob
import pandas as pd
import os

# Functions
def cat_wrapper(input_list):

    for input_file in input_list:
        if input_file == input_list[0]:
            full_cat = pd.read_csv(input_file)
        else:
            df2 = pd.read_csv(input_file)
            frames = [full_cat, df2]
            full_cat = pd.concat(frames)

    return full_cat


# Main
out_dir = 'C:/data/lunar_output/'
input_directory = f'{out_dir}fine-tuned/cat_stats/'
output_directory = f'{out_dir}catalogs/'
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

filelist = sorted(glob.glob(f'{input_directory}*.csv'))
catalog = cat_wrapper(filelist)
catalog.to_csv(f'{output_directory}GradeA_thermal_mq_catalog.csv', index=False)
print('Saved the catalog!')

