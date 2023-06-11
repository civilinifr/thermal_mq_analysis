"""
Plots PGV information for the dataset.
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def main():
    """
    Main function for histogram plotting
    :return:
    """
    # Setup the directories
    out_dir = 'C:/data/lunar_output/'
    results_dir = f'{out_dir}results/'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    pgv_results_dir = f'{results_dir}PGV/'
    if not os.path.exists(pgv_results_dir):
        os.mkdir(pgv_results_dir)

    # Read in the catalog
    catfile = f'{out_dir}catalogs/GradeA_thermal_mq_catalog_final.csv'
    df = pd.read_csv(catfile)

    evids_to_exclude = ['770425-00-M1', '761111-21-M2', '770416-10-M1', '770325-13-M6', '770114-15-M1',
                        '761105-08-M1', '770314-22-M1', '761021-07-M1', '760901-14-M2']

    total_rows_to_delete = []
    for evid in evids_to_exclude:
        evid_row = np.where(df['evid'].values == evid)[0]
        for evid_row_ind in evid_row:
            total_rows_to_delete.append(evid_row_ind)

    df = df.drop(total_rows_to_delete)

    pgv = df['PGV'].values
    pgv_geo1 = df[df['geophone'] == 'geo1']['PGV'].values
    pgv_geo2 = df[df['geophone'] == 'geo2']['PGV'].values
    pgv_geo3 = df[df['geophone'] == 'geo3']['PGV'].values
    pgv_geo4 = df[df['geophone'] == 'geo4']['PGV'].values


    # Plot the image
    fig = plt.figure(figsize=(10, 10))
    ax0 = plt.subplot2grid(shape=(4, 2), loc=(0, 0), rowspan=2, colspan=2, fig=fig)
    ax0.scatter(np.arange(len(pgv_geo2)), pgv_geo2, edgecolors='black', color='orange')
    ax0.scatter(np.arange(len(pgv_geo3)), pgv_geo3, edgecolors='black', color=lighten_color('green', 1.1))
    ax0.scatter(np.arange(len(pgv_geo4)), pgv_geo4, edgecolors='black', color='gray')
    ax0.scatter(np.arange(len(pgv_geo1)), pgv_geo1, edgecolors='black')
    # ax0.set_ylim((0, 500))
    ax0.set_yscale('log')
    ax0.set_xlim((0, len(pgv_geo1)))
    ax0.set_ylabel('PGV (nm/s)', fontweight='bold')
    ax0.set_xlabel('Evid', fontweight='bold')
    ax0.set_title('PGV Values All', fontweight='bold')

    ax1 = plt.subplot2grid(shape=(4, 2), loc=(2, 0), rowspan=1, colspan=1, fig=fig)
    ax1.hist(pgv_geo1, bins=10, range=(0, 500), edgecolor='black')
    ax1.set_ylim((0, 1400))
    ax1.set_xlabel('PGV (nm/s)', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title('Geo1', fontweight='bold')

    ax2 = plt.subplot2grid(shape=(4, 2), loc=(2, 1), rowspan=1, colspan=1, fig=fig)
    ax2.hist(pgv_geo2, bins=10, range=(0, 500), color='orange', edgecolor='black')
    ax2.set_ylim((0, 1400))
    ax2.set_xlabel('PGV (nm/s)', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Geo2', fontweight='bold')

    ax3 = plt.subplot2grid(shape=(4, 2), loc=(3, 0), rowspan=1, colspan=1, fig=fig)
    ax3.hist(pgv_geo3, bins=10, range=(0, 500), color=lighten_color('green', 1.1), edgecolor='black')
    ax3.set_ylim((0, 1400))
    ax3.set_xlabel('PGV (nm/s)', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('Geo3', fontweight='bold')

    ax4 = plt.subplot2grid(shape=(4, 2), loc=(3, 1), rowspan=1, colspan=1, fig=fig)
    ax4.hist(pgv_geo4, bins=10, range=(0, 500), color='gray', edgecolor='black')
    ax4.set_ylim((0, 1400))
    ax4.set_xlabel('PGV (nm/s)', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('Geo4', fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{pgv_results_dir}pgv_cumulative_vals.png')
    fig.savefig(f'{pgv_results_dir}pgv_cumulative_vals.eps')
    return


main()
