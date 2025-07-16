import numpy as np
from pyCGM2.Utils import files

from pyCGM2.Mek import mekTools

import matplotlib.pyplot as plt

def gather(data_group,label):
    count = 0

    list_of_arrays = []
    for groupPath, set_obj in mekTools.iter_sets(data_group):
        if label in groupPath and "Cycle" in groupPath:
            print(f"Set de {label} detected in : {groupPath}")
            setDetect = data_group.retrieve_set(data_group.name()+groupPath)
            name= setDetect.name()
            values = setDetect.read()
            list_of_arrays.append(values)
            count+=1
    array_3d = np.stack(list_of_arrays, axis=0)

    return array_3d

def plot(rootGroup, layoutPathFile):
    config = files.openFile(None,layoutPathFile)
    

    rows = config['rows']
    cols = config['cols']

 
    fig, axes = plt.subplots(rows, cols, figsize=(8.27,11.69), dpi=100, facecolor="white")
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    fig.suptitle("Time-normalized Muscle Plot", fontsize=14)


    for ax in axes.flat:
        ax.set_ylabel("angle (deg)",size=8)
        ax.tick_params(axis='x', which='major', labelsize=6)
        ax.tick_params(axis='y', which='major', labelsize=6)

        # ax.set_xlabel("Cycle %",size=8)



    for i, plot_cfg in enumerate(config['plots']):
        row, col = plot_cfg['position']
        if rows == 1 and cols == 1:
            ax = axes
        elif rows == 1 or cols == 1:
            ax = axes[max(row, col)]
        else:
            ax = axes[row][col]

        for curve in plot_cfg.get('curves', []):
            label_data = curve['data']
            label, axis_str = label_data.split(".")
            axis = int(axis_str)
            color = curve.get('color')
            legend_label = curve.get('label')

            values = gather(rootGroup, label)
            data = values[:, :, axis].mean(axis=0)

            ax.plot(data, label=legend_label, color=color)
        
        ax.set_title(plot_cfg['title'],size=8)
        ax.set_ylabel(plot_cfg['ylabel'],size=8)
        ax.set_ylim(plot_cfg['ylim'])
        ax.legend(fontsize=6, loc="upper right")


    plt.tight_layout()
    plt.show()




