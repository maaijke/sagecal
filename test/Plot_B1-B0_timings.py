#!/home/hspreeuw/anaconda3/bin/python

import numpy as np
from glob import glob
import os
import pylab as pyl

marker_size = 12 

number_of_datasets = 5

CPU_stations, CPU_median_times, GPU_stations, GPU_median_times  = np.zeros((4, number_of_datasets))

def compute_median_of_B1_minus_B0(B0_file, B1_file):

    B0_times = np.genfromtxt(B0_file, usecols = 4, skip_header = 12, skip_footer= 1)
    B1_times = np.genfromtxt(B1_file, usecols = 4, skip_header = 12, skip_footer= 1)

    return np.median(B1_times-B0_times)

CPU = 0
GPU = 0
path = "klok_resultaten/"

for filename in glob(path + "B0/" + "*.output"):
    # We want a list with just two strings: the number of stations and the processing device.
    filename_without_path = os.path.basename(filename)
    parsed_filename = filename_without_path.split(".")[0].split("_")
    processing_unit = parsed_filename[1]
    print("processing_unit = ", processing_unit)
    
    B0_file, B1_file = path + "B0/" + filename_without_path, path + "B1/" + filename_without_path
    
    if processing_unit == "CPU":
        CPU_stations[CPU] = parsed_filename[0]
        CPU_median_times[CPU]  = compute_median_of_B1_minus_B0(B0_file, B1_file)
        CPU += 1

    elif processing_unit == "GPU":
        GPU_stations[GPU] = parsed_filename[0]
        GPU_median_times[GPU]  = compute_median_of_B1_minus_B0(B0_file, B1_file)
        GPU += 1
    else:
        print("Something went wrong, unexpected file name.")
      
print([CPU_stations.min(), GPU_stations.min()]) 
print()
xmax = max([CPU_stations.max(), GPU_stations.max()])
pyl.xlim(0, 1.15 * xmax)

ymax = max([CPU_median_times.max(), GPU_median_times.max()])
pyl.ylim(1e-1, 2 * ymax)

pyl.xlabel("Number of stations", fontsize = marker_size)
pyl.ylabel("Array beam prediction time (s)", fontsize = marker_size)
pyl.semilogy(CPU_stations, CPU_median_times, 'gv', ms= marker_size, label = "CPU version of Sagecal")
pyl.semilogy(GPU_stations, GPU_median_times, 'ro', ms= marker_size, label = "GPU version of Sagecal")
pyl.legend(loc = 2)
# pyl.title("Sagecal array beam prediction times for five numbers of stations", fontsize = int(marker_size))
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
pyl.text(60, 1, "LOFAR", ha="center", va="center", size=20, bbox=bbox_props)
pyl.text(512, 13, "SKA", ha="center", va="center", size=20, bbox=bbox_props)
# pyl.show()
pyl.savefig(path + "Sagecal_array_beam_prediction_times.png", bbox_inches = "tight")
