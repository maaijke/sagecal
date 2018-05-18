#!/home/hspreeuw/anaconda3/bin/python

import numpy as np
from glob import glob
import os
import pylab as pyl

marker_size = 12 

number_of_datasets = 5

CPU_stations, CPU_median_times, GPU_stations, GPU_median_times  = np.zeros((4, number_of_datasets))

def compute_median(some_file):

    times = np.genfromtxt(some_file, usecols = 4, skip_header = 12, skip_footer= 1)
    return np.median(times)

CPU = 0
GPU = 0
path_to_B1 = "klok_resultaten/B1/"

for filename in glob(path_to_B1 + "*.output"):
    # We want a list with just two strings: the number of stations and the processing device.
    parsed_filename = os.path.basename(filename).split(".")[0].split("_")
    processing_unit = parsed_filename[1]
    print("processing_unit = ", processing_unit)
    
    if processing_unit == "CPU":
        CPU_stations[CPU] = parsed_filename[0]
        CPU_median_times[CPU]  = compute_median(filename)
        CPU += 1

    elif processing_unit == "GPU":
        GPU_stations[GPU] = parsed_filename[0]
        GPU_median_times[GPU]  = compute_median(filename)
        GPU += 1
    else:
        print("Something went wrong, unexpected file name.")
      
print([CPU_stations.min(), GPU_stations.min()]) 
print()
xmax = max([CPU_stations.max(), GPU_stations.max()])
pyl.xlim(0, 1.15 * xmax)

ymax = max([CPU_median_times.max(), GPU_median_times.max()])
pyl.ylim(3e-1, 2 * ymax)

pyl.xlabel("Number of stations", fontsize = marker_size)
pyl.ylabel("Sky model conversion plus beam prediction time (s)", fontsize = marker_size)
pyl.semilogy(CPU_stations, CPU_median_times, 'gv', ms= marker_size, label = "CPU version of Sagecal")
pyl.semilogy(GPU_stations, GPU_median_times, 'ro', ms= marker_size, label = "GPU version of Sagecal")
pyl.legend(loc = 2)
pyl.title("Sagecal sky model conversion plus beam prediction times for five numbers of stations", fontsize = int(marker_size))
# pyl.show()
pyl.savefig(path_to_B1 + "Sagecal_sky_model_conversion_plus_beam_prediction_times.png", bbox_inches = "tight")
