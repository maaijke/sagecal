#!/home/hspreeuw/anaconda3/bin/python

import numpy as np
from glob import glob
import os
import pylab as pyl
from mpl_toolkits.mplot3d import Axes3D
from pylab import cm

marker_size = 10 

numbers_of_stations = np.array([64, 128, 256, 384, 512]) 
# numbers_of_stations = np.array([64, 128, 256, 384]) 
numbers_of_sources = np.arange(1,6)*10000

CPU_median_times, GPU_median_times  = np.zeros((2, numbers_of_stations.size, numbers_of_sources.size))

def compute_median(some_file):

    times = np.genfromtxt(some_file, usecols = 4, skip_header = 12, skip_footer= 1)
    return np.median(times)

path_to_timings = "klok_resultaten/B1/varying_number_of_sources/"

for filename in glob(path_to_timings + "*.output"):
    # We want a list with just two strings: the number of stations and the processing device.
    parsed_filename = os.path.basename(filename).split(".")[0].split("_")

    # print()
    # print("np.where(numbers_of_stations == parsed_filename[0]) = ", np.where(numbers_of_stations == int(parsed_filename[0])))
    # print("parsed_filename[0] = ", parsed_filename[0])
    numbers_of_stations_index = np.where(numbers_of_stations == int(parsed_filename[0]))[0][0]
    numbers_of_sources_index = np.where(numbers_of_sources == int(parsed_filename[1]))[0][0]

    processing_unit = parsed_filename[2]
    print("processing_unit = ", processing_unit)
    
    if processing_unit == "CPU":
        CPU_median_times[numbers_of_stations_index, numbers_of_sources_index]  = compute_median(filename)

    elif processing_unit == "GPU":
        GPU_median_times[numbers_of_stations_index, numbers_of_sources_index]  = compute_median(filename)
    else:
        print("Something went wrong, unexpected file name.")
      
fig = pyl.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(numbers_of_stations, numbers_of_sources)
surf = ax.plot_surface(X, Y, GPU_median_times, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
pyl.show()
# xmax = max([CPU_stations.max(), GPU_stations.max()])
# pyl.xlim(0, 1.15 * xmax)
# 
# ymax = max([CPU_median_times.max(), GPU_median_times.max()])
# pyl.ylim(3e-1, 2 * ymax)
# 
# pyl.xlabel("Number of stations", fontsize = marker_size)
# pyl.ylabel("Sky model conversion plus beam prediction time (s)", fontsize = marker_size)
# pyl.semilogy(CPU_stations, CPU_median_times, 'gv', ms= marker_size, label = "CPU version of SAGECal")
# pyl.semilogy(GPU_stations, GPU_median_times, 'ro', ms= marker_size, label = "GPU version of SAGECal")
# pyl.legend(loc = 2)
# # pyl.title("Sagecal sky model conversion plus beam prediction times for five numbers of stations", fontsize = int(marker_size))
# bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
# pyl.text(60, 1.8, "LOFAR", ha="center", va="center", size=20, bbox=bbox_props)
# pyl.text(512, 40, "SKA", ha="center", va="center", size=20, bbox=bbox_props)
# # pyl.show()
# pyl.savefig(path_to_B1 + "SAGECal_sky_model_conversion_plus_beam_prediction_times.pdf", bbox_inches = "tight")
