import numpy as np

from functions import process_recursively, save_result, save_subs_geo, save_gephi, save_centers
from osgeo import gdal

data_name = "2022"  # THE OUPUT FOLDER NAME OF YOUR IMAGE
image_path = r"Sample\NTL\%s.tif"%(data_name)  # THE INPUT IMAGE PATH

output_csv_path = "Results/%s.csv" % (data_name)
output_hie_path = "Subs/%s.tif" % (data_name)
output_cen_path = "Cens/%s.shp" % data_name
output_gephi_dir = "Gephi/%s/" % data_name

print("--------------------- Processing ---------------------")
focus = "light"  # "light" or "dark"

header = gdal.Open(image_path)
image = header.ReadAsArray()
projection = header.GetProjection()
geotransform = header.GetGeoTransform()

nodata = header.GetRasterBand(1).GetNoDataValue()
inputraster = image.astype(np.int64)
inputraster[image == nodata] = -1

results = process_recursively(inputraster, focus)

i = len(results)
d_array = np.array([item['d'] for item in results])
s_array = np.array([item['s'] for item in results])
lr_array = np.array([item['lr'] for item in results])
output_list = [item['output'] for item in results]
xy_list = [item['xy'] for item in results]
sources = [item['sources'] for item in results]
targets = [item['targets'] for item in results]
d_indexes = [item['d_index'] for item in results]

flattened_sources = [source for sublist in sources for source in sublist]
flattened_targets = [target for sublist in targets for target in sublist]

decs = np.array(d_array).sum()
v = i * decs
lr = lr_array.sum()
print("--------------------- Finished ---------------------")
print("Final result: lr = %d, V = %d" % (lr, v))

save_result(output_csv_path, d_array, s_array, lr_array, i)
save_subs_geo(output_hie_path, output_list, inputraster, xy_list, projection, geotransform, mask=(image == nodata))
save_gephi(output_gephi_dir, flattened_sources, flattened_targets, d_indexes)
save_centers(output_cen_path, xy_list, projection, geotransform)