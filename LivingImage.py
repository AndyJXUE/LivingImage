import numpy as np
from functions import process_recursively, save_result, save_subs, save_gephi, save_centers
from PIL import Image

data_name = "3-1"  # THE OUTPUT FOLDER NAME OF YOUR IMAGE
image_path = r"Sample\Huan\%s.png" % data_name  # THE INPUT IMAGE PATH

output_csv_path = "Results/%s.csv" % data_name
output_hie_path = "Subs/%s.png" % data_name
output_cen_path = "Cens/%s.shp" % data_name
output_gephi_dir = "Gephi/%s/" % data_name

print("--------------------- Processing ---------------------")
focus = "light"  # "light" or "dark"

image = Image.open(image_path)
inputraster = np.array(image).astype(np.int64)

if image.mode == 'RGBA':

    alpha_channel = inputraster[:, :, 3]
    transparent_pixels = alpha_channel == 0

    if len(inputraster.shape) > 2:
        inputraster = np.dot(inputraster[..., 0:3], [0.2989, 0.5870, 0.1140])
    inputraster[transparent_pixels] = -1
else:
    if len(inputraster.shape) > 2:
        inputraster = np.dot(inputraster[..., 0:3], [0.2989, 0.5870, 0.1140])


# Process the image recursively
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

# Save the results
# save_result(output_csv_path, d_array, s_array, lr_array, i)
# save_subs(output_hie_path, output_list, image, xy_list, mode='gray')
# save_gephi(output_gephi_dir, flattened_sources, flattened_targets, d_indexes)
save_centers(output_cen_path, xy_list)