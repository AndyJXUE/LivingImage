import numpy as np
from functions import process_recursively, save_result, save_subs
from PIL import Image

data_name = "1"  # THE OUPUT FOLDER NAME OF YOUR IMAGE
image_path = r"Sample\7-1.png"  # THE INPUT IMAGE PATH

output_csv_path = "Results/%s.csv" % (data_name)
output_hie_path = "Subs/%s.png" % (data_name)

print("--------------------- Processing ---------------------")
focus = "dark"  # "light" or "dark"

inputraster = np.array(Image.open(image_path)).astype(np.int64)
if len(inputraster.shape) > 2:
    inputraster = np.dot(inputraster[..., 0:3], [0.2989, 0.1140, 0.5870])

results = process_recursively(inputraster, focus)

i = len(results)
d_array = np.array([item['d'] for item in results])
s_array = np.array([item['s'] for item in results])
lr_array = np.array([item['lr'] for item in results])
output_list = [item['output'] for item in results]
xy_list = [item['xy'] for item in results]

decs = np.array(d_array).sum()
v = i * decs
lr = lr_array.sum()
print("--------------------- Finished ---------------------")
print("Final result: lr = %d, V = %d" % (lr, v))

save_result(output_csv_path, d_array, s_array, lr_array, i)
save_subs(output_hie_path, output_list, inputraster, xy_list)
