import numpy as np
from functions import ProcessHierachy, save_result, write_centroids, write_subs
from PIL import Image

data_name = "1"  # THE OUPUT FOLDER NAME OF YOUR IMAGE
image_path = "3.jpg"  # THE INPUT IMAGE PATH

output_csv_path = "table/%s.csv" % (data_name)
output_hie_path = "hie/%s.png" % (data_name)
output_points_path = "points/%s.shp" % (data_name)

print("--------------------- Processing ---------------------")
focus = "light"  # "light" or "dark"


inputraster = np.array(Image.open(image_path)).astype(np.int64)
if len(inputraster.shape) > 2:
    inputraster = np.dot(inputraster[..., 0:3], [0.2989, 0.1140, 0.5870])

(
    return_list,
    s_list,
    d_list,
    lr_list,
    area_list,
    centers_list,
    output,
    d_index_list,
    xy_list,
) = ([], [], [], [], [], [], [], [], [])
xy = np.zeros(2, dtype=int)
shape = np.shape(inputraster)


i, d, decs = 1, 1, 0


while d != 0:
    if i == 1:
        (
            return_list,
            _,
            s,
            lr,
            centroid,
            xy,
            d_index,
            lb_images,
            areas,
        ) = ProcessHierachy([inputraster], xy, focus)
    else:
        return_list, d, s, lr, centroid, xy, d_index, _, areas = ProcessHierachy(
            return_list, xy, focus
        )

    if len(return_list) != 0:
        print("--------------------- hierachy %d has done ---------------------" % (i))
        print("Hierachy %d, D=%d, S=%d, lr=%d" % (i, d, s, lr))

        i += 1
        s_list.append(s)
        lr_list.append(lr)
        d_list.append(d)
        xy_list.append(xy)

    centers_list.append(np.array(centroid))
    d_index_list.append(d_index)
    output.append(return_list)
    area_list.append(areas)

    if len(return_list) == 0 and i == 1:
        break

decs = np.array(d_list).sum()
v = i * decs
print("Final result: lr = %d, V = %d" % (np.array(lr_list).sum(), v))


# save_result(output_csv_path, d_list, s_list, lr_list, i)
# write_centroids(output_points_path, centers_list, d_index_list, area_list)
# write_subs(output_hie_path, output, inputraster, lb_images, xy_list)
