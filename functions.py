import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from osgeo import gdal, ogr, osr
from tqdm import tqdm

def percent_clip(img_array, percent):
    # 将图像转换为numpy数组

    # 计算图像的最大和最小值
    max_val = np.max(img_array)
    min_val = np.min(img_array)

    # 计算要裁剪的最大和最小值
    clip_max = max_val - (max_val - min_val) * percent / 100.0
    clip_min = min_val + (max_val - min_val) * percent / 100.0

    # 将图像裁剪到指定的最大和最小值之间
    img_array = np.clip(img_array, clip_min, clip_max)

    return img_array

def BinarizeRaster(inputraster, focus="dark"):
    array = np.ravel(inputraster)

    idx = np.where(array == -1)
    array = np.delete(array, idx)
    if len(array):
        in_raster_mean = np.mean(array)
    else:
        in_raster_mean = 0

    # in_raster_max = np.max(array)

    if focus == "light":
        reclass = np.where(
            inputraster > in_raster_mean, 1, 0
        )  # change this for dark or white focus. 1,0 for dark (Lena), 0,1 for light (nighttime)
    else:
        reclass = np.where(
            (inputraster <= in_raster_mean) & (inputraster != -1), 1, 0
        )  # change this for dark or white focus. 1,0 for dark (Lena), 0,1 for light (nighttime)
    reclass = np.array(reclass, dtype=np.uint8)

    del array, idx
    return reclass

def head_tail_breaks(array, break_per=0.4):
    rats = []
    rat_in_head, head_mean, ht_index = 0, 0, 1

    array = np.ravel(array)

    while (rat_in_head <= break_per) and (len(array) > 1) and np.mean(array) > 0:
        mean = np.mean(array)

        head_mean = array[array > mean]

        count_total = len(array)
        count_head_mean = len(head_mean)

        rat = count_head_mean / count_total
        rats.append(rat)

        if rat_in_head == 0:
            rat_in_head = rat
        else:
            rat_in_head = np.mean(rats)

        if rat_in_head < break_per:  # adjust ht breaks seperator
            ht_index += 1
        array = head_mean
    # ht_index = len(record)
    del head_mean, array, rats
    return ht_index

def Clip(img, lb_images, stats, centers, xy_last, thre=0.00001):
    # 获取所有连通区域的面积
    areas = stats[:, cv2.CC_STAT_AREA]

    # 获取所有符合要求的区域的索引
    large_regions_indices = np.where(areas >= thre)[0]

    # 初始化空列表
    regions_list, centers_list, original_xy = [], [], []

    # 循环处理每个连通区域，并使用 tqdm 跟踪处理进度
    # with tqdm(total=len(large_regions_indices), desc='Clipping regions') as pbar:
    for i in large_regions_indices:
        # 计算当前连通区域的边界框，并将其添加到列表中
        x, y, w, h, _ = stats[i]

        region = img[y : y + h, x : x + w]
        region_lb = lb_images[y : y + h, x : x + w]

        region = np.where(region_lb == (i + 1), region, -1)

        xy = np.array([x, y])
        centroid = centers[i].astype(int)

        # if region.shape!=img.shape:
        # 将提取的连通区域添加到列表中
        regions_list.append(region)
        centers_list.append(centroid + xy_last)
        original_xy.append(xy + xy_last)
        # pbar.update(1)
    del img, lb_images, stats, centers
    return regions_list, centers_list, original_xy

def process_hierarchy(regions_list, xy_last_list, focus, nodes, break_per=0.4, first=False):
    return_list, centers_list, xy_list, d_index, areas_list, sources, targets = [], [], [], [], [], [], []
    s, d, lr = 0, 0, 0

    with tqdm(total=len(regions_list), desc="Processing regions") as pbar:
        for i, region in enumerate(regions_list):

            node = nodes[i]
            bin_image = BinarizeRaster(region, focus)

            num_labels, lb_images, stats, centers = cv2.connectedComponentsWithStats(
                bin_image, connectivity=4
            )
            num_labels -= 1
            stats = stats[1:]
            centers = centers[1:]

            areas = np.sqrt(stats[:, cv2.CC_STAT_AREA])  # Remove background
            ht_index = head_tail_breaks(areas, break_per)

            if ht_index > 2 or first:

                d_index.append(node)
                regions_list_new, centers, xy = Clip(
                    region, lb_images, stats, centers, xy_last_list[i] 
                )
                
                sources.extend([node] * len(regions_list_new))
                targets.extend([ nodes[-1] + i + 1 + len(return_list) for i in range(len(regions_list_new))])

                return_list.extend(regions_list_new)
                centers_list.extend(centers)
                xy_list.extend(xy)
                d += 1

                s += num_labels
                lr += ht_index * num_labels
                areas_list.extend(areas)

            pbar.update(1)
    nodes = [i + nodes[-1] + 1 for i in range(len(return_list))]
    return return_list, d, s, lr, xy_list, d_index, nodes, sources, targets

def process_recursively(inputraster, focus):
    i = 0
    xy = np.zeros(2, dtype=int)
    results, nodes = [], [1]

    while True:
        return_list, d, s, lr, xy, d_index, nodes, sources, targets = process_hierarchy(
            [inputraster] if i == 0 else return_list, xy, focus, nodes)

        if len(return_list) == 0:
            break

        print(f"--------------------- hierarchy {i+1} has done ---------------------")
        print(f"Hierarchy {i}, D={d}, S={s}, lr={lr}")

        i += 1
        results.append({
            'output': return_list,
            's': s,
            'd': d,
            'lr': lr,
            'd_index': d_index,
            'xy' :xy,
            'sources': sources,
            'targets': targets
        })

    return results

def save_result(csv_path, d_array, s_array, lr_array, i):

    folder, _ = os.path.split(csv_path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    h_array = lr_array / s_array
    i_array = np.arange(1, i+1).astype(int)
    
    u_array = np.zeros(i)
    u_array[:-1] = s_array[:-1] - d_array[1:]
    u_array[-1] = s_array[-1]
    
    per_array = (s_array-u_array) / s_array

    # Create a DataFrame directly from the arrays
    df = pd.DataFrame({
        "I": i_array,
        "D": d_array,
        "S": s_array,
        "H": h_array.round(2),  # Round 'H' to two decimal places
        "U": u_array.astype(int),
        "%": per_array.round(2),
        "LR": lr_array
    })

    # Write the DataFrame to a CSV file
    df.to_csv(csv_path, index=False)

def apply_colormap(input_image, colormap_name):
    cmap = plt.get_cmap(colormap_name)
    colored_image = (cmap(input_image) * 255).astype(np.uint8)
    return colored_image

def save_subs(output_hie_path, output, inputraster, xy_list, mask=None, mode="rgb"):
    folder, _ = os.path.split(output_hie_path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    himgs = []
    for i, lr in enumerate(output):
        himg = np.zeros_like(inputraster, dtype=np.int8)

        for j, current_image in enumerate(lr):
            y, x = xy_list[i][j]
            current_image = np.where(current_image > 0, i + 1, 0).astype(np.int8)
            x_end, y_end = x + current_image.shape[0], y + current_image.shape[1]
            himg[x:x_end, y:y_end] = current_image

        himgs.append(himg)

    base = np.zeros_like(inputraster, dtype=np.int8)
    for h in himgs:
        base = np.where(h != 0, h, base)

    if mode == 'rgb':
        base = base/base.max()
        colored_image = apply_colormap(base, "Spectral")
        cv2.imwrite(output_hie_path, colored_image)
        
    elif mode == 'gray':
        cv2.imwrite(output_hie_path, base)

def save_subs_geo(output_hie_path, output, inputraster, xy_list, projection, geotransform, mask=None):

    folder, _ = os.path.split(output_hie_path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    himgs = []
    for i, lr in enumerate(output):
        himg = np.zeros_like(inputraster, dtype=np.int8)

        for j, current_image in enumerate(lr):
            y, x = xy_list[i][j]
            current_image = np.where(current_image > 0, i + 1, 0).astype(np.int32)
            x_end, y_end = x + current_image.shape[0], y + current_image.shape[1]
            himg[x:x_end, y:y_end] = current_image

        himgs.append(himg)

    base = np.zeros_like(inputraster, dtype=np.int8)
    for h in himgs:
        base = np.where(h != 0, h, base)

    if mask is not None:
        base[mask] = 255

    driver = gdal.GetDriverByName('GTiff')
    output_dataset = driver.Create(output_hie_path, base.shape[1], base.shape[0], 1, gdal.GDT_Int16)

    output_dataset.SetGeoTransform(geotransform)
    output_dataset.nodata = 255
    output_dataset.SetProjection(projection)
    output_dataset.GetRasterBand(1).WriteArray(base)
    output_dataset.FlushCache()

def save_gephi(output_gephi, sources, targets, d_indexes):
    if not os.path.exists(output_gephi):
        os.makedirs(output_gephi)

    output_gephi_edge = os.path.join(output_gephi, "edge.csv")
    output_gephi_node_dec = os.path.join(output_gephi, "edge_dec.csv")

    df_edge = pd.DataFrame({
        "Source": sources,
        "Target": targets,
        "I": [1] * len(sources),
    })

    f_indexes = [d_index for sublist in d_indexes for d_index in sublist]
    valid_nodes = set(f_indexes)
    df_edge_dec = df_edge[df_edge["Source"].isin(valid_nodes) & df_edge["Target"].isin(valid_nodes)]
    
    for d, d_list in enumerate(d_indexes):
        for i in range(len(df_edge_dec['I'])):
            s = sources[i]
            if s in d_list:
                df_edge_dec.iloc[i]['I'] = d + 1
                
    df_edge.to_csv(output_gephi_edge, index=False)
    df_edge_dec.to_csv(output_gephi_node_dec, index=False)

def save_centers(output_cen_path, xy_list, projection=None, geotransform=None):
    # Ensure the output directory exists
    folder, _ = os.path.split(output_cen_path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Create a new Shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.CreateDataSource(output_cen_path)
    layer = ds.CreateLayer(output_cen_path, geom_type=ogr.wkbPoint)
    
    # Define fields (h, x, y)
    layer.CreateField(ogr.FieldDefn('h', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('x', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('y', ogr.OFTReal))

    for h, xys in enumerate(xy_list):
        for xy in xys:
            x, y = xy  # Extract x and y coordinates

            if geotransform is not None:
                # Apply geotransformation
                x_map = geotransform[0] + x * geotransform[1] + y * geotransform[2]
                y_map = geotransform[3] + x * geotransform[4] + y * geotransform[5]
            else:
                # If no geotransform, use x and y as they are
                x_map = float(x)
                y_map = float(y)

            # Create a new feature
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField('h', h)
            feature.SetField('x', x_map)
            feature.SetField('y', y_map)
            
            # Create point geometry
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(x_map, y_map)
            feature.SetGeometry(point)
            
            # Add the feature to the layer
            layer.CreateFeature(feature)
            
    # If a projection is provided, set the spatial reference
    if projection is not None:
        layer.GetSpatialRef().ImportFromWkt(projection)
    
    # Cleanup
    ds = None
