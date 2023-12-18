import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import gaussian_kde

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

def Clip(img, lb_images, stats, centers, xy_last, thre=0.00001, focus="dark"):
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

def process_hierarchy(regions_list, xy_last_list, focus, break_per=0.4, first=False):
    return_list, centers_list, xy_list, d_index, areas_list = [], [], [], [], []
    s, d, lr = 0, 0, 0

    with tqdm(total=len(regions_list), desc="Processing regions") as pbar:
        for i, region in enumerate(regions_list):
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
                d_index.append(i)
                regions_list_new, centers, xy = Clip(
                    region, lb_images, stats, centers, xy_last_list[i], focus=focus 
                )
                return_list.extend(regions_list_new)
                centers_list.extend(centers)
                xy_list.extend(xy)
                d += 1

                s += num_labels
                lr += ht_index * num_labels
                areas_list.extend(areas)
            pbar.update(1)

    return return_list, d, s, lr, xy_list, d_index,

def process_recursively(inputraster, focus):
    i = 0
    xy = np.zeros(2, dtype=int)
    results = []

    while True:
        return_list, d, s, lr, xy, d_index = process_hierarchy(
            [inputraster] if i == 0 else return_list, xy, focus)

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
            'xy': xy
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

def heatmap(heat_path, fcenters_list, inputraster, width, height):
    folder, _ = os.path.split(heat_path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    # 将网格点展平
    grid_points = np.vstack((x_grid.ravel(), y_grid.ravel()))
    # 提取坐标点
    coordinate_points = np.array(fcenters_list).T
    # 计算核密度估计
    kde = gaussian_kde(coordinate_points, 0.1)
    # 在网格上计算核密度值
    density_values = kde(grid_points)
    density_map = density_values.reshape(height, width)
    # 对密度图进行平滑以获得更圆润的结果
    # density_map = gaussian_filter(density_map, sigma=5)
    # 将密度值映射到对数空间
    log_density_map = np.log(density_map + 1)  # 加1以避免log(0)

    # 找到最小和最大的对数密度值
    min_log_density = np.min(log_density_map)
    max_log_density = np.max(log_density_map)
    mask = (
        (log_density_map - min_log_density) / (max_log_density - min_log_density)
    ) * 255
    # mask = 100000*((density_map-density_map.min())/(density_map.max()-density_map.min())).astype(np.uint8)
    heat_img = cv2.applyColorMap(
        mask.astype(np.uint8), cv2.COLORMAP_JET
    )  # 此处的三通道热力图是cv2使用GBR排列
    inputraster[inputraster == -1] = 255
    inputraster = cv2.cvtColor(inputraster.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    add_img = cv2.addWeighted(inputraster, 0.3, heat_img, 0.7, 0)
    cv2.imwrite(heat_path, add_img)

def write_decs(output_hie_path, output, inputraster, lb_images, xy_list, d_index_list):
    # All substructures, except H1
    folder, _ = os.path.split(output_hie_path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    base = np.zeros(np.shape(inputraster)).astype(np.int8)
    for i in range(0, len(output) - 1):
        himg = np.zeros(np.shape(inputraster)).astype(np.int8)
        d = d_index_list[i + 1]
        lr = output[i]
        for j in d:
            c = lr[j]
            # 获取当前影像和坐标位置
            ci = np.where(c >= 0, i + 1, 0).astype(np.int8)
            y, x = xy_list[i][j]
            # 将当前影像插入到新影像中
            himg[x : x + ci.shape[0], y : y + ci.shape[1]] = ci
        base[himg != 0] = 0
        if i == 0:
            base += himg
        else:
            base[himg != 0] = 0
            base = base + himg
    cv2.imwrite(output_hie_path, base)

def apply_colormap(input_image, colormap_name):
    cmap = plt.get_cmap(colormap_name)
    colored_image = (cmap(input_image) * 255).astype(np.uint8)
    return colored_image

def save_subs(output_hie_path, output, inputraster, xy_list):
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
    base = base/base.max()

    colored_image = apply_colormap(base, "Spectral")
    cv2.imwrite(output_hie_path, colored_image)