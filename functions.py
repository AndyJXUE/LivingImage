import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from osgeo import ogr
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter


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
    # 找到包含值为255的元素的索引

    if focus == "light":
        idx = np.where(array == -1)
    else:
        idx = np.where(array == -1)

    # 删除包含值为255的元素
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

    # idx = np.where(array < 0)

    # array = np.delete(array, idx)

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

        if focus == "light":
            region = np.where(region_lb == (i + 1), region, -1)
        elif focus == "dark":
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


def ProcessHierachy(regions_list, xy_last_list, focus, break_per=0.4, first=False):
    return_list, centers_list, xy_list, d_index, areas_list = [], [], [], [], []
    s, d, lr = 0, 0, 0

    with tqdm(total=len(regions_list), desc="Processing regions") as pbar:
        for i in range(0, len(regions_list)):
            region = regions_list[i]
            bin_image = BinarizeRaster(region, focus)

            num_labels, lb_images, stats, centers = cv2.connectedComponentsWithStats(
                bin_image, connectivity=4
            )
            stats_ori = stats[:, cv2.CC_STAT_AREA]
            num_labels -= 1
            stats = stats[1:, :]
            centers = centers[1:, :]

            areas = stats[:, cv2.CC_STAT_AREA]  # 去除背景
            areas = np.sqrt(areas)
            ht_index = head_tail_breaks(areas, break_per)

            if ht_index > 2 or first == True:
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

    del regions_list, xy_last_list, num_labels, stats, centers, ht_index
    return return_list, d, s, lr, centers_list, xy_list, d_index, lb_images, areas_list


def save_result(csv_path, d_list, s_list, lr_list, i):
    # 创建一个空的 DataFrame，只包含列名
    df = pd.DataFrame(columns=["I", "D", "S", "LR", "H"])
    d = np.array(d_list)
    s = np.array(s_list)
    lr = np.array(lr_list)
    h = lr / s
    arr = np.arange(1, i)
    lr = np.append(lr, np.sum(lr_list))
    s = np.append(s, np.sum(s_list))
    arr = np.append(arr, 0)
    d = np.append(d, d.sum())

    # 将这些数组转换为 pandas Series
    s1 = pd.Series(arr, name="I")
    s2 = pd.Series(d, name="D")
    s3 = pd.Series(s, name="S")
    s6 = pd.Series(lr, name="LR")
    s7 = pd.Series(h, name="H")

    # 将这三列 Series 合并成一个 DataFrame
    df = pd.concat([s1, s2, s3, s6, s7], axis=1)

    # 将 I 字段格式化为整数
    df["I"] = df["I"].astype(int)

    # 将 H 字段格式化为小数点后两位
    formatted_h = df["H"].apply(lambda x: "{:.2f}".format(x))
    df["H"] = formatted_h

    # 将 DataFrame 写入 CSV 文件
    df.to_csv(csv_path, index=False)


# Function to create a directory if it doesn't exist
def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def write_centroids(cen_path, centers_list, d_index_list, area_list):
    folder, _ = os.path.split(cen_path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    driver = ogr.GetDriverByName("ESRI Shapefile")
    out_ds = driver.CreateDataSource(cen_path)
    out_layer = out_ds.CreateLayer(
        os.path.splitext(cen_path)[0], geom_type=ogr.wkbPoint
    )

    # 创建字段定义
    field_def_id = ogr.FieldDefn("id", ogr.OFTInteger)
    field_def_d = ogr.FieldDefn("d", ogr.OFTInteger)
    field_def_a = ogr.FieldDefn("a", ogr.OFTInteger)

    # 将字段定义添加到图层
    out_layer.CreateField(field_def_id)
    out_layer.CreateField(field_def_d)
    out_layer.CreateField(field_def_a)

    for hie in range(len(centers_list)):
        for p in range(len(centers_list[hie])):
            a = area_list[hie][p] * (hie + 1)

            x_geo = centers_list[hie][p, 0]
            y_geo = -centers_list[hie][p, 1]

            feature = ogr.Feature(out_layer.GetLayerDefn())
            feature.SetField("id", hie + 1)
            feature.SetField("a", np.sqrt(a))
            if hie < len(centers_list) - 1:
                if p in d_index_list[hie + 1]:
                    feature.SetField("d", 1)
                else:
                    feature.SetField("d", 0)
            else:
                feature.SetField("d", 0)

            wkt = f"POINT ({x_geo} {y_geo})"
            point = ogr.CreateGeometryFromWkt(wkt)
            feature.SetGeometry(point)
            out_layer.CreateFeature(feature)
    # 释放资源
    out_ds = None


def write_dcentroids(cen_path, centers_list, d_index_list, area_list):
    folder, _ = os.path.split(cen_path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    driver = ogr.GetDriverByName("ESRI Shapefile")
    out_ds = driver.CreateDataSource(cen_path)
    out_layer = out_ds.CreateLayer(
        os.path.splitext(cen_path)[0], geom_type=ogr.wkbPoint
    )

    # 创建字段定义
    field_def_id = ogr.FieldDefn("id", ogr.OFTInteger)
    field_def_d = ogr.FieldDefn("w", ogr.OFTInteger)

    # 将字段定义添加到图层
    out_layer.CreateField(field_def_id)
    out_layer.CreateField(field_def_d)

    for hie in range(len(centers_list) - 1):
        for p in range(len(centers_list[hie])):
            x_geo = centers_list[hie][p, 0]
            y_geo = -centers_list[hie][p, 1]

            if hie < len(centers_list) - 1:
                if p in d_index_list[hie + 1]:
                    # for t in range((hie+1)**2):
                    feature = ogr.Feature(out_layer.GetLayerDefn())
                    a = area_list[hie][p] * (hie + 1)
                    feature.SetField("id", hie + 1)
                    feature.SetField("w", a)

                    wkt = f"POINT ({x_geo} {y_geo})"
                    point = ogr.CreateGeometryFromWkt(wkt)
                    feature.SetGeometry(point)
                    out_layer.CreateFeature(feature)
    # 释放资源
    out_ds = None


def flatten(lst):
    result = []
    for elem in lst:
        if isinstance(elem, list):
            result.extend(flatten(elem))
        else:
            result.append(elem)
    return result


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


def write_subs(output_hie_path, output, inputraster, lb_images, xy_list):
    folder, _ = os.path.split(output_hie_path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    himgs = []
    for i in range(0, len(output)):
        himg = np.zeros(np.shape(inputraster)).astype(np.int8)

        if i == 0:
            himg = np.where(lb_images > 0, 1, 0)
        else:
            lr = output[i]
            # with tqdm(total=len(lr), desc='Processing sub') as pbar:
            for j in range(len(lr)):
                # 获取当前影像和坐标位置
                current_image = np.where(lr[j] > 0, i + 1, 0).astype(np.int8)
                y, x = xy_list[i][j]
                # 将当前影像插入到新影像中
                himg[
                    x : x + current_image.shape[0], y : y + current_image.shape[1]
                ] = current_image
        himgs.append(himg)

    base = np.zeros(np.shape(inputraster)).astype(np.int8)
    for i in range(len(himgs)):
        h = himgs[i]
        if i == 0:
            base += h
        else:
            base[h != 0] = 0
            base = base + h

    cv2.imwrite(output_hie_path, base)
