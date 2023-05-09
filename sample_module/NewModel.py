# Имеем всю матрицу высот. Принимаем номер узла в матрице и радиус Аккуратова.
# Возвращаем матрицу вероятностей в границах радиуса Аккуратова и набор точек границы полигона

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from django.contrib.gis.gdal import GDALRaster
import math

SEC_PER_DEGREE = 60 * 60

def convert(distance_km, long):
    # радиус Земли
    R = 6371
    km_per_lat = 2*math.pi*R / 360
    km_per_long = 2*math.pi*math.cos(math.radians(abs(long)))*R / 360
    return distance_km / km_per_lat, distance_km / km_per_long

def coord_from_geo(lat, lng, rst):
    top_left = (rst.extent[3], rst.extent[0])
    array_dim = (rst.bands[0].width, rst.bands[0].height)
    sec_per_width = (rst.extent[2] - rst.extent[0]) * SEC_PER_DEGREE / array_dim[0]
    sec_per_height = (rst.extent[3] - rst.extent[1]) * SEC_PER_DEGREE / array_dim[1]
    sec_rel_coords = (top_left[0] - lat) * SEC_PER_DEGREE, (lng - top_left[1]) * SEC_PER_DEGREE
    return (int(round(sec_rel_coords[1] / sec_per_width, 0)),
            int(round(sec_rel_coords[0] / sec_per_height, 0)))


def geo_from_coords(x, y, rst):
    gt = rst.geotransform
    x_geo = gt[0] + y*gt[1] + x*gt[2]
    y_geo = gt[3] + y*gt[4] + x*gt[5]
    return y_geo, x_geo

# Расстояние в метрах между узлами в сетке высот
GRID_STEP = 30


class QPoint:
    def __init__(self, cur_point, to_go):
        self.cur_point = cur_point
        self.x, self.y = cur_point
        self.to_go = to_go


def get_heights(file_name: str) -> np.array:
    """
    Читаем сетку высот из csv файла file_name.
    Результат возвращается в виде numpy array
    """
    rst = GDALRaster(file_name, write=False)
    return rst


# def get_point(lat, long) -> tuple:
#     """
#     Интерфейс-заглушка для получения стартовой точки (индексы)
#     """
#     return 250, 200


def get_radius(h, w) -> float:
    L = 0.73 * h * (math.log10(w) + 1)
    return L


def main_model(params):
    rst = GDALRaster('height_map.tif', write=False)
    # heights = get_heights('height_map.tif').to_numpy()
    center = params['point_start']
    # Полная карта Хибин
    # plot_map(heights)

    # Получение входных параметров
    real_start_point = coord_from_geo(center[1], center[0], rst)
    max_radius = get_radius(params['h'], rst.bands[0].data(offset=real_start_point, size=(1, 1))[0])

    radius_lat, radius_long = convert(max_radius/1000, center[1])
    top_left = center[0] - radius_long, center[1] - radius_lat
    top_left_index = coord_from_geo(top_left[1], top_left[0], rst)
    # Подготовка вероятностей
    result_size = (real_start_point[0] - top_left_index [0] + 1) * 2
    # Для вероятностей создаётся квадратная матрица размером
    # с описанный квадрат. Стартовая точка в середине
    result = np.zeros((2 * result_size + 1, 2 * result_size + 1))
    result[result_size, result_size] = 1.
    start_x = start_y = len(result) // 2
    start_point = start_x, start_y

    local_heights = rst.bands[0].data(offset=top_left_index, size=(result_size, result_size))[0]

    queue = [QPoint(start_point, max_radius)]

    while queue:
        queue, result = update_queue(queue, local_heights, result)

    polygon_coord = []
    for coord in result:
        new_coord = coord[0] + top_left_index[0], coord[1] + top_left_index[1]
        polygon_coord.append(geo_from_coords(new_coord[0], new_coord[1], rst))
    # Распределение вероятностей рядом с картой
    # plot_heights_and_possibilities(local_heights, result)

    # plot_possibilities(result)
    return polygon_coord


def update_queue(queue, heights, possibilities):
    """
    Принимаем очередь, находим новые направления, добавляем их к очереди
    """
    point = queue[0]

    ways = ((0, -1),
            (-1, 0),
            (1, 0),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1))
    delta = 0

    # Собираем информацию о точках вокруг
    # TODO сделать этот обход без цикла на базе numpy
    for way in ways:
        new_x = way[0] + point.cur_point[0]
        new_y = way[1] + point.cur_point[1]
        way_height = heights[new_x, new_y]
        delta_height = heights[point.cur_point] - way_height
        delta_height = 0 if delta_height < 0 else delta_height
        delta += delta_height

    for way in ways:
        new_x = way[0] + point.cur_point[0]
        new_y = way[1] + point.cur_point[1]
        way_height = heights[new_x, new_y]
        delta_height = heights[point.cur_point] - way_height
        # Рассматриваем только точки ниже
        delta_height = 0 if delta_height < 0 else delta_height
        if delta != 0 and delta_height != 0:
            # Накапливаем вероятности исходя из перепадов высот
            possibilities[new_x, new_y] += np.round(delta_height / delta * possibilities[point.cur_point], 2)
            # Вероятность закреплена на 1. Возможны точки, где накапливается большая вероятность
            if possibilities[new_x, new_y] > 1:
                possibilities[new_x, new_y] = 1
            # Реальный пройденный путь по диагонали параллелепипеда
            real_length = np.sqrt(abs(way[0]) * GRID_STEP ** 2 +
                                  abs(way[1]) * GRID_STEP ** 2 +
                                  delta ** 2)
            # Остаточный путь после текущего отрезка
            to_go = point.to_go - real_length
            # Очередь пополняется, если есть путь для прохождения,
            # вероятность выше 5 процентов
            if to_go > 0 and possibilities[new_x, new_y] > 0.05:
                queue.append(QPoint((new_x, new_y), to_go))
    queue.pop(0)
    # Пошаговая печать
    # plot_possibilities(possibilities)
    return queue, possibilities


def create_polygon(possibility):
    points = []
    for i, line in enumerate(possibility):
        j = 0
        flag = False
        while j < len(line) and line[j] == 0:
            j += 1
        else:
            if j != len(line):
                points.insert(0, (i, j))
                flag = True
        while j < len(line) and line[j] > 0:
            j += 1
        else:
            if flag:
                points.append((i, j))
                flag = False
    points.append(points[0])
    return points


def plot_heights_and_possibilities(heights, results):
    x = y = len(results) // 2
    fig, axs = plt.subplots(1, 2)

    axs[0].axis('off')
    axs[1].axis('off')

    im1 = axs[0].imshow(heights, cmap='gist_earth')
    im2 = axs[1].imshow(results, cmap='gist_gray')

    axs[0].plot(x, y, 'bo')
    axs[1].plot(x, y, 'bo')

    plt.colorbar(im1, ax=axs[0])
    plt.colorbar(im2, ax=axs[1])

    plt.show()


def plot_possibilities(results):
    x = y = len(results) // 2
    plt.axis('off')
    plt.imshow(results, cmap='gist_gray')
    plt.plot(x, y, 'bo')
    plt.colorbar()
    plt.show()


def plot_map(heights, mode=False):
    plt.axis('off')
    plt.imshow(heights, cmap='gist_earth')
    if mode == 'local':
        x = y = len(heights) // 2
        plt.plot(x, y, 'bo')
    plt.colorbar()
    plt.show()


result = main_model()
poly_result = create_polygon(result)

x = [i[1] for i in poly_result]
y = [i[0] for i in poly_result]

plt.plot(x, y, 'ro', linestyle='--')
plt.imshow(result, cmap='gist_gray')
plt.show()
