import os
import numpy as np
from pywhu3d.tool import WHU3D

# 设置数据集根目录
data_root = '../../data/'  # 根目录不需要修改，因为库会自动访问
h5_folder = '../../data/als/h5/'  # h5 文件所在文件夹

# 输出文件夹
output_folder = './whu3d_remap/'  # 您可以修改为您希望保存文件的文件夹路径
os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹，如果已存在则忽略

# 修改后的label_mapping，确保与模型的语义标签一致,原本5是Boat
label_mapping = {
    200000: 3,  # bridge -> Bridge
    200101: 6,  # building -> Building
    200200: 2,  # water -> Water
    200301: 8,  # tree -> Tree -> 8
    200400: 1,  # veg -> Vegetation
    200500: 7,  # low veg -> 7
    200601: 9,  # light -> 9
    200700: 10,  # electric-> 10
    200800: 0,  # ground -> Terrain
    200900: 11,  # others-> 11
    100500: 4,  # vehicle -> Vehicle
    100600: 12   # non vehicle -> 12
}

color_mapping = {
    3: [255, 0, 0],    # Bridge - 红色
    6: [0, 255, 0],    # Building - 绿色
    2: [0, 0, 255],    # Water - 蓝色
    8: [255, 255, 0],  # Tree - 黄色
    1: [0, 255, 255],  # Vegetation - 青色
    7: [0, 128, 0],    # Low Vegetation - 深绿色
    9: [128, 128, 128], # Light - 灰色
    10: [255, 165, 0],  # Electric - 橙色
    0: [128, 0, 0],    # Ground - 深红色
    11: [255, 192, 203], # Others - 粉色
    4: [255, 0, 255],   # Vehicle - 品红色
    12: [128, 128, 0]   # Non-Vehicle - 橄榄色
}

# 使用 os 读取 als 文件夹下所有的 h5 文件
h5_files = [os.path.splitext(f)[0] for f in os.listdir(h5_folder) if f.endswith('.h5')]  # 提取文件名前缀，去掉 .h5
print(h5_files)  # 打印提取后的文件名（不带扩展名）

from tqdm import tqdm

for h5_file in tqdm(h5_files, desc="Processing H5 files"):

    whu3d = WHU3D(data_root=data_root, data_type='als', format='h5', scenes=[h5_file])
    h5_file_path = os.path.join(h5_folder, f'{h5_file}.h5')  # 获取文件的完整路径
    # 获取特定场景的数据
    coords = whu3d.data[h5_file]['coords']
    instances = whu3d.labels[h5_file]['instances']
    semantics = whu3d.labels[h5_file]['semantics']
    # 语义标签映射
    mapped_semantics = np.vectorize(label_mapping.get)(semantics)
    # 重新排序实例编号
    new_instances = np.full_like(instances, -100)
    # 1. 标记所有非6的实例为-100，计算-100的比例
    non_building_mask = mapped_semantics != 6
    # non_building_count = np.sum(non_building_mask)
    # total_points = len(instances)
    # non_building_ratio = non_building_count / total_points
    # print(f"Non-building ratio: {non_building_ratio:.4f}")

    # 对于非建筑物的实例，设置为 -100
    new_instances[non_building_mask] = -100

    # 2. 对于建筑物（语义标签为6）的实例，重新编号
    building_mask = mapped_semantics == 6
    unique_instances = np.unique(instances[building_mask])
    new_instance_id = 1

    # Debug: Check unique instances
    # print(f"Unique instances with building type (6): {unique_instances}")

    for instance_id in unique_instances:
        instance_mask = instances == instance_id
        # 只有当实例语义为6时，才会重新排序
        new_instances[instance_mask] = new_instance_id
        new_instance_id += 1

    # Debug: Check if instance numbering is correct
    # print(f"New instance numbers after renumbering: {np.unique(new_instances)}")

    # 修改第九列（即所有建筑物实例的标记），将建筑物标记为1
    additional_column = np.full_like(instances, -100)
    additional_column[building_mask] = 1  # 将所有建筑物标记为1

    # 获取颜色值
    colors = np.array([color_mapping.get(label, [0, 0, 0]) for label in mapped_semantics])

    # 合并坐标、颜色、语义标签、实例编号和第九列标记为1（建筑物）
    point_cloud_data = np.column_stack((coords, colors, mapped_semantics, new_instances, additional_column))

    # 生成输出文件路径
    output_file_path = os.path.join(output_folder, f'{h5_file}.txt')

    # 保存为 txt 文件
    np.savetxt(output_file_path, point_cloud_data, fmt='%.4f %.4f %.4f %d %d %d %d %d %d', delimiter=' ')
    # print(f'Saved {h5_file} to {output_file_path}')
