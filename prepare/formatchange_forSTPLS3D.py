import os
import numpy as np

# 定义每个类的颜色（RGB）
color_mapping = {
    0: [0, 255, 0],      # Ground
    1: [255, 0, 0],      # Truck
    2: [0, 255, 255],    # LowVegetation
    3: [0, 200, 0],      # MediumVegetation
    4: [0, 128, 0],      # HighVegetation
    5: [255, 255, 0],    # Vehicle
    6: [128, 0, 128],    # Building
    7: [255, 0, 255],    # Aircraft
    8: [128, 128, 0],    # MilitaryVehicle
    9: [0, 0, 255],      # Bike
    10: [255, 165, 0],   # Motorcycle
    11: [255, 105, 180], # LightPole
    12: [0, 255, 127],   # StreetSign
    13: [169, 169, 169], # Clutter
    14: [139, 69, 19],   # Fence
    15: [128, 128, 128], # Road
    17: [135, 206, 250], # Windows
    18: [255, 255, 255], # Dirt
    19: [0, 255, 0]      # Grass
}

# 路径设置
input_folder = '../../data/STPLS3D'  # 你的txt文件夹路径
output_folder = './stpls3d_remap'  # 输出的路径
os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹

# 读取每个文件并进行转换
for file_name in os.listdir(input_folder):
    if file_name.endswith('.txt'):
        file_path = os.path.join(input_folder, file_name)
        data = np.loadtxt(file_path, delimiter=',')  # 读取txt文件，格式为逗号分隔

        # 假设数据结构是每行：[x, y, z, r, g, b, semantic_label, instance_label]
        coords = data[:, :3]  # 获取坐标 [x, y, z]
        colors = data[:, 3:6].astype(int)  # 获取颜色 [r, g, b]
        semantic_labels = data[:, 6]  # 语义标签
        instance_labels = data[:, 7].astype(int)  # 实例标签

        # 强制将 semantic_labels 转换为整数类型
        semantic_labels = semantic_labels.astype(int)
        instance_labels = instance_labels.astype(int)

        # 对于非建筑物的实例标签，设置为 -100
        instance_labels[semantic_labels != 1] = -100

        # 互换语义标签 1 和 6
        semantic_labels[semantic_labels == 1] = -1
        semantic_labels[semantic_labels == 6] = 1
        semantic_labels[semantic_labels == -1] = 6

        # 细粒度建筑分类设置为 1
        fine_grained_building_category = np.where(semantic_labels == 6, 1, -100)

        # 处理每一行数据并保存为新的格式
        output_data = np.column_stack((coords, colors, semantic_labels, instance_labels, fine_grained_building_category))

        # 生成输出文件路径
        output_file_path = os.path.join(output_folder, f"{file_name}")

        # 保存到新的txt文件
        # 修改 fmt 部分：前 3 列为浮点数格式，后 6 列为整数格式
        np.savetxt(output_file_path, output_data, fmt='%.4f %.4f %.4f %d %d %d %d %d %d', delimiter=' ')

        print(f"Converted {file_name} and saved to {output_file_path}")
