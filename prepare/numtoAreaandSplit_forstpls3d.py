import os
import shutil

# 设置包含 .txt 文件的文件夹路径
folder_path = '../prepare/stpls3d_remap/'  # 请根据实际情况修改路径

# 获取文件夹中所有的 .txt 文件
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# 按文件名排序（如果需要）
txt_files.sort()

# 创建训练集、验证集和测试集文件夹（如果不存在）
train_folder = '../dataset/UrbanBIS/original/stpls3d/train/'
val_folder = '../dataset/UrbanBIS/original/stpls3d/val/'
test_folder = '../dataset/UrbanBIS/original/stpls3d/test/'

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# 文件总数
total_files = len(txt_files)

# 计算每个集的文件数量
train_count = int(total_files * 0.6)  # 训练集比例：60%
val_count = int(total_files * 0.2)  # 验证集比例：20%
test_count = total_files - train_count - val_count  # 测试集比例：20%

# 全局计数器，用于命名拆分后的文件
counter = 1

# 拷贝并重命名文件，然后分配到相应文件夹
for i, txt_file in enumerate(txt_files, start=1):
    old_path = os.path.join(folder_path, txt_file)

    # 根据比例选择目标文件夹
    if i <= train_count:
        destination_folder = train_folder
    elif i <= train_count + val_count:
        destination_folder = val_folder
    else:
        destination_folder = test_folder

    # 打开原始文件
    with open(old_path, 'r') as file:
        lines = file.readlines()

    # 计算每一份的大小（每个文件分为3部分）
    total_lines = len(lines)
    part_size = total_lines // 3

    # 拆分并保存文件
    for j in range(3):
        # 每部分的数据
        start_idx = j * part_size
        end_idx = start_idx + part_size if j < 2 else total_lines  # 确保最后一部分包括所有剩余的行
        part_lines = lines[start_idx:end_idx]

        # 生成新文件名：Area + 数字
        new_name = f'Area{counter}.txt'  # 使用 Area + 数字 格式，counter 是全局计数器
        destination_path = os.path.join(destination_folder, new_name)

        # 将拆分后的数据写入新的文件
        with open(destination_path, 'w') as new_file:
            new_file.writelines(part_lines)

        print(f'Created {new_name} in {destination_folder}')

        # 更新计数器
        counter += 1

print("All files have been split and saved.")
