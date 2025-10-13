import os
import shutil

# 设置包含 .txt 文件的文件夹路径
folder_path = '../prepare/whu3d_remap/'  # 请根据实际情况修改路径

# 获取文件夹中所有的 .txt 文件
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# 按文件名排序（如果需要）
txt_files.sort()

# 创建训练集、验证集和测试集文件夹（如果不存在）
train_folder = '../dataset/UrbanBIS/original/whu3d/train/'
val_folder = '../dataset/UrbanBIS/original/whu3d/val/'
test_folder = '../dataset/UrbanBIS/original/whu3d/test/'

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# 文件总数
total_files = len(txt_files)

# 计算每个集的文件数量
train_count = int(total_files * 0.7)  # 训练集比例：70%
val_count = int(total_files * 0.15)  # 验证集比例：15%
test_count = total_files - train_count - val_count  # 测试集比例：15%

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

    # 生成新文件名
    new_name = f'Area{i}.txt'
    destination_path = os.path.join(destination_folder, new_name)

    # 拷贝文件到目标文件夹
    shutil.copy2(old_path, destination_path)
    print(f'Copied {txt_file} to {destination_path}')

    # 在目标文件夹内对文件进行重命名
    # 文件已经被拷贝，所以重命名的操作是在目标文件夹中
    os.rename(destination_path, os.path.join(destination_folder, new_name))
    print(f'Renamed {txt_file} to {new_name}')
