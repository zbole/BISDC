import os
import numpy as np

test_folder = '../dataset/UrbanBIS/original/stpls3d/test/'  # 请根据实际情况修改路径

test_files = [f for f in os.listdir(test_folder) if f.endswith('.txt')]

output_folder = '../dataset/UrbanBIS/original/stpls3d/test_split/'
os.makedirs(output_folder, exist_ok=True)

area_number = 26

for test_file in test_files:
    file_path = os.path.join(test_folder, test_file)
    data = np.loadtxt(file_path, delimiter=' ')
    middle_index = len(data) // 2
    part1 = data[:middle_index]
    part2 = data[middle_index:]
    part1_file = os.path.join(output_folder, f"Area{area_number}.txt")
    part2_file = os.path.join(output_folder, f"Area{area_number + 1}.txt")
    np.savetxt(part1_file, part1, fmt='%.4f %.4f %.4f %d %d %d %d %d %d', delimiter=' ')
    np.savetxt(part2_file, part2, fmt='%.4f %.4f %.4f %d %d %d %d %d %d', delimiter=' ')
    print(f"File {test_file} split into two parts: {part1_file}, {part2_file}")
    area_number += 2
