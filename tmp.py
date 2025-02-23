import os
import csv

folder_path1 = "./data/validation/images"
folder_path2 = "./data/validation/segmentations"

# 使用 os.scandir 获取文件列表，并拼接完整路径
with os.scandir(folder_path1) as entries:
    all_files1 = [os.path.join(folder_path1, entry.name) for entry in entries if entry.is_file()]
with os.scandir(folder_path2) as entries:
    all_files2 = [os.path.join(folder_path2, entry.name) for entry in entries if entry.is_file()]

# 确保两个列表长度一致，如果不一致可能需要根据实际情况处理
if len(all_files1) != len(all_files2):
    print("两个文件夹中的文件数量不匹配，可能无法正确一一对应。")
else:
    # 生成要写入 CSV 的数据
    rows = []
    for file1, file2 in zip(all_files1, all_files2):
        rows.append([file1, file2])

    # 写入 CSV 文件
    csv_file_path = 'validation.csv'
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 写入每一行数据
        for row in rows:
            writer.writerow(row)
    print(f"已将匹配结果存入 {csv_file_path} 文件。")