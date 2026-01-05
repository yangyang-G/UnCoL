import os

filepath = '/amax/data/luwenjing/P1_UPCoL/Datasets/TBAD128'
patient_files = []
for file in os.listdir(filepath):
    if "Patient" in file:
        patient_files.append(file[:-3])

output_file_path = '/amax/data/luwenjing/P4_UKD-SAM/UKDSAM/codes/datasets/patient_filenames.txt'

# 将符合条件的文件名写入文本文件
with open(output_file_path, 'w') as file:
    for name in patient_files:
        file.write(f"{name}\n")

print(f"Patient filenames have been saved to {output_file_path}")
