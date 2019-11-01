import os
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
image_dir = Path("/home/mohan/Age_gender/Datasets/FG-Net/FGNET/")
out_path = "/home/mohan/Age_gender/Datasets/FG-Net/FGNET/Age_f"
age_folders = [(0,5),(6,15),(16,25),(26,40),(41,50),(51,70),(71,100)]
age_fold = ['00','01','02','03','04','05','06','07','08','09']
for image_path in image_dir.glob("**/*.JPG"):
    image_name = image_path.name  # [age]_[gender]_[race]_[date&time].jpg
    age = image_name[4:6]
    # print(image_name)
    # ages.append(min(int(age), 100))
    # print(age)
    for i in range(100):
        # print(i)
        # print(age)
        # age = int(age)
        # print(type(age))
        # for j in age_fold:
        #     if age == age_fold[j]:
        #         age = age_fold[j][1]
        age = int(age)
        if i == age:
            print("insuide",age)
            if not os.path.exists(out_path + str(i)):
                os.makedirs(out_path + str(i))
            try:
                shutil.copy(image_path,out_path + str(i))
            except Exception as e:
                print("filename",image_path)

# for r, d, f in os.walk(path):
#     for folder in d:
#         folders.append(os.path.join(r, folder))

# a = [4,2,3,1,5,6]
# if 6 in a:
#     a = a.index(6)
#     print(a)