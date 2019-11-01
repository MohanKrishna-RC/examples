import os
out_path = "/home/mohan/Documents/Untitled Folder/"
# for j in range(100):
#     if not os.path.exists(out_path + str(j)):
#         os.makedirs(out_path + str(j))


age_folders = ['child','male_teen','male_youth','male_youngadult','male_adult','male_senior',
      'female_teen','female_youth','female_youngadult','female_adult','female_senior']

for j in age_folders:
    if not os.path.exists(out_path + str(j)):
        os.makedirs(out_path + str(j))