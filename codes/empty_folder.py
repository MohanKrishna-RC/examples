import os

out_path = "Untitled_Folder/"

# for j in range(100):
#     if not os.path.exists(out_path + str(j)):
#         os.makedirs(out_path + str(j))

any_list_of_names = []

for i in range(10):
    any_list_of_names.append("''")

for j in any_list_of_names:
        if not os.path.exists(out_path + str(j)):
            os.makedirs(out_path + str(j))