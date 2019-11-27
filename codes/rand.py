import os
import random

path ='/home/mohan/Documents/BARC/TrainingData/train_all_indian/male_youngadult'
files = os.listdir(path)
index = random.randrange(0, len(files))
dst = 
print(index)
for i in range(index):
    if not os.path.exists(dst + '45' + str(i)):
        os.makedirs(dst + str(i))
    shutil.copy(file[index],dst + str(i))


import os 
  
# Function to rename multiple files 
def main(): 
    i = 0
      
    for filename in os.listdir("/home/mohan/Documents/BARC/TrainingData/train_all_indian/male_younadult"): 
        dst ="Hostel" + str(i) + ".jpg"
        src ='xyz'+ filename 
        dst ='xyz'+ dst 
          
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
        i += 1
  
# Driver Code 
if __name__ == '__main__': 

for file in file_list:
    file = file.replace(' ','')
import glob
import os
file_list = glob.glob('/home/mohan/Documents/BARC/TrainingData/train_all_indian/female_senior/*')
for file in file_list:
    # print(file)
    try:
        # src ='/home/mohan/Documents/BARC/TrainingData/train_all_indian/female_senior'+ file 
        file = file.replace(' ','')
        print(file)
        dst = file
        dst =  dst
        os.rename(file,dst)
    except Exception as e:
        print("filename", file)



src ='/home/mohan/Documents/BARC/TrainingData/train_all_indian/female_senior/*'+ file
file = file.replace(' ','')
print(file)
dst = file
dst =  dst
os.rename(file,dst)
# except Exception as e:
#     print("filename", file)


def main(): 
    i = 0
      
    for filename in os.listdir("/home/mohan/Documents/BARC/TrainingData/train_all_indian/male_younadult/"): 
        print(filename)
        # dst ="Hostel" + str(i) + ".jpg"
        # src ='xyz'+ filename 
        # dst ='xyz'+ dst 
          
        # # rename() function will 
        # # rename all the files 
        # os.rename(src, dst) 
        i += 1
  
# Driver Code 
if __name__ == '__main__':
    main()


import os
for dirname in os.listdir("/home/mohan/Documents/BARC/TrainingData/train_all_indian"):
    print(dirname)
    if os.path.isdir(dirname):
        for i, filename in enumerate(os.listdir(dirname)):
            os.rename(dirname + "/" + filename, dirname + "/" + str(i) + ".bmp")

import os
import glob
path =  os.getcwd()
# filenames = os.listdir('/home/mohan/Documents/BARC/TrainingData/train_all_indian/male_youngadult')
filenames = glob.glob('/home/aodev/alok/insightface/deploy/TrainingData/Folder_need/*/*')


# print(filenames)
for filename in filenames:
    # print(filename)
    os.rename(filename, filename.replace(" ", "-"))