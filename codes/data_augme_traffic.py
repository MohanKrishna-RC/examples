
# images32 = np.array(images)
# for image in images32:
#     print(type(image))
images = cv2.imread(train_data_directory)
(H, W) = image.shape[:2]

for image in images:
    res = cv2.resize(image,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    res.imwrite('')


if not os.path.exists('small'): #Does the folder "small" exists ?
   os.makedirs('small') #If not, create it
pic_num=1   #Initialize var pic_num with value 1

for i in ['']: #For every element of this list (containing 'test' only atm)
    try: #Try to
        if os.path.exists(str(pic_num)+'.jpg'): #If the image pic_num (= 1 at first) exists
            print(i) #Prints 'test' (because it's only element of the list)
            #Initialize var img with image content (opened with lib cv2)
            img=cv2.imread(str(pic_num)+'.jpg') 
            #We resize the image to dimension 100x100 and store result in var resized_image
            resized_image=cv2.resize(img,(100,100)) 
            #Save the result on disk in the "small" folder
            cv2.imwrite("small/"+str(pic_num)+'.jpg',resized_image)
        pic_num+=1 #Increment variable pic_num
    except Exception as e: #If there was a problem during the operation
        print(str(e)) #Prints the exception

for file_name in os.listdir(train_data_directory):
    print("Processing %s" % file_name)
    # images = Image.open(os.path.join(file_name,))
    file_names = [os.path.join(file_name, f) 
                      for f in os.listdir(file_name) 
                      if f.endswith(".ppm")]
    print(file_names)
    for f in file_names:
        print(f)
        image = cv2.imread(f)
    # print(image)
    # x,y = image.size
    # new_dimensions = (x/2, y/2)
        output = cv2.resize(image,(100,100))
    
        output_file_name = os.path.join('/home/mohan/small', "small_" + f)
        cv2.imwrite(output_file_name, "JPEG", quality = 95)

print("All done")
def load(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                    if os.path.isdir(os.path.join(data_directory, d))]
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                        for f in os.listdir(label_directory) 
                        if f.endswith(".ppm")]
    images = []
    for f in file_names:
        print(f)
        # f = str(f)
        images.append(cv2.imread(f))
        print("ff")
    for image in images:
        print(image)
        output = cv2.resize(image,(100,100))
        print("df")
        print(output)
        # output_file_name = os.path.join('/home/mohan/small', "small_" + str(output) + '.ppm') 
        cv2.imwrite("/home/mohan/small"+str(f)+'.ppm',output)

def load(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                    if os.path.isdir(os.path.join(data_directory, d))]
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                        for f in os.listdir(label_directory) 
                        if f.endswith(".ppm")]
    for f in file_names:
        # print(f)
        # f = str(f)
        image = cv2.imread(f)
        # print("ff")
        # print(image)
        output = cv2.resize(image,(100,100))
        # print("df")
        # print(output)
        # output_file_name = os.path.join('/home/mohan/small', "small_" + str(output) + '.ppm') 
        cv2.imwrite("/home/mohan/small/" + str(f).split('/')[-1],output)
        print("/home/mohan/small/" + str(f).split('/')[-1])

        if os.path.exists("/home/mohan/small/" + str(f).split('/')[-1]):
            print('phat gaya')

load(train_data_directory)

PATH = os.getcwd()
# Define data path
# data_path = PATH + '/data'
data_dir_list = os.listdir(train_data_directory)

img_data_list=[]

for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )

        input_img_resize=cv2.resize(input_img,(128,128))
        img_data_list.append(input_img_resize)