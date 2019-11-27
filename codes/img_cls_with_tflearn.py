# Residual blocks
# 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
n = 5
batch_size = 1000
num_classes = 10
epochs_shortrun = 5
epochs_longrun = 200
save_dir = "./files"
res_dir = "./results"
model_name = 'resnet_cifar10'

import os

save_fn = model_name + ".tfsave"
save_file = os.path.join(save_dir, save_fn)

if not os.path.isdir(res_dir):
  os.makedirs(res_dir)

ckpt_dir = os.path.join(save_dir,"checkpoints")

if not os.path.isdir(ckpt_dir):
    os.makedirs(ckpt_dir)

tblog_dir = os.path.join(save_dir,"tflogs")
if not os.path.isdir(tblog_dir):
    os.makedirs(tblog_dir)
event_dir = os.path.join(tblog_dir,model_name)

#from __future__ import division, print_function, absolute_import
import numpy as np
import dill as pickle
from math import *

def setup_tf():
  # set random seeds for reproducibility
  from tflearn import init_graph
  import tensorflow as tf
  tf.reset_default_graph()
  init_graph(seed=343)
  tf.set_random_seed(343)
  np.random.seed(343)

def setup_cifar_data(verbose=False):
    from tflearn.datasets import cifar10
    from tflearn.data_utils import to_categorical
    datadir = "/cifar/"
    datafile = datadir+"cifar-10-python.tar.gz"
    if not os.path.isfile(datafile):
        import shutil,tarfile
        os.makedirs(datadir, exist_ok=True)
        shutil.copyfile("/home/mohan/Downloads/cifar-10-python.tar.gz", datafile)  #copy from one source to another
        with tarfile.open(datafile, "r:gz") as f:
            f.extractall(datadir)
    (X, Y),(testX, testY) = cifar10.load_data(dirname="/cifar")
    Y = to_categorical(Y, num_classes)
    testY = to_categorical(testY, num_classes)
    return (X,Y,testX,testY)

def setup_cifar_labels():
    with open("/cifar/cifar-10-batches-py/batches.meta", 'rb') as fo:
        labels = pickle.load(fo) 
    return labels

# Function to find latest checkpoint file
def last_ckpt(dir):
  fl = os.listdir(dir)
  fl = [x for x in fl if x.endswith(".index")]
  cf = ""
  if len(fl) > 0:
    steps = [float(x.split("-")[1][0:-6]) for x in fl]
    m = max(steps)
    cf = fl[steps.index(m)]
    cf = os.path.join(dir,cf)
  return(cf)

def load_model_from_file(model,file):
    # load data from tarfile to model
    import tarfile
    with tarfile.open(file, "r:bz2") as tar:
        try:
            tar.getmember(save_fn+".index")
            tar.getmember(save_fn+".meta")
            tar.getmember(save_fn+".data-00000-of-00001")
        except KeyError:
            print("Minimum training results files not found!\n")
            tar.extractall(path=save_dir)
            print("Loading {}...".format(save_file))
            model.load(save_file, weights_only=False)
  
def cifar_grid(Xset,Yset,inds,n_col, predictions=None):
    #Visualizing CIFAR 10, takes indicides and shows in a grid
    import matplotlib.pyplot as plt
    if predictions is not None:
        if Yset.shape != predictions.shape:
            print("Predictions must equal Yset in length!")
            return(None)
    N = len(inds)
    n_row = int(ceil(1.0*N/n_col))
    fig, axes = plt.subplots(n_row,n_col,figsize=(10,10))
    clabels = labels["label_names"]
    for j in range(n_row):
        for k in range(n_col):
            i_inds = j*n_col+k
            i_data = inds[i_inds]
            axes[j][k].set_axis_off()
            if i_inds < N:
                axes[j][k].imshow(Xset[i_data,...], interpolation='nearest')
                label = clabels[np.argmax(Yset[i_data,...])]
                axes[j][k].set_title(label)
                if predictions is not None:
                    pred = clabels[np.argmax(predictions[i_data,...])]
                    if label != pred:
                        label += " n"
                        axes[j][k].set_title(pred, color='red')            
    fig.set_tight_layout(True)
    return fig

import os
import tensorflow as tf
# makes Tensorflow shush about SSE and such
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.device('/cpu:0')

setup_tf()

x_train, y_train, x_test, y_test = setup_cifar_data(verbose=True)
labels = setup_cifar_labels()

indices = [np.random.choice(range(len(x_train))) for i in range(36)]

cifar_grid(x_train,y_train,indices,6)



from tflearn import ImagePreprocessing, ImageAugmentation
from tflearn import input_data, DNN
from tflearn import conv_2d, residual_block
from tflearn import batch_normalization, activation, global_avg_pool 
from tflearn import fully_connected, Momentum, regression
from tflearn.callbacks import Callback

print("Using real-time data augmentation.\n")
# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True, 
  mean=[ 0.49139968, 0.48215841, 0.44653091 ])

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_crop([32, 32], padding=4)

# Building Residual Network
net = input_data(shape=[None, 32, 32, 3],
                 data_preprocessing=img_prep,
                 data_augmentation=img_aug)
net = conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = residual_block(net, n, 16)
net = residual_block(net, 1, 32, downsample=True)
net = residual_block(net, n-1, 32)
net = residual_block(net, 1, 64, downsample=True)
net = residual_block(net, n-1, 64)
net = batch_normalization(net)
net = activation(net, 'relu')
net = global_avg_pool(net)

# Regression
net = fully_connected(net, num_classes, activation='softmax')
mom = Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net = regression(net, optimizer=mom, loss='categorical_crossentropy')

# Initialize model
ckpt_file = os.path.join(ckpt_dir,"model.ckpt")
model = DNN(net, checkpoint_path=ckpt_file,
                    max_checkpoints=10, clip_gradients=0.,
                    tensorboard_dir=tblog_dir,tensorboard_verbose=0)

# disabled until directories can be written to /results
#cff = last_ckpt(ckpt_dir)
#if cff != "":
#  print("Loading ",cff,"...")
#  model.load(cff)

# define the early-stop callback

class EarlyStoppingCallback(Callback):
    def __init__(self, val_loss_thresh, val_loss_patience):
        """ minimum loss improvement setup """
        self.val_loss_thresh = val_loss_thresh
        self.val_loss_last = float('inf')
        self.val_loss_patience = val_loss_patience
        self.val_loss_squint = 0
    
    def on_batch_end(self, training_state, snapshot=False):
        """ loss improvement threshold w/ patience """
        # Apparently this can happen.
        if training_state.val_loss is None:
             return
        
        if (self.val_loss_last - training_state.val_loss) < self.val_loss_thresh:
          # unacceptable!
          if self.val_loss_squint >= self.val_loss_patience:
            raise StopIteration
          else:
            self.val_loss_squint += 1
        else:
          # we good again - reset
          self.val_loss_last = training_state.val_loss
          self.val_loss_squint = 0

# Initializae our callback.
early_stopping_cb = EarlyStoppingCallback(val_loss_thresh=0.001,val_loss_patience=25)

sess = tf.Session()
tflearn.is_training(False, session=sess)

load_model_from_file(model, $$ref{{["~:output","6b8fff21-5f3b-4936-89f1-2c2868f631d3","resnet_cifar10.tar.bz2"]}})

print("Starting to train...")

# checkpoints disabled until directories can be written to /results
try:
  model.fit(x_train, y_train, n_epoch=epochs_longrun, 
            validation_set=(x_test, y_test),
     snapshot_epoch=True, snapshot_step=None,
     show_metric=True, batch_size=batch_size, shuffle=True,
     run_id='resnet_cifar10',
     callbacks=early_stopping_cb)
except StopIteration:
  print("Got bored, stopping early.")

print("Training complete.")

model.save(save_file)

# copy events file to /results for history plotting
evfiles = list(filter(os.path.isfile, glob.glob(os.path.join(event_dir, 
                                           "events.out.tfevents.*"))))
evfiles.sort(key=lambda x: os.path.getmtime(x))
shutil.copyfile(os.path.join(event_dir,evfiles[-1]),
                os.path.join(res_dir,model_name+".tfevents"))

# can only save single files to /results, so let's tar the saves
tar_file = os.path.join(res_dir,model_name)+".tar.bz2"
with tarfile.open(tar_file, "w:bz2") as tar:
  for name in [x for x in os.listdir(save_dir) 
               if x.startswith(save_fn)]:
    tar.add(os.path.join(save_dir, name), arcname=name)

acc = model.evaluate(x_test, y_test)

# While we've got it set up, run predictions 
# for the test batch and save to file
x_test_copy = np.copy(x_test) # copy because predict() modifies input (bug?)
y_pred = model.predict(x_test_copy)
with open("/results/test_predictions.dat","wb") as f:
  pickle.dump(y_pred,f)

print("Average accuracy: {:.2f}%.".format(acc[0]*100))

with open($$ref{{["~:output","d9a327cc-5038-429d-8c8f-c197c05e7e41","test_predictions.dat"]}},"rb") as f:
  y_pred = pickle.load(f)

indices = [np.random.choice(range(len(x_test))) for i in range(36)]
labels = setup_cifar_labels()

cifar_grid(x_test,y_test,indices,6, predictions=y_pred)


#Insted of tensorboard we can pull the training history out of the files and plot it ourselves.
from tensorboard.backend.event_processing import event_accumulator
import shutil

# need to copy out to get .tfevents extension, because...raisins
shutil.copy($$ref{{["~:output","6b8fff21-5f3b-4936-89f1-2c2868f631d3","resnet_cifar10.tfevents"]}},"/tmp/hist.tfevents")

ea = event_accumulator.EventAccumulator("/tmp/hist.tfevents",
  size_guidance={ # see below regarding this argument
  event_accumulator.SCALARS: 0
})

ea.Reload() # loads events from file

# fiddly stuff to inspect tags/scalar data entries
#print(ea.Tags())
#print([x for x in ea.Tags()['scalars'] if not x.startswith("Momentum")])

# pull out four metrics, plot
hist = {
  'Accuracy' : [x.value for x in ea.Scalars('Accuracy')],
  'Validation Accuracy' : [x.value for x in 
                           ea.Scalars('Accuracy/Validation')],
  'Loss' : [x.value for x in ea.Scalars('Loss')],
  'Validation Loss' : [x.value for x in ea.Scalars('Loss/Validation')]
}

import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
keys = ['Accuracy', 'Loss', 'Validation Accuracy', 'Validation Loss']
for i,thing in enumerate(keys):
  trace = hist[thing]
  plt.subplot(2,2,i+1)
  plt.plot(range(len(trace)),trace)
  plt.title(thing)

fig.set_tight_layout(True)
fig

#After training the model we can test it with new images.

labels = setup_cifar_labels()

img = tf.read_file($$ref{{["~:output","18657582-57db-4a97-a68e-18ef47ea2e16","dog.jpg"]}})
img = tf.image.decode_jpeg(img, channels=3)
img.set_shape([None, None, 3])
img = tf.image.resize_images(img, (32, 32))
img = img.eval(session=sess) # convert to numpy array
img = np.expand_dims(img, 0) # make 'batch' of 1
img = img/255.0

pred = model.predict(img)
pred = labels["label_names"][np.argmax(pred)]
pred
