import matplotlib
import matplotlib.pyplot as plt
import time
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
# Show all tags in the log file
# try:
for filename in glob.iglob('/home/mohan/logs/*', recursive=True):
      event_acc = EventAccumulator(filename)
      event_acc.Reload()
      print(event_acc.Tags())

# # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
w_times, step_nums  = zip(*event_acc.Scalars('Accuracy/Accuracy'))
print(w_times)
print(step_nums)
print(vals)
w_times, step_nums  = zip(*event_acc.Scalars('loss/loss'))
print(w_times)
print(step_nums)
print(vals)

# pull out four metrics, plot
while True:
  hist = {
    'Accuracy' : [x.value for x in event_acc.Scalars('Accuracy/Accuracy')],
    # 'Validation Accuracy' : [x.value for x in 
    #                         event_acc.Scalars('val_acc')],
    'Loss' : [x.value for x in event_acc.Scalars('loss/loss')]
    # 'Validation Loss' : [x.value for x in event_acc.Scalars('val_loss')]
  }
  
  fig = plt.figure()
  keys = ['Accuracy', 'Loss']
  for i,thing in enumerate(keys):
      trace = hist[thing]
      plt.subplot(2,2,i+1)
      plt.plot(range(len(trace)),trace)
      plt.title(thing)
  # plt.show()
  fig.set_tight_layout(True)
  # plt.close(fig=None)
  time.sleep(15)





