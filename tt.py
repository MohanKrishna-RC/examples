import datetime
a = datetime.datetime.now()
b = datetime.datetime(2018,11,13,13,10,10, 0)

import wget
url = 'http://www.futurecrew.com/skaven/song_files/mp3/razorback.mp3'
filename = wget.download(url)
filename

#To_play_all_songs_in_a_row

# from pydub import AudioSegment
# from pydub.playback import play
# import glob

# for file in glob.glob('/home/mohan/Downloads/Untitled Folder/*'):
#     print(file)
#     for i in range(len(file)):
#         sound = AudioSegment.from_file(file, format="mp3")
#         play(sound)


#To_play_all_songs_in_as_our_wish

from pydub import AudioSegment
from pydub.playback import play
import glob
import datetime
import pygame

a = len(list(glob.iglob("/home/mohan/Downloads/Untitled Folder/*", recursive=True)))
print(a)
b = list(glob.iglob("/home/mohan/Downloads/Untitled Folder/*", recursive=True))
# for file in glob.glob('/home/mohan/Downloads/Untitled Folder/*'):
#     print(file)
#     for i in range(a):
#         sound = AudioSegment.from_file(file, format="mp3")
#         play(sound[i])
sound = []
for i in range(a):
    sound.append(AudioSegment.from_file(b[i], format="mp3"))

# while True:
#     play(sound[16])
#     play(sound[9])
#     res = input(“Continue Y/N : ”)
#     if res == ‘N’ or res == ‘n’:
#         break
time = datetime.datetime.now()
time1 = datetime.datetime(2018,11,13,13,10,10, 0)
time2 = datetime.datetime(2018,11,13,16,16,40, 0)
time3 = datetime.datetime(2018,11,13,18,30,10, 0)
while True:
    for i in range(a):
        if time == time1 or time == time2 or time == time3: 
            time.sleep(100)
        else:
            #play(sound[i])
            play(sound[1])
            play(sound[2])
            play(sound[5])
            print(i)
            continue

def song():
    if time == time1 or time == time2 or time == time3: 
        time.sleep(100)
    else:
        play(sound[1])


