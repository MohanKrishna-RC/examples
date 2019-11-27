import sys
import datetime
import imageio

# VALID_EXTENSIONS = ('png', 'jpg','JPEG')


# def create_gif(filenames, duration):
#     images = []
#     for filename in filenames:
#         images.append(imageio.imread(filename))
#     output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
#     imageio.mimsave(output_file, images, duration=duration)


# if __name__ == "__main__":
#     script = sys.argv.pop(0)

#     if len(sys.argv) < 2:
#         print('Usage: python {} <duration> <path to images separated by space>'.format(script))
#         sys.exit(1)

#     duration = float(sys.argv.pop(0))
#     filenames = sys.argv


#     if not all(f.lower().endswith(VALID_EXTENSIONS) for f in filenames):
#         print('Only png and jpg files allowed')
#         sys.exit(1)

#     create_gif(filenames, duration)

import glob
import moviepy.editor as mpy

gif_name = ""
fps = 12
Images_dir = ""
file_list = glob.glob(Images_dir + '*.jpg') # Get all the pngs in the current directory
list.sort(file_list, key=lambda x: int(x.split('_')[1].split('.jpg')[0])) # Sort the images by #, this may need to be tweaked for your use case
clip = mpy.ImageSequenceClip(file_list, fps=fps)
clip.write_gif('{}.gif'.format(gif_name), fps=fps)