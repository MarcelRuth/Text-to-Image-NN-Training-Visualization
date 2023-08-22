import sys
sys.path.append('src/')

from train import train
from video_creator import images_to_video
import shutil
import os

mode = 'Picture'
# else 'Text'

# for both modes
name = 'PRS' # name of the generated png and mp4 file
             # only of mp4 if 'Picture' mode

# for 'Text' mode
text = "PRS" # will be written on the picture
image_size = (1440, 900)

# for 'Picture' mode
img_path = 'examples/Benzene.png'

# training parameters
num_epochs = 10
learning_rate = 0.001
batch_size = 16000

# video settings
save_interval = 2
fps = 24

# clear images
shutil.rmtree('images')
os.makedirs('images')
print(f'Cleared images')  
   
# Train the model
_ = train(mode=mode, 
          text=text if mode == 'Text' else None,
          name=name if mode == 'Text' else None,  
          img_path=img_path if mode == 'Picture' else None, 
          num_epochs=num_epochs, 
          learning_rate=learning_rate, 
          save_interval=save_interval, 
          batch_size=batch_size)

# Generate the training visualization video
# Use the 'clear images' option only when you
# are sure that you like the produced video!
images_to_video(image_folder='images',
                video_path=f'videos/{name}.mp4',
                fps=fps,
                clear_images=True)