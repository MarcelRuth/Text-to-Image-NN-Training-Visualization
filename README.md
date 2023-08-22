# Text-to-Image Neural Network Training Visualization

# Inspired and Partly copied from https://github.com/MaxRobinsonTheGreat/mandelbrotnn

This repository contains code to visualize the training of a neural network model in PyTorch. 
The neural network is trained to learn 2D representations of a given text or picture.
After training, these images are compiled into a video, visualizing the learning process.

## Usage

1. **Generate 2D Data Points from Text:**
   - The `text_to_points` function in `src/text_to_image.py` generates 2D data points that visually represent a given text.
   
2. **Train the Neural Network:**
   - The `train` function in `src/train.py` contains the training loop, where the neural network is trained to learn the representations of the text.
   
3. **Create a Video from Saved Images After Training:**
   - After training, call the `images_to_video` function from `src/video_creator.py` to compile the saved images into a video.

## Requirements

Install the necessary packages with:

pip install -r requirements.txt

## Instructions

### Clone the Repository:
git clone https://github.com/yourusername/Text-to-Image-NN-Training-Visualization.git
### Navigate to the Project Directory:
cd Text-to-Image-NN-Training-Visualization
### Install the Requirements:
pip install -r requirements.txt
### Run Your Training Script:
Change the settings in the run_example.py file and start creating cool videos!


# Example:

Xylene | ML-Xylene
:-: | :-:
<img src="examples/Aromatic.png" width="350" title="Xylene"> | ![](videos/Aromatic.gif)
