import cv2
import os
import shutil 
import torch

def images_to_video(image_folder='images',
                    video_path='videos/training_video.avi',
                    fps=24,
                    clear_images=False):
    """
    Convert images in a specified folder to a video.
    
    Parameters:
    - image_folder (str): Path to the folder containing the images.
    - video_path (str): Path where the generated video will be saved.
    - fps (int): Frames per second for the generated video.
    - clear_images (Bool): Clears the images folder. 
    """
    
    # Get all files from the folder
    images = [img for img in os.listdir(image_folder) if not img.startswith('._')]
    
    # Sort the file names
    images = sorted(images)
    
    # Read the first image to get the shape
    example_img = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = example_img.shape

    # Define the codec using VideoWriter_fourcc and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for i in range(len(images)):
        img_path = os.path.join(image_folder, images[i])
        img = cv2.imread(img_path)
        out.write(img)  # Writes a frame to the video

    # Finalize the video file
    out.release()
    cv2.destroyAllWindows()
    
    print(f'Video saved at {video_path}')

    # Option to clear the images folder after saving the video
    if clear_images:
        shutil.rmtree(image_folder)
        os.makedirs(image_folder)
        print(f'Cleared images from {image_folder}')    

def renderModel(model, resx, resy, xmin=-2.4, xmax=1, yoffset=0, linspace=None, max_gpu=False):
    """ 
    Generates an image of a model's predition of the mandelbrot set in 2d linear\
    space with a given resolution. Prioritizes resolution over ease of positioning,\
    so the resolution is always preserved and the y range cannot be directly tuned.

    Parameters: 
    model (torch.nn.Module): torch model with input size 2 and output size 1
    resx (int): width of image
    resy (int): height of image
    xmin (float): minimum x value in the 2d space
    xmax (float): maximum x value in the 2d space
    yoffset (float): how much to shift the y position
    max_depth (int): max depth param for mandelbrot function
    linspace (torch.tensor())): linear space of (x, y) points corresponding to each\
        pixel. Shaped into batches such that shape == (resx, resy, 2) or shape == \
        (resx*resy, 2). Default None, and a new linspace will be generated automatically.
    max_gpu (boolean): if True, the entire linspace will be squeezed into a single batch. 
        Requires decent gpu memory size and is significantly faster.

    Returns: 
    numpy array: 2d float array representing an image 
    """
    with torch.no_grad():
        model.eval()
        if linspace is None:
            linspace = generateLinspace(resx, resy, xmin, xmax, yoffset)
        
        linspace = linspace.cuda()
        
        if not max_gpu:
            # slices each row of the image into batches to be fed into the nn.
            im_slices = []
            for points in linspace:
                im_slices.append(model(points))
            im = torch.stack(im_slices, 0)
        else:
            # otherwise cram the entire image in one batch
            if linspace.shape != (resx*resy, 2):
                linspace = torch.reshape(linspace, (resx*resy, 2))
            im = model(linspace).squeeze()
            im = torch.reshape(im, (resy, resx))


        im = torch.clamp(im, 0, 1) # doesn't add weird pure white artifacts
        linspace = linspace.cpu()
        torch.cuda.empty_cache()
        model.train()
        return im.squeeze().cpu().numpy()


def generateLinspace(resx, resy, xmin=-2.4, xmax=1, yoffset=0):
    iteration = (xmax-xmin)/resx
    X = torch.arange(xmin, xmax, iteration).cuda()[:resx]
    y_max = iteration * resy/2
    Y = torch.arange(-y_max-yoffset,  y_max-yoffset, iteration)[:resy]
    linspace = []
    for y in Y:
        ys = torch.ones(len(X)).cuda() * y
        points = torch.stack([X, ys], 1)
        linspace.append(points)
    return torch.stack(linspace, 0)
