import torch
import torch.nn as nn
import torch.optim as optim
from text_to_image import create_glowy_text_image, ImageDataset
from model import SkipConn
from video_creator import renderModel
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

def train(mode, 
          num_epochs=10, 
          learning_rate=0.001, 
          save_interval=2, 
          save_dir='images', 
          image_size=(1920, 1080), 
          batch_size=10000, 
          **kwargs): 

    if mode == 'Text':
        text = kwargs.get('text')
        if not text:
            raise ValueError("Text mode requires 'text' argument.")
        name = kwargs.get('name')
        if not name:
            raise ValueError("Text mode requires 'name' argument.")
        target_path = create_glowy_text_image(text, name, image_size)
    elif mode == 'Picture':
        img_path = kwargs.get('img_path')
        if not img_path:
            raise ValueError("Picture mode requires 'img_path' argument.")
        target_path = img_path
    else:
        raise ValueError("Invalid mode specified.")

    """
    Train a neural network to learn 2D representations of a given text.
    
    Parameters:
    - img_path (str):           Path to the target picture
    - text (str):               The input text string.
    - name (str):               The name of the generated png 
    - num_epochs (int):         The number of training epochs.
    - learning_rate (float):    The learning rate for the optimizer.
    - save_interval (int):      The interval at which glowing images will be saved during training.
    - save_dir (str):           The directory where the glowing images will be saved during training.
    
    Returns:
    - trained_model (nn.Module): The trained PyTorch model.
    """

    # source https://github.com/MaxRobinsonTheGreat/mandelbrotnn
    dataset = ImageDataset(target_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    resx, resy = dataset.width, dataset.height
    linspace = torch.stack(torch.meshgrid(torch.linspace(-1, 1, resx),
                                          torch.linspace(1, -1, resy)),
                                          dim=-1).cuda()
    
    # rotate the linspace 90 degrees
    linspace = torch.rot90(linspace, 1, (0, 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SkipConn(hidden_size=300, num_hidden_layers=30).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    iteration, frame = 0, 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        for x, y in tqdm(loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            x, y = x.cuda(), y.cuda()

            # Forward pass
            y_pred = model(x).squeeze()

            # Compute loss
            loss = criterion(y_pred, y)
            epoch_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            optimizer.step()
        
            # Save glowing images at specified intervals
            if iteration % save_interval == 0:
                plt.imsave(f'{save_dir}/frame_{frame:04d}.png',
                            renderModel(model,
                                        resx=resx,
                                        resy=resy,
                                        linspace=linspace), cmap='magma', origin='lower')
                frame += 1
            iteration += 1
        # Log the average loss per epoch
        print(f'Epoch {epoch+1}, Average Loss: {epoch_loss / len(loader)}')

    print('Training complete.')
    return model
