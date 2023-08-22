import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib.pyplot as plt

sys.path.append('src/')

def create_glowy_text_image(text,
                            name,
                            image_size):
    # 0. set file path
    file_path = 'examples/' + name.replace(' ', '') + '.png'
    
    # 1. Create a black canvas with the specified image size
    img = Image.new('RGB', image_size, 'black')
    draw = ImageDraw.Draw(img)
    
    # 2. Load a font for the text. We use the DejaVuSans-Bold font as an example.
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font_size = min(image_size)  # Start with a font size that's as large as the smallest image dimension
    
    # 3. Determine the largest possible font size that fits the canvas.
    # This is done by continuously decreasing the font size until the text fits within the image.
    font = ImageFont.truetype(font_path, font_size)
    while font.getsize(text)[0] > img.width or font.getsize(text)[1] > img.height:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)
    
    # Calculate the position to center the text on the canvas
    text_width, text_height = draw.textsize(text, font=font)
    position = ((img.width - text_width) / 2, (img.height - text_height) / 2)
    
    # 4. Render the text onto the canvas with the determined font size
    draw.text(position, text, fill='white', font=font)
    
    # 5. Apply a glow effect to the text.
    # This is achieved by creating a blurred version of the image and then blending it with the original.
    blurred = img.filter(ImageFilter.GaussianBlur(radius=5))
    img = Image.blend(img, blurred, alpha=0.5)
    img.save(file_path)

    return file_path

# source https://github.com/MaxRobinsonTheGreat/mandelbrotnn
class ImageDataset(Dataset):
    def __init__(self, image_path):
        # Load image, convert to grayscale and scale pixel values to [0, 1]
        self.image = Image.open(image_path).convert('L')
        self.image = ToTensor()(self.image)

        # Get image dimensions
        self.height, self.width = self.image.shape[1:]

    def __len__(self):
        return self.height * self.width

    def __getitem__(self, idx):
        # Convert flat index to 2D coordinates
        row = idx // self.width
        col = idx % self.width

        # Scale coordinates to [-1, 1]
        input_tensor = torch.tensor([col / (self.width / 2) - 1, (self.height-row) / (self.height / 2) - 1])

        # Get pixel value
        output_tensor = self.image[0, row, col]

        return input_tensor, output_tensor
    
    def display_image(self):
        # uses the getitem method to get each pixel value and displays the final image. used for debugging
        image = torch.zeros((self.height, self.width))
        for i in range(len(self)):
            row = i // self.width
            col = i % self.width
            image[row, col] = self[i][1]
        plt.imshow(image, cmap='gray')
        plt.show()



