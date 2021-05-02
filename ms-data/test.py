import os
import copy

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

### VARIABLES
CONTENT_DIR = "ms-data/content"
STYLE_DIR = "ms-data/style"
RESULT_DIR = "ms-data/result"

DEVICE = torch.device("cuda")

STYLIZE_MODES = range(4)
STYLIZE_FIRST, STYLIZE_SECOND, STYLIZE_HALF, STYLIZE_MIX = STYLIZE_MODES

# WOW THIS FAR MORE COMPLICATED!
class ImageManager(object):
    def __init__(self):
        super(ImageManager, self).__init__()
        self.content_source_size = None
        self.toPilImage = transforms.ToPILImage()

    def load_images(self, content_img_name, style_img_name, style2_img_name):
        content_img_path = os.path.join(CONTENT_DIR, content_img_name)
        style_img_path = os.path.join(STYLE_DIR, style_img_name)
        style2_img_path = os.path.join(STYLE_DIR, style2_img_name)
        
        img1 = Image.open(content_img_path)
        img2 = Image.open(style_img_path)
        img3 = Image.open(style2_img_path)

        self.content_source_size = img1.size

        scale = 256 / max(img1.size)
        img1 = img1.resize((round(img1.size[0] * scale), round(img1.size[1] * scale)), Image.ANTIALIAS)

        scale = 256 / min(img2.size)
        img2 = img2.resize((round(img2.size[0] * scale), round(img2.size[1] * scale)), Image.ANTIALIAS)

        scale = 256 / min(img3.size)
        img3 = img3.resize((round(img3.size[0] * scale), round(img3.size[1] * scale)), Image.ANTIALIAS)

        loader = transforms.Compose([
                    transforms.CenterCrop((img1.size[1], img1.size[0])), #нормируем размер изображения
                    transforms.ToTensor()])  #превращаем в удобный формат

        img1 = loader(img1).unsqueeze(0)
        img2 = loader(img2).unsqueeze(0)
        img3 = loader(img3).unsqueeze(0)

        return img1.to(DEVICE, torch.float), img2.to(DEVICE, torch.float), img3.to(DEVICE, torch.float)

    def show_image(self, tensor, title=None, save_result=False):
        """Отрисовка и сохранение
        """
        img = tensor.cpu().clone()
        img = img.squeeze(0)
        img = self.toPilImage(img)

        plt.figure()
        plt.imshow(img)
        if title is not None:
            plt.title(title)
        plt.grid(False)
        plt.axis('off')
        plt.show()

        if save_result:
            save_path = os.path.join(RESULT_DIR, "{}.jpg".format(title if title else "stylized"))

            if self.content_source_size is not None:
                img = img.resize(self.content_source_size, Image.ANTIALIAS)
            else:
                print("Invalid content_source_size: {}".format(self.content_source_size))

            img.save(save_path)


if __name__ == '__main__':
    img_manager = ImageManager()
    content_img_name = "corgi.jpg"
    style_img_name = "mosaic.jpg"
    style2_img_name = "vg_starry_night.jpg"

    content_img, style_img, style2_img = img_manager.load_images(content_img_name, style_img_name, style2_img_name)

    img_manager.show_image(content_img, title='Content Image')

    img_manager.show_image(style_img, title='Style Image')

    img_manager.show_image(style2_img, title='Style 2 Image')