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

import model

### VARIABLES
CONTENT_DIR = "ms-data/content"
STYLE_DIR = "ms-data/style"
RESULT_DIR = "ms-data/result"

DEVICE = torch.device("cuda")

STYLIZE_FIRST, STYLIZE_SECOND, STYLIZE_HALF, STYLIZE_MIX = [0, 1, 2, 3]

def load_images(content, style1, style2):
    content_path = os.path.join(CONTENT_DIR, content)
    style1_path = os.path.join(STYLE_DIR, style1)
    style2_path = os.path.join(STYLE_DIR, style2)

    img1 = Image.open(content_path)
    img2 = Image.open(style1_path)
    img3 = Image.open(style2_path)

    scale = 256 / max(img1.size)
    img1 = img1.resize((round(img1.size[0] * scale), round(img1.size[1] * scale)), Image.ANTIALIAS)

    scale = 256 / min(img2.size)
    img2 = img2.resize((round(img2.size[0] * scale), round(img2.size[1] * scale)), Image.ANTIALIAS)

    scale = 256 / min(img3.size)
    img3 = img3.resize((round(img3.size[0] * scale), round(img3.size[1] * scale)), Image.ANTIALIAS)

    loader = transforms.Compose([transforms.CenterCrop(img1.size[::-1]), transforms.ToTensor()])

    img1 = loader(img1).unsqueeze(0)
    img2 = loader(img2).unsqueeze(0)
    img3 = loader(img3).unsqueeze(0)

    return img1.to(DEVICE, torch.float), img2.to(DEVICE, torch.float), img3.to(DEVICE, torch.float)

def show_image(img, title='', save=False):
    img = img.cpu().clone()
    img = img.squeeze(0)
    img = transforms.functional.to_pil_image(img)

    plt.figure()
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.grid(False)
    plt.axis('off')

    if save:
        save_path = os.path.join(RESULT_DIR, "{}.jpg".format(title if title else "stylized"))
        img.save(save_path)


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
        img = tensor.cpu().clone()
        img = img.squeeze(0)
        img = self.toPilImage(img)

        plt.figure()
        plt.imshow(img)
        if title is not None:
            plt.title(title)
        plt.grid(False)
        plt.axis('off')


        if save_result:
            save_path = os.path.join(RESULT_DIR, "{}.jpg".format(title if title else "stylized"))

            if self.content_source_size is not None:
                img = img.resize(self.content_source_size, Image.ANTIALIAS)
            else:
                print("Invalid content_source_size: {}".format(self.content_source_size))

            img.save(save_path)


if __name__ == '__main__':
    content_img_name = "corgi.jpg"
    style_img_name = "mosaic.jpg"
    style2_img_name = "vg_starry_night.jpg"

    content_img, style_img, style2_img = load_images(content_img_name, style_img_name, style2_img_name)

    show_image(content_img, title='Content Image')
    show_image(style_img, title='Style Image')
    show_image(style2_img, title='Style 2 Image')
    plt.show() # show all images

    style_transfer = model.StyleTransfer(content_img, style_img, style2_img)

    def stylize(mode):
        style_transfer.build_model(mode)
        return style_transfer.run(num_steps=300, content_weight=1, style_weight=1e5, style2_weight=2e5)
    
    stylized_img = stylize(STYLIZE_MIX)

    show_image(stylized_img, title='stylized_first', save=True)
    plt.show() # show all images