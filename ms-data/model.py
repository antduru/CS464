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

import loss
import norm

DEVICE = torch.device("cuda")

STYLIZE_FIRST, STYLIZE_SECOND, STYLIZE_HALF, STYLIZE_MIX = 0, 1, 2, 3

class StyleTransfer():
    def __init__(self, cnn, normalization_mean, normalization_std, content_layers, style_layers, content_img, style_img, style2_img):
        self.cnn = cnn
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std

        self.content_layers = content_layers
        self.style_layers = style_layers

        self.model = None
        self.step = 0

        self.content_losses = []
        self.style_losses = []
        self.style2_losses = []

        self.content_img = content_img
        self.style_img = style_img
        self.style2_img = style2_img

        self.mode = None

    def build_model(self, mode):
        print('Building the style transfer model..')

        if mode not in [0, 1, 2, 3]:
            raise ValueError("Invalid mode: {}".format(mode))

        use_mask = True if mode == STYLIZE_HALF else False

        # just in order to have an iterable access to list of content/style losses
        self.content_losses.clear()
        self.style_losses.clear()
        self.style2_losses.clear()

        cnn = copy.deepcopy(self.cnn)

        # normalization module
        normalization = norm.Normalization(self.normalization_mean, self.normalization_std).to(DEVICE)

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0 # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self.content_layers:
                # add content loss:
                content_feature = model(self.content_img).detach()
                content_loss = loss.ContentLoss(content_feature)
                model.add_module("content_loss_{}".format(i), content_loss)
                self.content_losses.append(content_loss)

            if name in self.style_layers:
                # add style loss:
                if mode != STYLIZE_SECOND:
                    style_feature = model(self.style_img).detach()
                    style_loss = loss.StyleLoss(style_feature, use_mask, invert_mask=False)
                    model.add_module("style_loss_{}".format(i), style_loss)
                    self.style_losses.append(style_loss)

                if mode != STYLIZE_FIRST:
                    style2_feature = model(self.style2_img).detach()
                    style2_loss = loss.StyleLoss(style2_feature, use_mask, invert_mask=True)
                    model.add_module("style2_loss_{}".format(i), style2_loss)
                    self.style2_losses.append(style2_loss)

        # now we trim off the layers after the last content and style losses
        #выбрасываем все уровни после последнего style loss или content loss
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], loss.ContentLoss) or isinstance(model[i], loss.StyleLoss):
                break

        self.model = model[:(i + 1)]
        self.mode = mode

    @staticmethod
    def get_input_optimizer(xx):
        return optim.LBFGS([xx.requires_grad_()])

    def run(self, num_steps, content_weight, style_weight, style2_weight):
        """
        Run the style transfer
        """
        def closure():
            # correct the values
            # это для того, чтобы значения тензора картинки не выходили за пределы [0;1]
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()

            self.model(input_img)

            content_score = 0
            style_score = 0
            style2_score = 0

            for cl in self.content_losses:
                content_score += cl.loss

            #взвешивание ошибки на контенте
            content_score *= content_weight

            if self.mode != STYLIZE_SECOND:
                for sl in self.style_losses:
                    style_score += sl.loss
                #взвешивание ошибки на стиле 1
                style_score *= style_weight

            if self.mode != STYLIZE_FIRST:
                for sl2 in self.style2_losses:
                    style2_score += sl2.loss
                #взвешивание ошибки на стиле 2
                style2_score *= style2_weight

            loss = content_score

            if self.mode == STYLIZE_FIRST:
                loss += style_score
            elif self.mode == STYLIZE_SECOND:
                loss += style2_score
            else:
                loss += style_score + style2_score

            loss.backward()

            self.step += 1
            if self.step % 50 == 0:
                print(log_pattern.format(
                    self.step, content_score.item(),
                    style_score if isinstance(style_score, int) else style_score.item(),
                    style2_score if isinstance(style2_score, int) else style2_score.item()))

            return loss

        if self.mode is None:
            raise ValueError("No init mode: {}".format(self.mode))

        input_img = self.content_img.clone().detach()
        optimizer = self.get_input_optimizer(input_img)

        log_pattern = 'Step: {}\tContent Loss: {:.4f} Style Loss: {:.4f} Style 2 Loss: {:.4f}'
        print('Optimizing..')
        self.step = 0
        while self.step <= num_steps:
            optimizer.step(closure)

        # a last correction...
        input_img.data.clamp_(0, 1)

        return input_img.cpu().detach()

if __name__ == '__main__':
    content_layers_default = ('conv_4',)
    style_layers_default = ('conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5')
    
    vgg19 = models.vgg19(pretrained=True).features.to(DEVICE).eval()
