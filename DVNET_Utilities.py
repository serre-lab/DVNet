import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import os
import cv2

def transformation_sequence_from_image(image, seq_length, rotation = 20, scale = 2, shear=20, translation=20, clamp=[0,1],
                                       randomize = False, blur = False, sharpen = False, noise = False):
    """
    Generate a series of images from a given image representing a continuous affine transformation sequence. The output
    if a series of images which starts from the reference image and then changes in a linear "walk" along the affine
    transformation space until it reaches either a defined or randomly generated endpoint.

    Note at this time, there is no jitter / noise or direction change in the walk.

    Though not explicit, it is assumed that normalized, model-ready images are being provided, as though taken from a
    batch of images (e.g. input_image = image_batch[0,:,:,:])
    
    input:
        image:      A single (channel x size x size) sized image, presumably of Tensor type
        seq_length: The length of the output sequence, which will be one dimension larger than image and shaped as
                    (seq_length x channel x size x size) suitable for input into a neural network model, with the sequence
                    arranged along the batch dimension
        
        rotation:   Parameters to control the rotation, scale, shear, and translation of the sequence
        scale:      If randomize is False, these represent the amount of variation at the last frame
        shear:      If randomize is True, these represent the maximum range over which a translation may occur, but the
        transla...: actual end point of this sequence is chosen randomly from this range
        
        clamp:      After transformation and optional post-processing, clamp the output image to be between these values.

        randomize:  Flag to determine whether the exact transformation input should be used, or a randomly generated
                    transformation based on the inputs (intended to generated augmentation samples on-the-fly)
        blur:       Flag to determine whether to apply a Gaussian blur to the transformed images, to reduce edge artifact.
                    Can provide True for a default kernel, or provide a tuple specifying (kernel_size, sigma).
        sharpen:    Flag to determine whether to sharpen the transformed images. Can provide True for default sharpness,
                    or provide a sharpness parameter. Utilizes torchvision adjust_sharpness.
        noise:      Flag to determine whether to corrupt the translated image with white noise, randomly generated on
                    each frame. Can provide True for default noise (uniform 0-1, 30%), or a tuple of (Amplitude, Percent)
                 
    output:
        imstack:    (seq_length x channel x size x size) image stack representing the trasformation applied to the image
    """
    imstack = torch.zeros(seq_length,image.size()[0],image.size()[1],image.size()[2])

    if(randomize):
        Ro = np.linspace(0,random.uniform(-rotation,rotation),seq_length)
        Sc = np.linspace(1,random.uniform(1./scale,scale),seq_length)
        Tx = np.linspace(0,random.uniform(-translation,translation),seq_length)
        Ty = np.linspace(0,random.uniform(-translation,translation),seq_length)
        Sh = np.linspace(0,random.uniform(-shear,shear),seq_length)
    else:
        Ro = np.linspace(0,rotation,seq_length)
        Sc = np.linspace(1,scale,seq_length)
        Tx = np.linspace(0,translation,seq_length)
        Ty = np.linspace(0,translation,seq_length)
        Sh = np.linspace(0,shear,seq_length)
    
    for j,A in enumerate(zip(Ro,Sc,Tx,Ty,Sh)):
        ii = TF.affine(image,translate=(A[2],A[3]),scale=A[1],shear=A[4],angle=A[0])
        if blur:
            if blur is tuple:
                ii = TF.gaussian_blur(ii, kernel_size=blur[0], sigma=blur[1])
            else:
                sigma=A[1]*7./8.
                ii = TF.gaussian_blur(ii, kernel_size=int(2*np.ceil(2*sigma)+1),sigma=sigma)
        if sharpen:
            if sharpen is not bool:
                ii = TF.adjust_sharpness(ii, sharpen)
            else:
                ii = TF.adjust_sharpness(ii, 1.2)
        if noise:
            if noise is tuple:
                ii = corrupt_white_noise(ii,ratio=noise[1],amplitude=noise[0])
            else:
                ii = corrupt_white_noise(ii,0.30)
        if clamp:
            ii = torch.clamp(ii,clamp[0],clamp[1])
        imstack[j,:,:,:] = ii

    return imstack,(Tx,Ty,Sc,Ro,Sh)

def corrupt_white_noise(x,ratio,amplitude=1):
    return x*(1-ratio) + torch.rand(x.size())*ratio*amplitude