import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.logging import logging

def compute_weighted_masked_loss(loss, mask=None):
    # loss: C x H x W, mask: 1 x H x W
    assert len(loss.shape) == 3
    if mask is None:
        avg_loss = loss.mean(0)
        return avg_loss.mean(), loss
    else:
        if mask.shape[0] != 1:
            logging.error(f'Mask MUST be 1 channel!')
            raise ValueError
        if loss.shape[1] != mask.shape[1] or loss.shape[2] != mask.shape[2]:
            logging.error(f'Diff and mask MUST have the same shape, got {loss.shape} and {mask.shape} respectively!')
            raise ValueError
        avg_loss = (loss * mask).mean(0)
        return avg_loss.sum() / mask.mean(0).sum(), loss * mask

def masked_l1(network_output, gt, mask=None):
    # C x H x W
    loss = torch.abs((network_output - gt))
    return compute_weighted_masked_loss(loss, mask=mask)

def masked_l2(network_output, gt, mask=None):
    # C x H x W
    loss = (network_output - gt) ** 2
    return compute_weighted_masked_loss(loss, mask=mask)

def masked_ssim(img1, img2, mask=None, window_size=11):
    # C x H x W
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _masked_ssim(img1, img2, window, window_size, channel, 
                mask=mask)

def _masked_ssim(img1, img2, window, window_size, channel, mask=None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return compute_weighted_masked_loss(ssim_map, mask=mask)

def masked_mse(img1, img2, mask=None):
    loss = (img1 - img2) ** 2
    return compute_weighted_masked_loss(loss, mask=mask)

def masked_psnr(img1, img2, mask=None):
    mse, diff = masked_mse(img1=img1, img2=img2, mask=mask)
    return 20 * torch.log10(1.0 / torch.sqrt(mse)), diff

# Losses for relightable gaussian
def cal_gradient(data):
    """
    data: [1, C, H, W]
    """
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(data.device)

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(data.device)

    weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    grad_x = F.conv2d(data, weight_x, padding='same')
    grad_y = F.conv2d(data, weight_y, padding='same')
    gradient = torch.abs(grad_x) + torch.abs(grad_y)

    return gradient

def bilateral_smooth_loss(data, image, mask):
    """
    image: [C, H, W]
    data: [C, H, W]
    mask: [C, H, W]
    """
    rgb_grad = cal_gradient(image.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)  # [1, H, W]
    data_grad = cal_gradient(data.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)  # [1, H, W]

    smooth_loss = (data_grad * (-rgb_grad).exp() * mask).mean()

    return smooth_loss

# Losses for gaussian shader
def erode(img_in, erode_size=4):
    img_out = np.copy(img_in)
    kernel = np.ones((erode_size, erode_size), np.uint8)
    img_out = cv2.erode(img_out, kernel, iterations=1)

    return img_out

def predicted_normal_loss(normal, normal_ref, alpha=None):
    """Computes the predicted normal supervision loss defined in ref-NeRF."""
    # normal: (3, H, W), normal_ref: (3, H, W), alpha: (3, H, W)
    if alpha is not None:
        device = alpha.device
        weight = alpha.detach().cpu().numpy()[0]
        weight = (weight*255).astype(np.uint8)

        weight = erode(weight, erode_size=4)

        weight = torch.from_numpy(weight.astype(np.float32)/255.)
        weight = weight[None,...].repeat(3,1,1)
        weight = weight.to(device) 
    else:
        weight = torch.ones_like(normal_ref)

    w = weight.permute(1,2,0).reshape(-1,3)[...,0].detach()
    n = normal_ref.permute(1,2,0).reshape(-1,3).detach()
    n_pred = normal.permute(1,2,0).reshape(-1,3)
    loss = (w * (1.0 - torch.sum(n * n_pred, axis=-1))).mean()

    return loss

def delta_normal_loss(delta_normal_norm, alpha=None):
    # delta_normal_norm: (3, H, W), alpha: (3, H, W)
    if alpha is not None:
        device = alpha.device
        weight = alpha.detach().cpu().numpy()[0]
        weight = (weight*255).astype(np.uint8)

        weight = erode(weight, erode_size=4)

        weight = torch.from_numpy(weight.astype(np.float32)/255.)
        weight = weight[None,...].repeat(3,1,1)
        weight = weight.to(device) 
    else:
        weight = torch.ones_like(delta_normal_norm)

    w = weight.permute(1,2,0).reshape(-1,3)[...,0].detach()
    l = delta_normal_norm.permute(1,2,0).reshape(-1,3)[...,0]
    loss = (w * l).mean()

    return loss

def zero_one_loss(img):
    zero_epsilon = 1e-3
    val = torch.clamp(img, zero_epsilon, 1 - zero_epsilon)
    loss = torch.mean(torch.log(val) + torch.log(1 - val))
    return loss

def fourier2d(image):
    f = torch.fft.fft2(image)
    fshift = torch.fft.fftshift(f)
    return fshift

def inv_fourier2d(fshift):
    f_ishift = torch.fft.ifftshift(fshift)
    img = torch.fft.ifft2(f_ishift)
    img = torch.abs(img)
    return img

def compute_amp_and_deg(image, pass_filter=None):
    fft = fourier2d(image)
    if pass_filter is not None:
        fft = fft * pass_filter
    amp = torch.abs(fft)
    deg = torch.where(
        fft.real == 0, 
        torch.sign(fft.imag) * torch.pi / 2, 
        torch.arctan(fft.imag / fft.real))
    return amp, deg

def freq_reg_loss(img1, img2, pass_filter=None):
    C, H, W = img1.shape
    denorm = 1 / np.sqrt(H * W)
    amp1, deg1 = compute_amp_and_deg(img1, pass_filter=pass_filter)
    amp1 = amp1.sum(-1).sum(-1) * denorm
    deg1 = deg1.sum(-1).sum(-1) * denorm
    amp2, deg2 = compute_amp_and_deg(img2, pass_filter=pass_filter)
    amp2 = amp2.sum(-1).sum(-1) * denorm
    deg2 = deg2.sum(-1).sum(-1) * denorm
    return torch.abs(amp1 - amp2).mean() + torch.abs(deg1 - deg2).mean()

def get_filter2d(H, W, cutoff_freq=10):
    x = torch.linspace(-W//2, W//2, W)
    y = torch.linspace(-H//2, H//2, H)
    y, x = torch.meshgrid(y, x)
    h_freq = cutoff_freq
    w_freq = cutoff_freq / H * W
    distance = torch.sqrt(x**2/w_freq**2 + y**2/h_freq**2)
    lp_filter = torch.where(distance <= 1, 1, 0)
    return lp_filter[None]

