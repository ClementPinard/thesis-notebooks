import torch
from torch.nn import functional as F


def multiscale(smooth_loss):
    def accumulate(pred, img, kappa, *args, **kwargs):
        if type(pred) not in [tuple, list]:
            pred = [pred]

        loss = 0
        weight = 1.

        for pred_scaled in pred:
            b, _, h, w = pred_scaled.shape
            if img is not None:
                img_scaled = F.interpolate(img, (h, w), mode='area').norm(p=1, dim=1, keepdim=True)
            else:
                img_scaled = None
            loss += smooth_loss(pred_scaled, img_scaled, kappa, *args, **kwargs) * weight
            weight /= 2
        return loss
    return accumulate


@torch.no_grad()
def img_gradient(img, shift=1):
    '''
    To get a normalized gradient, the difference map must be divided by shift
    typical shift sizes are 1 to get gradients between points,
    and 2, to get gradients at points.
    If img input is of size B, C, H, W ,
    the outputs will be of sizes
    B, 1, H-shift, W and B, 1, H, W-shift
    '''

    dy_i = img[:, :, shift:] - img[:, :, :-shift]
    dy_i = dy_i.norm(dim=1, keepdim=True)

    dx_i = img[:, :, :, shift:] - img[:, :, :, :-shift]
    dx_i = dx_i.norm(dim=1, keepdim=True)

    return dy_i/shift, dx_i/shift


@multiscale
def diffusion_loss(pred_map, img, kappa):

    deltaS = pred_map[:,:,1:] - pred_map[:,:,:-1]
    deltaE = pred_map[:,:,:,1:] - pred_map[:,:,:,:-1]

    if img is not None:
        dy_i, dx_i = img_gradient(img, 1)
        gy = (deltaS**2) * torch.exp(-((dy_i/kappa)**2))
        gx = (deltaE**2) * torch.exp(-((dx_i/kappa)**2))
    else:
        # Classic anisotropic diffusion loss.
        # Not really useful for depth smoothing
        gy = -kappa**2 * torch.exp(-((deltaS/kappa)**2))
        gx = -kappa**2 * torch.exp(-((deltaE/kappa)**2))
    return gx.mean() + gy.mean()


@multiscale
def grad_diffusion_loss(pred, img, kappa):
    if img is not None:
        dy_i, dx_i = img_gradient(img, 2)
        gx = torch.exp(-(dx_i.abs()/kappa)**2)
        gy = torch.exp(-(dy_i.abs()/kappa)**2)
    else:
        gx = gy = 1

    dy2 = pred[:,:, 2:] - 2 * pred[:,:,1:-1] + pred[:,:,:-2]
    dx2 = pred[:,:,:, 2:] - 2 * pred[:,:,:,1:-1] + pred[:,:,:,:-2]
    dy2 *= gy
    dx2 *= gx
    return (dx2*gx).pow(2).mean() + (dy2*gy).pow(2).mean()


@multiscale
def TV_loss(pred, img, kappa):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if img is not None:
        dy_i, dx_i = img_gradient(img, 1)
        gx = torch.exp(-dx_i/kappa)
        gy = torch.exp(-dy_i/kappa)
    else:
        gx, gy = 1, 1
    dy = pred[:, :, 1:] - pred[:, :, :-1]
    dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    return (gx*dx).abs().mean() + (gy*dy).abs().mean()


@multiscale
def TVV_loss(pred, img, kappa=0.1):
    if img is not None:
        dy_i, dx_i = img_gradient(img, 2)
        gx = torch.exp(-(dx_i.abs()/kappa)**2)
        gy = torch.exp(-(dy_i.abs()/kappa)**2)
    else:
        gx = gy = 1

    dy2 = pred[:,:, 2:] - 2 * pred[:,:,1:-1] + pred[:,:,:-2]
    dx2 = pred[:,:,:, 2:] - 2 * pred[:,:,:,1:-1] + pred[:,:,:,:-2]
    return (dx2*gx).abs().mean() + (dy2*gy).abs().mean()


@multiscale
def robust_grad_diffusion_loss(pred_map, img=None, kappa=0.1, gamma=0.1, iterations=10, loss_function='MSE'):
    deltaS = torch.zeros_like(pred_map)
    deltaE = deltaS.clone()
    deltaS[:,:,:-1] = pred_map[:,:,1:] - pred_map[:,:,:-1]
    deltaE[:,:,:,:-1] = pred_map[:,:,:,1:] - pred_map[:,:,:,:-1]

    diffused_deltaS, diffused_deltaE = gradient_diffusion([deltaS, deltaE], img, kappa, gamma, iterations)
    differences = (deltaS - diffused_deltaS, deltaE - diffused_deltaE)
    if loss_function == 'MSE':
        loss = (differences[0]**2).mean() + (differences[1]**2).mean()
    elif loss_function == 'abs':
        loss = differences[0].abs().mean() + differences[1].abs().mean()
    return loss


@multiscale
def robust_diffusion_loss(pred_map, img=None, kappa=0.1, gamma=0.1, iterations=10, loss_function='MSE'):
    diffused = PM_diffusion(pred_map, img, kappa, iterations, gamma)
    difference = (pred_map - diffused)
    if loss_function == 'MSE':
        loss = (difference**2).mean()
    elif loss_function == 'abs':
        loss = difference.abs().mean()
    return loss


@multiscale
def robust_grad_diffusion_loss2(pred_map, img=None, kappa=0.1, gamma=0.1, iterations=10):
    b,_,h,w = pred_map.shape

    deltaS = pred_map[:,:,2:] - pred_map[:,:,:-2]
    deltaE = pred_map[:,:,:,2:] - pred_map[:,:,:,:-2]

    smoothed_deltaS = PM_diffusion(deltaS, img[:,:,1:-1], kappa, iterations, gamma)
    smoothed_deltaE = PM_diffusion(deltaE, img[:,:,:,1:-1], kappa, iterations, gamma)

    loss = ((deltaS - smoothed_deltaS)**2).mean() + ((deltaE - smoothed_deltaE)**2).mean()
    return loss


@torch.no_grad()
def PM_diffusion(pred_map, img, kappa, iterations=1, gamma=0.1):
    b,c,h,w = pred_map.shape
    output = pred_map.clone()
    deltaS = torch.zeros_like(pred_map)
    deltaE = deltaS.clone()
    NS = deltaS.clone()
    EW = deltaS.clone()

    if img is not None:
        dy_i, dx_i = img_gradient(img, 1)
        deltaS_img = torch.zeros_like(pred_map)
        deltaE_img = deltaS_img.clone()

        deltaS_img[:,:,:-1,:] = dy_i
        deltaE_img[:,:,:,:-1] = dx_i

        # conduction gradients (only need to compute one per dim!)
        gS = torch.exp(-(deltaS_img/kappa)**2.)
        gE = torch.exp(-(deltaE_img/kappa)**2.)

    for _ in range(iterations):
        # calculate the diffs
        deltaS.zero_()
        deltaE.zero_()
        deltaS[:,:,:-1,:] = output[:,:,1:] - output[:,:,:-1]
        deltaE[:,:,:,:-1] = output[:,:,:,1:] - output[:,:,:,:-1]
        if img is None:
            gS = torch.exp(-(deltaS/kappa)**2)
            gE = torch.exp(-(deltaE/kappa)**2)

        # update matrices
        E = gE * deltaE
        S = gS * deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't ask questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[:,:,1:,:] -= S[:,:,:-1,:]
        EW[:,:,:,1:] -= E[:,:,:,:-1]
        output += gamma * (NS+EW)
    return output


@torch.no_grad()
def gradient_diffusion(gradients, img=None, kappa=0.1, gamma=0.1, iterations=10):
    '''
    Expected form for deltaS and deltaE is B,C,H,W
    deltaS[:,:,i,j] = pred[:,:,i+1,j] - pred[:,:,i,j]
    deltaS[:,:,H,:] = 0
    same thing for deltaE but with j.
    deltaE[:,:,i,j] = pred[:,:,i,j+1] - pred[:,:,i,j]
    deltaE[:,:,:,W] = 0

    It's important ! DO NOT compute gradient with a shift of 2
    '''
    deltaS, deltaE = gradients
    assert(deltaS.shape == deltaE.shape)

    gradS = deltaS.clone()
    gradE = deltaE.clone()
    if img is not None:
        dy_i, dx_i = img_gradient(img, 2)
        gE = torch.exp(-(dx_i/kappa)**2)
        gS = torch.exp(-(dy_i/kappa)**2)
    else:
        gE = gS = 1
    smoothed_deltaS = deltaS.clone()
    smoothed_deltaE = deltaE.clone()
    for i in range(iterations):
        div = torch.zeros_like(smoothed_deltaS)
        div[:,:,1:-1] = gS*(smoothed_deltaS[:,:,:-2] - smoothed_deltaS[:,:,1:-1])
        div[:,:,:,1:-1] += gE*(smoothed_deltaE[:,:,:,:-2] - smoothed_deltaE[:,:,:,1:-1])

        gradS[:] = div
        gradE[:] = div
        gradS[:,:,:-1,:] -= div[:,:,1:,:]
        gradE[:,:,:,:-1] -= div[:,:,:,1:]
        smoothed_deltaS += gamma * gradS
        smoothed_deltaE += gamma * gradE
    return smoothed_deltaS, smoothed_deltaE
