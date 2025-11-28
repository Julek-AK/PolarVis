"""
low-level torch functions that process image arrays 
"""

from core.utils import *
import torch


def compute_metapixel_torch(I_unpol, I_pol, theta_pol):
    I_1 = .5 * I_unpol + I_pol * torch.cos(theta_pol + torch.pi / 2) ** 2
    I_2 = .5 * I_unpol + I_pol * torch.cos(theta_pol - torch.pi / 4) ** 2
    I_3 = .5 * I_unpol + I_pol * torch.cos(theta_pol + torch.pi / 4) ** 2
    I_4 = .5 * I_unpol + I_pol * torch.cos(theta_pol) ** 2
    return torch.stack([I_1, I_2, I_3, I_4], dim=-1)


def metapixel_MSE(pred, true):
    return torch.mean((pred - true) ** 2, dim=-1)


def resolve_intensities(image_arr, device, test=False, verbose=False):
    """
    Takes an image array, splits into metapixels, then for each one computes the unpolarized and polarized intensities
    as well as angle of polarization utilising torch gradient descent
    """

    image_arr_metapx = raw_to_metapixels(image_arr)
    if test: image_arr_metapx = image_arr_metapx[:5, :5]

    # Initialise tensors
    metapx_tensor = torch.tensor(image_arr_metapx, dtype=torch.float32, device=device)
    H, W, _, _ = metapx_tensor.shape
    metapx_tensor = metapx_tensor.reshape(H, W, 4)

    I_unpol = torch.full((H, W), 1., device=device, requires_grad=True)
    I_pol = torch.full((H, W), 1., device=device, requires_grad=True)
    theta = torch.zeros((H, W), device=device, requires_grad=True)

    # Fitting loop
    optimizer = torch.optim.Adam([I_unpol, I_pol, theta], lr=.5)
    n_epochs = 3000
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred = compute_metapixel_torch(I_unpol, I_pol, theta)
        loss = metapixel_MSE(pred, metapx_tensor).mean()
        loss.backward()
        optimizer.step()
        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}: loss = {loss.item()}")
    if verbose: print(f"Epoch {n_epochs}: loss = {loss.item()}")

    result_tensor = torch.stack([I_unpol, I_pol, theta], dim=-1).detach().cpu()
    result_arr = result_tensor.numpy()

    return result_arr



# =============================================
# CHATGPT CODE FOR SOLVING THE PROBLEM ANALYTICALLY
# =============================================
PHIS = torch.tensor([torch.pi/2, -torch.pi/4, torch.pi/4, 0.0])  # shape (4,)

# TODO pseudo-inverse computation is completely independent from your image case,
# move its computation to pipeline initialisation
# also make sure this thing can handle batches, as for small cases cpu is far faster

def analytic_resolve_metapixels(image_arr_metapx, device="cpu", eps=1e-12):
    """
    image_arr_metapx: numpy or torch array of shape (H, W, 4) with the 4 measured intensities per metapixel
    Returns: numpy array (H, W, 3) with [I_unpol, I_pol, theta] per metapixel (theta in radians, in [0, pi))
    """
    # ensure torch tensor on device
    if not torch.is_tensor(image_arr_metapx):
        y = torch.tensor(image_arr_metapx, dtype=torch.float32, device=device)
    else:
        y = image_arr_metapx.to(device, dtype=torch.float32)
    H, W, n_ch = y.shape
    assert n_ch == 4, "expected 4 intensities per metapixel"

    # build design matrix X (4 x 3): cols = [1, cos(2phi), sin(2phi)]
    phis = PHIS.to(device)
    cos2phi = torch.cos(2.0 * phis)   # (4,)
    sin2phi = torch.sin(2.0 * phis)   # (4,)
    X = torch.stack([torch.ones_like(cos2phi), cos2phi, sin2phi], dim=1)  # (4,3)

    # compute pseudo-inverse once: pinv = (X^T X)^{-1} X^T  -> shape (3,4)
    XtX = X.t() @ X                    # (3,3)
    # small regularizer for numerical stability
    reg = eps * torch.eye(3, device=device)
    XtX_inv = torch.linalg.inv(XtX + reg)  # (3,3)
    pinv = XtX_inv @ X.t()                  # (3,4)

    # reshape y to (4, H*W)
    y_flat = y.view(-1, 4).t()  # (4, H*W)

    # solve for params: params = pinv @ y_flat -> (3, H*W)
    params = pinv @ y_flat

    A = params[0, :].view(H, W)   # 0.5 * (I_unpol + I_pol)
    C = params[1, :].view(H, W)   # B cos(2theta)
    D = params[2, :].view(H, W)   # -B sin(2theta)

    # B = sqrt(C^2 + D^2)
    B = torch.sqrt(C * C + D * D + eps)

    # I_pol = 2 * B
    I_pol = 2.0 * B

    # I_unpol = 2A - I_pol
    I_unpol = 2.0 * A - I_pol

    # compute 2*theta = atan2(-D, C)  -> theta = 0.5 * atan2(-D, C)
    two_theta = torch.atan2(-D, C)   # in (-pi, pi]
    theta = 0.5 * two_theta

    # normalize theta into [0, pi)
    theta = (theta % torch.pi)

    # clamp physically (optional): I_pol and I_unpol >= 0
    I_pol = torch.clamp(I_pol, min=0.0)
    I_unpol = torch.clamp(I_unpol, min=0.0)

    result = torch.stack([I_unpol, I_pol, theta], dim=-1)  # (H, W, 3)
    return result.cpu().numpy()


def analytic_resolve_metapixels_for_testing(img_arr: NDArray, **kwargs) -> NDArray:
    """
    Wrapper for the analytic_resolve_metapixels that ensures the function signature is
    as expected by the testing module
    """
    img_arr_metapx = raw_to_metapixel_list(img_arr)
    return analytic_resolve_metapixels(img_arr_metapx, **kwargs)
