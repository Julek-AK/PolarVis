"""
low-level torch functions that process image arrays 
"""

from core.utils import *


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