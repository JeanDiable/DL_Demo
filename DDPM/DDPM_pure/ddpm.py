import torch
import torch.nn as nn


class DDPM:
    def __init__(
        self, device, n_steps: int, min_beta: float = 0.0001, max_beta: float = 0.02
    ):
        """initialization of simple DDPM calss

        Args:
            device (cuda device or CPU):
            n_steps (int): total sample steps of DDPM algo
            min_beta (float, optional): min of hyperpara beta. Defaults to 0.0001.
            max_beta (float, optional): max of hyperpara beta . Defaults to 0.02.
        """
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)  # linear betas
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.n_steps = n_steps
        self.device = device

    def sample_forward(self, x, t, eps=None):
        """forward sampling of DDPM according to x_t = sqrt(alpha_t) * x + sqrt(1 - alpha_t) * eps

        Args:
            x (tensor): original img
            t (int): timestep
            eps (noise, optional): the gaussian noise to be added to the img. Defaults to None.

        Returns:
            tensor: sample result at timestep t
        """
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        if eps is None:
            eps = torch.randn_like(x)
        res = eps * torch.sqrt(1 - alpha_bar) + x * torch.sqrt(alpha_bar)
        return res

    def sample_backward(self, img_shape, net, device, simple_var=False):
        """total backward sampling of DDPM

        Args:
            img_shape (tuple): the shape of the input img
            net (nn.Module): the network using to optimize the noise
            device (CUDA devices or CPU):
            simple_var (bool, optional): using simple var or original var. Defaults to True.

        Returns:
            tensor: x_0
        """
        x = torch.randn(img_shape).to(device)
        net = net.to(device)
        for t in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_step(x, t, net, simple_var)
        return x

    def sample_backward_step(self, x_t, t, net, simple_var=False):
        """backward sample at each timestep

        Args:
            x_t (tensor): input img
            t (int): current timestep
            net (nn.module): the network using to optimize the noise
            simple_var (bool, optional):using simple var or original var. Defaults to True.

        Returns:
            tensor: the result of backward sampling at timestep t
        """
        n = x_t.shape[0]
        t_tensor = torch.tensor([t] * n, dtype=torch.long).to(x_t.device).unsqueeze(1)
        # t_tensor = torch.full((n, 1), t, dtype=torch.long, device=x_t.device)
        eps = net(x_t, t_tensor)

        if t == 0:
            noise = 0
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (
                    (1 - self.alpha_bars[t - 1])
                    / (1 - self.alpha_bars[t])
                    * self.betas[t]
                )
            noise = torch.randn_like(x_t) * torch.sqrt(var)

        mean = (
            x_t - (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) * eps
        ) / torch.sqrt(self.alphas[t])

        x_t = mean + noise

        return x_t
