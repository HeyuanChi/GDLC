import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import get_device

device, kwargs = get_device()


class DDPM(nn.Module):
    def __init__(self, model: nn.Module, num_timesteps=1000, beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        self.model = model.to(device)
        self.num_timesteps = num_timesteps
        
        betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]], dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        
        return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise

    def p_losses(self, x0, t):
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise=noise)
        pred_noise = self.model(x_noisy, t)
        
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def p_sample(self, x, t):
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_alphas_t = torch.sqrt(self._extract(self.alphas, t, x.shape))

        eps_theta = self.model(x, t)

        model_mean = (1.0 / sqrt_alphas_t) * (
            x - betas_t / sqrt_one_minus_alphas_cumprod_t * eps_theta
        )

        if (t > 0).all():
            posterior_var_t = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            x_prev = model_mean + torch.sqrt(posterior_var_t) * noise
        else:
            x_prev = model_mean

        return x_prev

    @torch.no_grad()
    def sample(self, shape):
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.num_timesteps)):
            t = torch.tensor([i]*shape[0], device=device)
            x = self.p_sample(x, t)
        return x

    def _extract(self, arr, t, x_shape):
        bs = t.shape[0]
        out = arr.gather(-1, t)
        return out.reshape(bs, *((1,) * (len(x_shape) - 1)))
