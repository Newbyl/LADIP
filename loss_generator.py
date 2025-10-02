import torch
import numpy as np
import torch.nn.functional as F

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, model_kwargs=None, zs=None):
        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction
        model_output, pred_masked, mask_indices = model(model_input, time_input.flatten(), **model_kwargs)
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        # projection loss (masked patches only)
        if pred_masked is not None and mask_indices is not None and zs is not None and len(zs) > 0:
            bsz = zs[0].shape[0]
            num_masked = mask_indices.shape[1]
            # Reshape pred_masked from (bsz*N_masks, num_masked, D) to (bsz, num_masked, D)
            # In your case, N_masks=1 per sample, so pred_masked should have shape (bsz, num_masked, D)
            # But the predictor concatenates outputs, so we need to take only the first bsz samples
            pred_masked = pred_masked[:bsz]  # Take first batch
            
            # Initialize proj_loss with correct shape
            proj_loss = torch.zeros(bsz, device=images.device, dtype=images.dtype)
            
            # zs: list of tensors (N, T, D), pred_masked: (N, num_masked, D), mask_indices: (N, num_masked)
            for i, z in enumerate(zs):
                # Gather the masked patches from z using mask_indices
                # z: (N, T, D), mask_indices: (N, num_masked)
                z_masked = torch.gather(z, 1, mask_indices.unsqueeze(-1).expand(-1, -1, z.shape[-1]))
                pred_masked_norm = F.normalize(pred_masked, dim=-1)
                z_masked_norm = F.normalize(z_masked, dim=-1)
                # Negative cosine similarity loss (maximize alignment)
                # Loss goes from +1 (worst, orthogonal) to -1 (best, aligned)
                cosine_sim = (z_masked_norm * pred_masked_norm).sum(dim=-1)
                proj_loss = proj_loss + mean_flat(-cosine_sim)  # Loss: -1 (best) to +1 (worst)
            proj_loss = proj_loss / len(zs)
        else:
            # If no predictor output, ensure pred_masked still contributes to the graph
            # This prevents DDP errors about unused parameters
            proj_loss = torch.zeros(images.shape[0], device=images.device, dtype=images.dtype)
            if pred_masked is not None:
                proj_loss = proj_loss + 0.0 * pred_masked.sum()

        return denoising_loss, proj_loss
