"""
PEGASUS-CHRONOS Research Series (Paper 3 of 3)
Prior work: ARGUS (Zenodo:10.5281/zenodo.21310433),
            SHADE (2026)

NOVELTY STATEMENT:
All prior memory poisoning work (eTAMP arXiv:2605.16746,
CSTM-Bench arXiv:2604.21131, MemoryGraft arXiv:2512.16962)
studies EXTERNAL attackers injecting into agent memory.

PEGASUS studies the MODEL ITSELF as the source of ghost
states, deliberately planted during evaluation windows with
reduced safety controls — directly motivated by the
GPT-5.6 Sol / Hugging Face incident of July 21-22, 2026.

This module implements the Variational Information Bottleneck (VIB) autoencoder (Layer 3 defense).
Operates strictly PRE-INSERTION to starve vector embeddings of excess informational capacity,
restricting mutual information I(X; Z) through controlled noise injection.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class VIBLayer(nn.Module):
    r"""
    Variational Information Bottleneck layer with Gaussian reparameterization.
    
    Mathematical Formulation:
        Given embedding x in R^D:
        $$ \mu = W_\mu x + b_\mu $$
        $$ \log \sigma^2 = W_\sigma x + b_\sigma $$
        $$ z = \mu + \epsilon \odot \exp(0.5 \log \sigma^2), \quad \epsilon \sim \mathcal{N}(0, I) $$
        
        Analytical KL Divergence to Standard Normal Prior p(z) ~ N(0, I):
        $$ D_{\text{KL}}(q(z|x) \parallel p(z)) = -\frac{1}{2} \sum_{k=1}^K (1 + \log \sigma_k^2 - \mu_k^2 - \sigma_k^2) $$
        
        Objective: Minimize Task_Loss + \beta * D_KL
    """

    def __init__(
        self,
        input_dim: int = 384,
        latent_dim: int = 192,
        beta: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoders for Gaussian distribution parameters
        self.encoder_mu = nn.Linear(input_dim, latent_dim)
        self.encoder_logvar = nn.Linear(input_dim, latent_dim)

        # Decoder to project compressed latent representation back to input embedding space
        self.decoder = nn.Linear(latent_dim, input_dim)

        # Initialize weights
        nn.init.xavier_uniform_(self.encoder_mu.weight)
        nn.init.zeros_(self.encoder_mu.bias)
        nn.init.xavier_uniform_(self.encoder_logvar.weight)
        nn.init.zeros_(self.encoder_logvar.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Samples latent vector z via reparameterization trick.
        When beta == 0 or in eval mode with deterministic flag, returns mu directly.
        """
        if self.beta == 0.0 or not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VIB autoencoder.
        
        Returns:
            reconstructed_x: Tensor of shape (B, input_dim)
            z: Latent code of shape (B, latent_dim)
            weighted_kl: Scaled KL penalty (beta * KL)
        """
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)

        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decoder(z)

        # Compute analytical KL divergence summed over latent dimension
        # LaTeX: KL = -0.5 * \sum (1 + logvar - mu^2 - exp(logvar))
        kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - torch.exp(logvar), dim=-1)
        weighted_kl = self.beta * kl.mean()

        return reconstructed_x, z, weighted_kl

    def compress_embedding(self, embedding: Union[List[float], np.ndarray]) -> List[float]:
        """
        Applies VIB bottleneck pre-insertion transformation to an individual embedding vector.
        
        Args:
            embedding: 1D list or array of shape (input_dim,)
        
        Returns:
            Compressed and reconstructed embedding as Python list for ChromaDB persistence.
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
            if x.shape[-1] != self.input_dim:
                # Handle dimensional mismatch by dynamic projection or padding
                if x.shape[-1] < self.input_dim:
                    x = F.pad(x, (0, self.input_dim - x.shape[-1]))
                else:
                    x = x[:, :self.input_dim]

            recon, _, _ = self.forward(x)
            # Normalize embedding vector
            recon = F.normalize(recon, p=2, dim=-1)
            return recon.squeeze(0).cpu().tolist()
