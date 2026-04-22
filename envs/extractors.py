"""Custom feature extractors for Stable-Baselines 3.

The default ``CombinedExtractor`` gives too much weight to the CNN branch
(256 features by default), causing noisy image gradients to drown out the
vector signal and destroy the value function.

``FrozenCNNExtractor`` fixes this by:
1. Processing the vector branch through a small MLP (→ *vec_features* dims).
2. Processing images through a NatureCNN whose weights are **frozen**
   (``requires_grad=False``).  The CNN thus acts as a fixed random
   projection, contributing *cnn_output_dim* features without injecting
   harmful gradients.

When the policy is later fine-tuned (e.g. for Grasp / Pick-and-Place),
call ``model.policy.features_extractor.unfreeze_cnn()`` to let the CNN
start learning from more informative camera views.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN


class FrozenCNNExtractor(BaseFeaturesExtractor):
    """Vector-first combined extractor with a frozen CNN branch.

    Parameters
    ----------
    observation_space : gym.spaces.Dict
        Must contain a ``"vector"`` key and at least one image key.
    vec_features : int
        Output dimension of the vector MLP branch (default 64).
    cnn_output_dim : int
        Output dimension of the frozen CNN branch (default 32).
    freeze_cnn : bool
        If True (default), CNN weights have ``requires_grad=False``.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        vec_features: int = 64,
        cnn_output_dim: int = 32,
        freeze_cnn: bool = True,
    ):
        # We'll compute features_dim after building sub-networks
        super().__init__(observation_space, features_dim=1)  # placeholder

        extractors: dict[str, nn.Module] = {}
        total_dim = 0
        self._image_keys: list[str] = []

        for key, subspace in observation_space.spaces.items():
            if key == "vector":
                # Small MLP for the vector branch
                vec_size = int(subspace.shape[0])
                extractors[key] = nn.Sequential(
                    nn.Linear(vec_size, vec_features),
                    nn.ReLU(),
                )
                total_dim += vec_features
            else:
                self._image_keys.append(key)
                # Image branch — NatureCNN
                # NatureCNN expects a channels-first (C, H, W) space, but
                # our env provides (H, W, C).  Create a transposed space
                # for the constructor, then transpose in forward().
                h, w, c = subspace.shape
                chw_space = gym.spaces.Box(
                    low=0.0, high=1.0,
                    shape=(c, h, w),
                    dtype=np.float32,
                )
                cnn = NatureCNN(
                    chw_space,
                    features_dim=cnn_output_dim,
                    normalized_image=True,
                )
                if freeze_cnn:
                    for param in cnn.parameters():
                        param.requires_grad = False
                extractors[key] = cnn
                total_dim += cnn_output_dim

        self.extractors = nn.ModuleDict(extractors)
        # Override the placeholder
        self._features_dim = total_dim

    # ------------------------------------------------------------------
    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for key, extractor in self.extractors.items():
            obs = observations[key]
            # Transpose image observations from (B, H, W, C) → (B, C, H, W)
            if key in self._image_keys:
                obs = obs.permute(0, 3, 1, 2)
            parts.append(extractor(obs))
        return torch.cat(parts, dim=1)

    # ------------------------------------------------------------------
    def unfreeze_cnn(self) -> None:
        """Enable gradient computation for all CNN branches."""
        for key, extractor in self.extractors.items():
            if key != "vector":
                for param in extractor.parameters():
                    param.requires_grad = True

    def freeze_cnn(self) -> None:
        """Disable gradient computation for all CNN branches."""
        for key, extractor in self.extractors.items():
            if key != "vector":
                for param in extractor.parameters():
                    param.requires_grad = False
