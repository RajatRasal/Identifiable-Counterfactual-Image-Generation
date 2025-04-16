from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.slate.slate import SLATE
from models.slot_attention.model import SlotAttentionAutoEncoder


class SlotWrapper(ABC):
    @property
    @abstractmethod
    def slot_size(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_slots(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def recon(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def slot_mask(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def compose(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SlotAEWrapper(SlotWrapper):
    def __init__(self, model: SlotAttentionAutoEncoder):
        self.model = model
        self.model.eval()

    @property
    def slot_size(self) -> int:
        return self.model.hid_dim

    @property
    def n_slots(self) -> int:
        return self.model.num_slots

    @torch.no_grad
    def recon(self, image: torch.Tensor) -> torch.Tensor:
        recon, _, _, _ = self.model(image)
        return recon

    @torch.no_grad
    def slot_mask(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # `image` has shape: [batch_size, num_channels, width, height].
        B, C, W, H = image.shape
        # `slots` has shape: [batch_size, num_slots, slot_size].
        slots = self.model.encode(image)

        # Broadcast slot features to a 2D grid and collapse slot dimension.
        _slots = slots.reshape((-1, self.slot_size)).unsqueeze(1).unsqueeze(2)
        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        _slots = _slots.repeat((1, 8, 8, 1))

        # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].
        x = self.model.decoder_cnn(_slots)

        # Undo combination of slot and batch dimension; split alpha masks.
        # `masks` has shape: [batch_size, num_slots, width, height, 1].
        recons, masks = x.reshape(
            image.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]
        ).split([3, 1], dim=-1)

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        masks = masks * recons + (1 - masks)

        # Resize slots for interface
        slots = slots.view(-1, self.slot_size)
        masks = masks.view(-1, W, H, C).permute(0, 3, 1, 2)

        return slots, masks

    @torch.no_grad
    def compose(self, slots: torch.Tensor) -> torch.Tensor:
        # Broadcast slot features to a 2D grid and collapse slot dimension.
        slots = slots.reshape((-1, self.slot_size)).unsqueeze(1).unsqueeze(2)
        slots = slots.repeat((1, 8, 8, 1))

        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        x = self.model.decoder_cnn(slots)

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = x.reshape(
            x.shape[0] // self.num_slots,
            -1,
            x.shape[1],
            x.shape[2],
            x.shape[3],
        ).split([3, 1], dim=-1)

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        # Recombine image.
        recon_combined = torch.sum(recons * masks, dim=1)
        # `recon_combined` has shape: [batch_size, width, height, num_channels].
        recon_combined = recon_combined.permute(0, 3, 1, 2)
        return recon_combined


class SLATEWrapper(SlotWrapper):
    def __init__(self, model: SLATE, tau: float, hard: bool, image_size: int):
        self.model = model
        self.tau = tau
        self.hard = hard
        self.image_size = image_size
        self.model.eval()

    @property
    def n_slots(self) -> int:
        return self.model.num_slots

    @property
    def slot_size(self) -> int:
        return self.model.slot_size

    @torch.no_grad
    def recon(self, image: torch.Tensor) -> torch.Tensor:
        slots, _ = self.slot_mask(image)
        recon = self.compose(slots)
        return recon

    @torch.no_grad
    def slot_mask(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, H, W = image.shape
        _, slots, attns, _, _, _ = self.model.slots(image, self.tau, self.hard)
        slots = slots.view(-1, self.slot_size)
        attns = attns.view(-1, 3, H, W)
        return slots, attns

    @torch.no_grad
    def compose(self, slots: torch.Tensor) -> torch.Tensor:
        slots = slots.view(-1, self.n_slots, self.slot_size)
        downsample_dim = self.image_size // 4
        gen_len = downsample_dim**2

        z_gen = slots.new_zeros(0)
        z_transformer_input = slots.new_zeros(
            slots.shape[0], 1, self.model.vocab_size + 1
        )
        z_transformer_input[..., 0] = 1.0
        for t in range(gen_len):
            decoder_output = self.model.tf_dec(
                self.model.positional_encoder(
                    self.model.dictionary(z_transformer_input)
                ),
                slots,
            )
            z_next = F.one_hot(
                self.model.out(decoder_output)[:, -1:].argmax(dim=-1),
                self.model.vocab_size,
            )
            z_gen = torch.cat((z_gen, z_next), dim=1)
            z_transformer_input = torch.cat(
                [
                    z_transformer_input,
                    torch.cat([torch.zeros_like(z_next[:, :, :1]), z_next], dim=-1),
                ],
                dim=1,
            )

        z_gen = (
            z_gen.transpose(1, 2)
            .float()
            .reshape(slots.shape[0], -1, downsample_dim, downsample_dim)
        )
        recon_transformer = self.model.dvae.decoder(z_gen)

        return recon_transformer.clamp(0.0, 1.0)
