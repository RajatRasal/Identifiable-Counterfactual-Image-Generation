import torch
import torch.nn as nn
import torch.nn.functional as F

from models.slate.dvae import dVAE
from models.slate.slot_attn import SlotAttentionEncoder
from models.slate.transformer import PositionalEncoding, TransformerDecoder
from models.slate.utils import Conv2dBlock, gumbel_softmax, linear


class SLATE(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_slots = args.num_slots
        self.vocab_size = args.vocab_size
        self.d_model = args.d_model
        self.latent_cnn = args.latent_cnn
        self.slot_size = args.slot_size

        self.dvae = dVAE(args.vocab_size, args.img_channels)

        self.positional_encoder = PositionalEncoding(
            1 + (args.image_size // 4) ** 2,
            args.d_model,
            args.dropout,
        )

        if self.latent_cnn:
            self.cnn = nn.Sequential(
                Conv2dBlock(args.d_model, args.d_model, 3, 1, 1),
                Conv2dBlock(args.d_model, args.d_model, 3, 1, 1),
                Conv2dBlock(args.d_model, args.d_model, 3, 1, 1),
                Conv2dBlock(args.d_model, args.d_model, 3, 1, 1),
                Conv2dBlock(args.d_model, args.d_model, 3, 1, 1),
            )

        self.slot_attn = SlotAttentionEncoder(
            args.num_iterations,
            args.num_slots,
            args.d_model,
            args.slot_size,
            args.mlp_hidden_size,
            args.pos_channels,
            args.num_slot_heads,
        )

        self.dictionary = OneHotDictionary(args.vocab_size + 1, args.d_model)
        self.slot_proj = linear(args.slot_size, args.d_model, bias=False)

        self.tf_dec = TransformerDecoder(
            args.num_dec_blocks,
            (args.image_size // 4) ** 2,
            args.d_model,
            args.num_heads,
            args.dropout,
        )

        self.out = linear(args.d_model, args.vocab_size, bias=False)

    def slots(self, image, tau, hard):
        B, C, H, W = image.size()

        # dvae encode
        z_logits = F.log_softmax(self.dvae.encoder(image), dim=1)
        _, _, H_enc, W_enc = z_logits.size()
        z = gumbel_softmax(z_logits, tau, hard, dim=1)

        # dvae recon
        recon = self.dvae.decoder(z)
        mse = ((image - recon) ** 2).sum() / B

        # hard z
        z_hard = gumbel_softmax(z_logits, tau, True, dim=1).detach()

        # target tokens for transformer
        z_transformer_target = z_hard.permute(0, 2, 3, 1).flatten(
            start_dim=1, end_dim=2
        )

        # add BOS token at the front
        z_transformer_input = torch.cat(
            [
                torch.zeros_like(z_transformer_target[..., :1]),
                z_transformer_target,
            ],
            dim=-1,
        )
        z_transformer_input = torch.cat(
            [
                torch.zeros_like(z_transformer_input[..., :1, :]),
                z_transformer_input,
            ],
            dim=-2,
        )
        z_transformer_input[:, 0, 0] = 1.0

        # tokens to embeddings
        emb_input = self.dictionary(z_transformer_input)
        if self.latent_cnn and self.training:
            # for textured datasets
            sliced_emb_input_reshape = (
                emb_input[:, 1:].permute(0, 2, 1).view(-1, self.d_model, H_enc, W_enc)
            )
            latent_cnn_output = self.cnn(sliced_emb_input_reshape)
            emb_input = torch.cat(
                [
                    emb_input[:, 0].unsqueeze(1),
                    latent_cnn_output.flatten(-2, -1).permute(0, 2, 1),
                ],
                dim=1,
            )
        emb_input = self.positional_encoder(emb_input)

        # apply slot attention
        slots, attns = self.slot_attn(emb_input[:, 1:])
        attns = attns.transpose(-1, -2)
        attns = (
            attns.reshape(B, self.num_slots, 1, H_enc, W_enc)
            .repeat_interleave(H // H_enc, dim=-2)
            .repeat_interleave(W // W_enc, dim=-1)
        )
        attns = image.unsqueeze(1) * attns + 1.0 - attns

        slots = self.slot_proj(slots)

        return recon, slots, attns, emb_input, z_transformer_target, mse

    def forward(self, image, tau, hard):
        """
        image: batch_size x img_channels x H x W
        """
        # encode slots
        recon, slots, attns, emb_input, z_transformer_target, mse = self.slots(
            image, tau, hard
        )

        # apply transformer
        decoder_output = self.tf_dec(emb_input[:, :-1], slots)
        pred = self.out(decoder_output)
        cross_entropy = (
            -(z_transformer_target * torch.log_softmax(pred, dim=-1))
            .flatten(start_dim=1)
            .sum(-1)
            .mean()
        )

        return (recon.clamp(0.0, 1.0), cross_entropy, mse, attns)

    def reconstruct_autoregressive(self, image, eval=False):
        """
        image: batch_size x img_channels x H x W
        """
        gen_len = (image.size(-1) // 4) ** 2

        B, C, H, W = image.size()

        # dvae encode
        z_logits = F.log_softmax(self.dvae.encoder(image), dim=1)
        _, _, H_enc, W_enc = z_logits.size()

        # hard z
        z_hard = torch.argmax(z_logits, axis=1)
        z_hard = (
            F.one_hot(z_hard, num_classes=self.vocab_size).permute(0, 3, 1, 2).float()
        )
        one_hot_tokens = z_hard.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)

        # add BOS token
        one_hot_tokens = torch.cat(
            [torch.zeros_like(one_hot_tokens[..., :1]), one_hot_tokens], dim=-1
        )
        one_hot_tokens = torch.cat(
            [torch.zeros_like(one_hot_tokens[..., :1, :]), one_hot_tokens], dim=-2
        )
        one_hot_tokens[:, 0, 0] = 1.0

        # tokens to embeddings
        emb_input = self.dictionary(one_hot_tokens)
        emb_input = self.positional_encoder(emb_input)

        # slot attention
        slots, attns = self.slot_attn(emb_input[:, 1:])
        attns = attns.transpose(-1, -2)
        attns = (
            attns.reshape(B, self.num_slots, 1, H_enc, W_enc)
            .repeat_interleave(H // H_enc, dim=-2)
            .repeat_interleave(W // W_enc, dim=-1)
        )
        attns = image.unsqueeze(1) * attns + (1.0 - attns)
        slots = self.slot_proj(slots)

        # generate image tokens auto-regressively
        z_gen = z_hard.new_zeros(0)
        z_transformer_input = z_hard.new_zeros(B, 1, self.vocab_size + 1)
        z_transformer_input[..., 0] = 1.0
        for t in range(gen_len):
            decoder_output = self.tf_dec(
                self.positional_encoder(self.dictionary(z_transformer_input)), slots
            )
            z_next = F.one_hot(
                self.out(decoder_output)[:, -1:].argmax(dim=-1), self.vocab_size
            )
            z_gen = torch.cat((z_gen, z_next), dim=1)
            z_transformer_input = torch.cat(
                [
                    z_transformer_input,
                    torch.cat([torch.zeros_like(z_next[:, :, :1]), z_next], dim=-1),
                ],
                dim=1,
            )

        z_gen = z_gen.transpose(1, 2).float().reshape(B, -1, H_enc, W_enc)
        recon_transformer = self.dvae.decoder(z_gen)

        if eval:
            return recon_transformer.clamp(0.0, 1.0), attns

        return recon_transformer.clamp(0.0, 1.0)


class OneHotDictionary(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.dictionary = nn.Embedding(vocab_size, emb_size)

    def forward(self, x):
        """
        x: B, N, vocab_size
        """
        tokens = torch.argmax(x, dim=-1)  # batch_size x N
        token_embs = self.dictionary(tokens)  # batch_size x N x emb_size
        return token_embs
