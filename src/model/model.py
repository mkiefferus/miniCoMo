import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DecoderBlock

class SimpleAutoencoder(nn.Module):
    def __init__(self, state_size=16, num_heads=2, encoding_dim=64):
        super().__init__()

        self.state_size = state_size
        self.num_heads = num_heads

        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(392, 256),  # 28*28 = 784
            nn.ReLU(),
            nn.Linear(256, encoding_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 392),
            nn.Sigmoid(), 
            nn.Unflatten(1, (1, 28, 14))
        )

        # ===== Attention Layers =====
        
        # self.token_proj  = nn.Linear(16, encoding_dim)
        self.state_embed = nn.Embedding(state_size, encoding_dim)

        # cross-attention stacks (state↔image)
        self.xattn_state = nn.ModuleList(
            DecoderBlock(encoding_dim, num_heads=4) for _ in range(num_heads)
        )
        self.xattn_img   = nn.ModuleList(
            DecoderBlock(encoding_dim, num_heads=4) for _ in range(num_heads)
        )

        self.norm_state = nn.LayerNorm(encoding_dim)


    def forward(self, x_half):
        # Encode
        img_enc = self.encoder(x_half).unsqueeze(1)  # B×1×d
        B, _, _ = img_enc.shape

        # Get state token
        idx = torch.arange(self.state_size, device=x_half.device)
        state  = self.state_embed(idx).unsqueeze(0).expand(B, -1, -1)  # B×S×d

        # Cross-attention
        output = [(state, img_enc)]
        for blk_s, blk_i in zip(self.xattn_state, self.xattn_img):
            state, _ = blk_s(*output[-1][::+1], None, None)
            img_enc, _ = blk_i(*output[-1][::-1], None, None)
            output.append((state, img_enc))

        state, img_enc = output[-1]
        
        # Decode
        pred = self.decoder(img_enc.squeeze(1))
        return pred