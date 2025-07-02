import torch
import torch.nn as nn
import torch.nn.functional as F

from math import prod

from .blocks import DecoderBlock
from .message import CollaborativeMessage

class CollabConfig:
    def __init__(self, 
                 state_size=16, 
                 num_heads=2, 
                 encoding_dim=64,
                 collab_n_agents=2,         # Number of collaborating agents
                 collab_message_size=32,    # Size of the collaborative message
                 collab_dec_depth=2):       # Depth of the collaborative decoder
        
        self.state_size = state_size
        self.num_heads = num_heads
        self.encoding_dim = encoding_dim
        self.collab_n_agents = collab_n_agents
        self.collab_message_size = collab_message_size
        self.collab_dec_depth = collab_dec_depth

class SimpleAutoencoder(nn.Module):
    def __init__(self, 
                 config: CollabConfig,
                 input_dim=(28,14)): # Half of MNIST image size (28x28)
        super().__init__()

        self.state_size = config.state_size
        self.encoding_dim = config.encoding_dim
        self.num_heads = config.num_heads
        self.input_dim = input_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prod(self.input_dim), 256),  # 28*28 = 784
            nn.ReLU(),
            nn.Linear(256, self.encoding_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.encoding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, prod(self.input_dim)),
            nn.Sigmoid(), 
            nn.Unflatten(1, (1, *input_dim))
        )

        # ===== Attention Layers =====
        
        # self.token_proj  = nn.Linear(16, encoding_dim)
        self.state_embed = nn.Embedding(self.state_size, self.encoding_dim)

        # cross-attention stacks (state↔image)
        self.xattn_state = nn.ModuleList(
            DecoderBlock(self.encoding_dim, num_heads=4) for _ in range(self.num_heads)
        )
        self.xattn_img   = nn.ModuleList(
            DecoderBlock(self.encoding_dim, num_heads=4) for _ in range(self.num_heads)
        )

        self.norm_state = nn.LayerNorm(self.encoding_dim)


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
    
    # def encoder(self, x):
    #     pass

    # def decoder(self, x):
    #     pass

class CollaborativeAutoencoder(nn.Module):
    def __init__(self, 
                 config: CollabConfig):
        super().__init__()

        pass