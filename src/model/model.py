import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from math import prod

from .blocks import DecoderBlock
from .message import CollaborativeMessage, SimpleCollaborativeMessage

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

class SimpleModel(nn.Module):
    def save(self, path):
        torch.save(
            dict(
                state_dict=self.state_dict(), 
                
            ),
            path,
        )

class SimpleAutoencoder(nn.Module):
    def __init__(self, 
                 config: CollabConfig,
                 input_dim=(28,14)):
        super().__init__()

        self.state_size = config.state_size
        self.encoding_dim = config.encoding_dim
        self.num_heads = config.num_heads
        self.input_dim = input_dim

        # Encoder
        self._encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prod(self.input_dim), 256),  # 28*28 = 784
            nn.ReLU(),
            nn.Linear(256, self.encoding_dim)
        )

        self._decoder = nn.Sequential(
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

    
    def _cross_attention(self, state, x):
        # Cross-attention
        output = [(state, x)]
        for blk_s, blk_i in zip(self.xattn_state, self.xattn_img):
            state, _ = blk_s(*output[-1][::+1], None, None)
            x, _ = blk_i(*output[-1][::-1], None, None)
            output.append((state, x))

        state, x = output[-1]
        return state, x

    
    def encoder(self, x):
        # Encode
        img_enc = self._encoder(x).unsqueeze(1)  # B×1×d
        B, _, _ = img_enc.shape

        # Get state token
        idx = torch.arange(self.state_size, device=x.device)
        state  = self.state_embed(idx).unsqueeze(0).expand(B, -1, -1)  # B×S×d
        
        state, img_enc = self._cross_attention(state=state, x=img_enc)
        return state, img_enc
        

    def decoder(self, x, state):
        state, x = self._cross_attention(state, x)
        pred = self._decoder(x.squeeze(1))
        return pred
    
    def forward(self, x):
        state, img_enc = self.encoder(x)
        pred = self.decoder(img_enc, state)
        return pred

        

class CollaborativeAutoencoder(nn.Module):
    def __init__(self, 
                 config: CollabConfig):
        super().__init__()

        self.collab_n_agents = config.collab_n_agents
        assert self.collab_n_agents <= 2, "Only 2 agents are supported for now."
        
        self.multi = self.collab_n_agents > 1
        self.collab_message_size = config.collab_message_size
        self.collab_dec_depth = config.collab_dec_depth

        if self.multi:
            self.collab_modules = nn.ModuleList([
                CollaborativeMessage(
                    state_size=config.encoding_dim,
                    message_size=config.collab_message_size,
                    n_agents=config.collab_n_agents,
                    num_heads=config.num_heads,
                    depth=config.collab_dec_depth,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    drop_message=0.0,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                )
                for _ in range(self.collab_n_agents)
            ])
        else: self.collab_modules = [None]

        input_dim = (28, 14)
        self.agents = nn.ModuleList([
            SimpleAutoencoder(
                config, 
                input_dim=input_dim
            ) 
            for _ in range(self.collab_n_agents)
        ])

    def forward(self, x):

        agent_state = []
        agent_img_enc = []
        agent_messages = []
        for id, (agent, collab_mod) in enumerate(zip(self.agents, self.collab_modules)):
            img = x[id]
            state, img_enc = agent.encoder(img)

            if self.multi:
                norm_state = agent.norm_state(state)
                ext_features = collab_mod.encode(norm_state, None)
                agent_messages.append(ext_features)
            else:
                agent_messages.append(None)
            
            agent_state.append(state)
            agent_img_enc.append(img_enc)

        preds = []
        for agent, collab_mod, state, img_enc, message in zip(
            self.agents,
            self.collab_modules,
            agent_state, 
            agent_img_enc, 
            agent_messages[::-1],
        ):

            if self.multi:
                state, _ = collab_mod.decode(state, message, training=self.training)

            pred = agent.decoder(img_enc, state)
            preds.append(pred)

        if self.multi:
            return preds[::-1]
        return preds