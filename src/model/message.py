import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DecoderBlock

import torch
import torch.nn as nn


class SimpleCollaborativeMessage(nn.Module):
    def __init__(self, 
                 state_size,
                 message_size,
                 norm_layer=nn.LayerNorm,
    ) -> None: 
        
        super().__init__()

        self.encoder_proj = nn.Linear(state_size, message_size)
        self.decoder_proj = nn.Linear(message_size, state_size)
        self.norm = norm_layer(state_size)

    def encode(self, state, ext_features=None):
        mean_state = state.mean(dim=1)
        message = self.encoder_proj(mean_state)
        return message.unsqueeze(1)

    def decode(self, state, ext_features, training:bool=False):
        message = ext_features.squeeze(1).squeeze(1)
        update_vector = self.decoder_proj(message)
        update_vector = update_vector.unsqueeze(1)
        
        updated_state = state + update_vector
        updated_state = self.norm(updated_state)

        return updated_state, ext_features

    
class CollaborativeMessage(nn.Module):
    """
    Collaborative message module.
    """
    def __init__(self, 
                 state_size,
                 message_size,
                 n_agents,
                 num_heads,
                 depth,
                 mlp_ratio,
                 qkv_bias=False,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0,
                 drop_message=0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 norm_message=True,
    ) -> None: 
        
        super().__init__()

        self.state_size = state_size
        self.n_agents = n_agents
        self.message_size = message_size

        
        # ===== Encoder Specific =====
        
        self.query_token = nn.Parameter(
            torch.randn(1, 1, message_size) * 0.2, requires_grad=True
        )

        self.enc_dec_to_out = nn.Linear(state_size, message_size)
        self.enc_in_to_dec = nn.Linear(message_size, state_size)

        # ===== Decoder Specific =====

        # Message placeholder
        self.message_placeholder = nn.Parameter(
            torch.randn(1, n_agents-1, 1, message_size) * 0.2, requires_grad=True
        )

        self.dec_emb_proj = nn.Linear(message_size, state_size)
        
        # Drop Messages
        self.drop_messages = drop_message

        # ===== Shared Parameters =====

        self.enc_blocks = nn.ModuleList(
            [
                DecoderBlock( # dust3r decoder block differs from croco decoder block (v not projected in croco forward pass)
                    dim=state_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_message,
                    rope=None,
                )
                for _ in range(depth)
            ]
        )

        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dim=state_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_message,
                    rope=None,
                )
                for _ in range(depth)
            ]
        )

    def _drop_messages(self, messages, training:bool=False):
        """
        Drop messages during training.
        """
        
        if self.drop_messages == 0.0 or not training:
            return messages
        
        device = messages.device
        chunks = messages.chunk(self.n_agents-1, dim=1)

        # Drop messages
        drop_mask = torch.rand(self.n_agents-1, device=device) < self.drop_messages

        # Fill dropped messages with zeros
        new_messages = [torch.zeros_like(chunk) if drop else chunk for chunk, drop in zip(chunks, drop_mask)]

        return torch.cat(new_messages, dim=1)

    def encode(self, state, ext_features):
        """
        Extract essentials from state for collaboration.
        """
        # Use learned query token to extract information from state
        x = self.query_token.expand(state.shape[0], -1, -1)

        # Use external_features as query
        if ext_features is not None:
            ext_features = ext_features.view(ext_features.shape[0], -1, ext_features.shape[3])
            x = torch.cat([x, ext_features], dim=1).mean(dim=1, keepdim=True)

        x = self.enc_in_to_dec(x) # Project to state size

        # Read-out state
        for blk in self.enc_blocks:
            x, _ = blk(x, state, None, None)

        # Project to message size
        x = self.enc_dec_to_out(x)

        return x

    def decode(self, state, ext_features, training:bool=False):
        """
        Update state with collaboration message.
        """

        # Perform dropout
        x = self._drop_messages(ext_features, training)

        ph = self.message_placeholder.expand(x.shape[0], -1, -1, -1)

        # Fill x with zeros to match placeholder (only dim 1, i.e. number of agents)
        if x.shape[1] < ph.shape[1]:
            padding_needed_dim1 = ph.shape[1] - x.shape[1]
            x = F.pad(x, (0, 0, 0, padding_needed_dim1), value=0.0)

        x = torch.add(x, ph)

        x_proj = self.dec_emb_proj(x) # Project to state size

        # Write to state
        for blk in self.dec_blocks:
            state, _ = blk(state, x_proj, None, None)

        return state, x