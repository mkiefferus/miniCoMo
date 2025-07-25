import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import copy

from pathlib import Path
from omegaconf import OmegaConf

from math import prod

from .blocks import DecoderBlock
from .collab import (
    CollabConfig,
    CollaborativeMessage, 
    SimpleCollaborativeMessage,
)

def _load_model(path:Path, device='cpu'):
    """Load a model from a given path."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    
    # Epoch already trained
    epoch = ckpt["epoch"]
    
    # Load model arguments
    args = ckpt["args"]
    OmegaConf.resolve(args)
    net = eval(args.model).to(device)
    net.load_state_dict(ckpt["state_dict"], strict=False)

    del ckpt # Free memory
    return net, args, epoch

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.best = dict(
            model=None,
            score=None,
            epoch=0,
        )

    def checkpoint(self):
        pass

    def save(self, train_config, epoch:int=0, model_name="model", save_best=False):
        out_path = Path(train_config.out_dir) / f"{model_name}.pth"
        torch.save(
            dict(
                state_dict=self.state_dict() if not save_best else self.best['model'], 
                args=train_config,
                epoch=epoch if not save_best else self.best['epoch'],
            ),
            out_path,
        )
        print(f">> Saving model to {out_path} ...")

    def check_best(self, score, epoch=0):
        self.is_best(score, epoch)

    def is_best(self, score, epoch=0):
        is_best = False
        if self.best['score'] is None or score > self.best['score']:
            self.best['model'] = copy.deepcopy(self.state_dict())
            self.best['score'] = score
            self.best['epoch'] = epoch
            print(f">> New best model found with score {score:.4f} at epoch {epoch}.")

            is_best = True
        return is_best

    def load_best(self):
        if self.best['model'] is not None:
            self.load_state_dict(self.best['model'])
            print(f">> Loaded best model with score {self.best['score']:.4f} at epoch {self.best['epoch']}.")
        else:
            print("No best model found.")


class SimpleAutoencoder(BaseModel):
    def __init__(self, 
                 config: CollabConfig):
        
        self.config = config
        
        super().__init__()

        self.state_size = config.state_size
        self.encoding_dim = config.encoding_dim
        self.num_heads = config.num_heads
        self.input_dim = config.img_size

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
            nn.Unflatten(1, (1, *config.img_size))  # Unflatten to (B, 1, 28, 14)
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

        

class CollaborativeAutoencoder(BaseModel):
    def __init__(self, 
                 config: CollabConfig):
        
        self.config = config
        super().__init__()

        self.collab_n_agents = config.collab_n_agents
        assert self.collab_n_agents <= 2, "Only 2 agents are supported for now."
        
        self.multi = self.collab_n_agents > 1
        self.collab_message_size = config.collab_message_size
        self.collab_dec_depth = config.collab_dec_depth

        if self.multi:
            self.collab_modules = nn.ModuleList([
                self._get_messenger()
                for _ in range(self.collab_n_agents)
            ])
        else: self.collab_modules = [None]

        self.agents = nn.ModuleList([
            SimpleAutoencoder(
                config
            ) 
            for _ in range(self.collab_n_agents)
        ])

    def _get_messenger(self):
        """Build messenger module based on config"""
        if self.config.collab_messenger == "CollaborativeMessage":
            kwargs = dict(
                n_agents=self.config.collab_n_agents,
                num_heads=self.config.num_heads,
                depth=self.config.collab_dec_depth,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_message=0.0,
            )
        else:
            kwargs = {}

        messenger = eval(self.config.collab_messenger)(
            state_size=self.config.encoding_dim,
            message_size=self.config.collab_message_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs
        )
        return messenger
    
    @classmethod
    def from_pretrained(cls, path, device='cpu'):
        if Path(path).is_file():
            return _load_model(path, device=device)
        else:
            raise FileNotFoundError(f"Model path {path} does not exist.")
    
    def save_single_agent(self, train_config, epoch: int = 0):
        """Save first collab agent."""
        single_agent_state_dict = {}
        
        for key, value in self.state_dict().items():
            if key.startswith("agents.0."):
                new_key = "agent." + key[len("agents.0."):]
                single_agent_state_dict[new_key] = value
            elif self.multi and key.startswith("collab_modules.0."):
                new_key = "collab_module." + key[len("collab_modules.0."):]
                single_agent_state_dict[new_key] = value

        out_path = Path(train_config.out_dir) / f"single_agent.pth"
        torch.save(
            dict(
                state_dict=single_agent_state_dict, 
                args=train_config,
                epoch=epoch,
            ),
            out_path,
        )
        print(f">> Saving single-agent model to {out_path} ...")

    @classmethod
    def load_single_agent(cls, path: str, cfg, device='cpu'):
        """Load single agent model and broadcast weights to all agents."""
        # Load model
        print(cfg)
        model = eval(cfg.model)

        # Load single agent checkpoint
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        single_agent_state_dict = ckpt['state_dict']
        current_epoch = ckpt.get('epoch', 0)
        
        # Build new state dict by broadcasting the weights
        broadcast_state_dict = {}
        multi_agent = cfg.collab_n_agents > 1

        for key, value in single_agent_state_dict.items():
            if key.startswith("agent."):
                base_key = key[len("agent."):]
                for i in range(cfg.collab_n_agents):
                    new_key = f"agents.{i}.{base_key}"
                    broadcast_state_dict[new_key] = value
            elif multi_agent and key.startswith("collab_module."):
                base_key = key[len("collab_module."):]
                for i in range(cfg.collab_n_agents):
                    new_key = f"collab_modules.{i}.{base_key}"
                    broadcast_state_dict[new_key] = value
        
        # Load broadcasted state dict into model
        model.load_state_dict(broadcast_state_dict, strict=False)
        print(f"Successfully created model and broadcasted weights from {path}.")

        del ckpt  # Free memory
        return model.to(device), None, current_epoch
        
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