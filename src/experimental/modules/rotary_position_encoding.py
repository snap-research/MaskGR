import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb
class RotaryPositionEncoding(nn.Module):
    """
    Rotary Position Encoding (RoPE) implementation.
    
    This implementation follows the approach from "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    (Su et al., 2021) and is widely used in modern transformer architectures.
    
    RoPE applies rotation matrices to the query and key vectors based on their positions,
    allowing the model to naturally encode relative position information.
    """
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        """
        Initialize Rotary Position Encoding.
        
        Args:
            dim: Dimension of the embeddings (must be even)
            max_position_embeddings: Maximum sequence length
            base: Base for the frequency computation
        """
        super().__init__()
        
        if dim % 2 != 0:
            raise ValueError(f"Rotary position encoding requires even dimension, got {dim}")
        
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
                                      
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
                                         
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len: int):
        """Pre-compute cos and sin embeddings for efficiency."""
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=True)
        self.register_buffer("sin_cached", emb.sin(), persistent=True)
    
    def forward(self, x: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        """
        Apply rotary position encoding to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, nhead, seq_len, head_dim)
            seq_len: Sequence length (if None, uses the second dimension of x)
            
        Returns:
            Tensor with rotary position encoding applied
        """
        if seq_len is None:
            seq_len = x.shape[2]                                                      
        
        if seq_len > self.max_position_embeddings:
            self._set_cos_sin_cache(seq_len)
        elif not hasattr(self, 'cos_cached') or self.cos_cached.size(0) < seq_len:
                                                                
            self._set_cos_sin_cache(max(seq_len, self.max_position_embeddings))
                                           
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
                                                                                         
        cos = cos.view(1, 1, seq_len, -1)
        sin = sin.view(1, 1, seq_len, -1)
         
        x_rope = self._apply_rotary_pos_emb(x, cos, sin)
        
        return x_rope
    
    def _apply_rotary_pos_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position embedding to the input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, nhead, seq_len, head_dim)
            cos: Cosine embeddings of shape (1, 1, seq_len, head_dim)
            sin: Sine embeddings of shape (1, 1, seq_len, head_dim)
            
        Returns:
            Tensor with rotary position encoding applied
        """                                              
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        
        cos_even = cos[..., ::2]
        sin_even = sin[..., ::2]

        rotated_even = x_even * cos_even - x_odd * sin_even
        rotated_odd = x_even * sin_even + x_odd * cos_even
        
        x_rope = torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)
        return x_rope


class RotaryMultiheadAttention(nn.Module):
    """
    Multi-head attention with Rotary Position Encoding (RoPE).
    
    This implementation applies RoPE to query and key vectors before computing attention scores,
    allowing the model to naturally encode relative position information.
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, batch_first: bool = False, **kwargs):
        """
        Initialize Rotary Multi-head Attention.
        
        Args:
            d_model: Dimension of the model
            nhead: Number of attention heads
            dropout: Dropout rate
            batch_first: Whether input is batch-first
            **kwargs: Additional arguments (ignored for compatibility)
        """
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.batch_first = batch_first
        
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"
        assert self.head_dim % 2 == 0, "head_dim must be even for rotary position encoding"
                                                  
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
                 
        self.dropout = nn.Dropout(dropout)
                        
        self.scale = math.sqrt(self.head_dim)
        self.rope = RotaryPositionEncoding(self.head_dim)
        

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: torch.Tensor = None, key_padding_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with rotary position encoding.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            attn_mask: Optional attention mask
            key_padding_mask: Optional key padding mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        if self.batch_first:
                                                          
            batch_size, seq_len, d_model = query.shape
                                                                    
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
                                                             
        else:
                                                             
            seq_len, batch_size, d_model = query.shape
        
        q = self.q_proj(query).view(seq_len, batch_size, self.nhead, self.head_dim).transpose(0, 1).transpose(1, 2)
        k = self.k_proj(key).view(seq_len, batch_size, self.nhead, self.head_dim).transpose(0, 1).transpose(1, 2)
        v = self.v_proj(value).view(seq_len, batch_size, self.nhead, self.head_dim).transpose(0, 1).transpose(1, 2)
        
        q = self.rope(q, seq_len)
        k = self.rope(k, seq_len)
                                  
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        if key_padding_mask is not None:
            if key_padding_mask.dim() == 2:
                                                                      
                key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            elif key_padding_mask.dim() == 4:
                                                               
                pass
            else:
                raise ValueError(f"key_padding_mask should have 2 or 4 dimensions, got {key_padding_mask.dim()}")
            
            if key_padding_mask.size(-1) != seq_len:
                if key_padding_mask.size(-1) > seq_len:
                    key_padding_mask = key_padding_mask[..., :seq_len]
                else:
                    pad_size = seq_len - key_padding_mask.size(-1)
                    key_padding_mask = F.pad(key_padding_mask, (0, pad_size), value=True)
            
                                                                                          
            key_padding_mask = key_padding_mask.expand(batch_size, self.nhead, seq_len, seq_len)
            scores = scores.masked_fill(key_padding_mask, float('-inf'))
        
                                                
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
                                           
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.out_proj(output)
        
        if self.batch_first:
            pass
        else:
            output = output.transpose(0, 1)
        
        return output, attn_weights



class RotaryTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with Rotary Position Encoding (RoPE).
    """
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, 
                 dropout: float = 0.1, activation: str = "relu", batch_first: bool = False, 
                 norm_first: bool = False, **kwargs):
        """
        Initialize Rotary Transformer Encoder Layer.
        
        Args:
            d_model: Dimension of the model
            nhead: Number of attention heads
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
            batch_first: Whether input is batch-first
            norm_first: Whether to use pre-norm architecture
            **kwargs: Additional arguments (ignored for compatibility)
        """
        super().__init__()
        
                              
        self.self_attn = RotaryMultiheadAttention(
            d_model, nhead, dropout, batch_first
        )
                             
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
                             
        self.activation = F.relu if activation == "relu" else F.gelu
                       
        self.norm_first = norm_first
        

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with rotary position encoding.
        """
        if self.norm_first:
                                   
            src2 = self.norm1(src)
            src2, _ = self.self_attn(
                src2, src2, src2,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask
            )
            src = src + self.dropout(src2)
            
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout(src2)
        else:
                                    
            src2, _ = self.self_attn(
                src, src, src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask
            )
            src = src + self.dropout(src2)
            src = self.norm1(src)
            
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout(src2)
            src = self.norm2(src)
        
        return src


class RotaryTransformerEncoder(nn.Module):
    """
    Transformer encoder with Rotary Position Encoding (RoPE).
    """
    
    def __init__(self, encoder_layer: RotaryTransformerEncoderLayer = None, num_layers: int = 6, norm: nn.Module = None,
                 d_model: int = None, nhead: int = None, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu", batch_first: bool = False,
                 norm_first: bool = False):
        """
        Initialize Rotary Transformer Encoder.
        
        Args:
            encoder_layer: Rotary transformer encoder layer (if None, will be created from parameters)
            num_layers: Number of layers
            norm: Final layer normalization
            d_model: Dimension of the model (used if encoder_layer is None)
            nhead: Number of attention heads (used if encoder_layer is None)
            dim_feedforward: Dimension of the feedforward network (used if encoder_layer is None)
            dropout: Dropout rate (used if encoder_layer is None)
            activation: Activation function ('relu' or 'gelu') (used if encoder_layer is None)
            batch_first: Whether input is batch-first (used if encoder_layer is None)
            norm_first: Whether to use pre-norm architecture (used if encoder_layer is None)
        """
        super().__init__()
        
        if encoder_layer is None:
            if d_model is None or nhead is None:
                raise ValueError("Either encoder_layer must be provided or both d_model and nhead must be provided")
            encoder_layer = RotaryTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=batch_first,
                norm_first=norm_first
            )
        
                                                                    
        self.layers = nn.ModuleList([
            RotaryTransformerEncoderLayer(
                d_model=encoder_layer.self_attn.d_model,
                nhead=encoder_layer.self_attn.nhead,
                dim_feedforward=encoder_layer.linear1.out_features,
                dropout=encoder_layer.dropout.p,
                activation="relu" if encoder_layer.activation == F.relu else "gelu",
                batch_first=encoder_layer.self_attn.batch_first,
                norm_first=encoder_layer.norm_first
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(
        self, src: torch.Tensor, mask: torch.Tensor = None, 
        src_key_padding_mask: torch.Tensor = None, 
        give_half_way_embedding: bool = False, extract_lyr = 4) -> torch.Tensor:
        """
        Forward pass with rotary position encoding.
        """
        output = src
        if extract_lyr is None and give_half_way_embedding: 
            extract_lyr = self.num_layers // 2
        for i, layer in enumerate(self.layers):
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask
            )
            if give_half_way_embedding and i == extract_lyr - 1:
                half_way_embedding = output
        
        if self.norm is not None:
            output = self.norm(output)

        if give_half_way_embedding:
            return output, half_way_embedding
        else:   
            return output