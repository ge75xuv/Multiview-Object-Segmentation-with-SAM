import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.encoding = self._get_positional_encoding(max_len, d_model)
        self.register_buffer('pe', self.encoding)  # not a learnable param

    def _get_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # shape: (1, max_len, d_model)

class QueryMemoryFusion(nn.Module):
    def __init__(self, num_queries, embed_dim, hidden_dim=128, fusion_type='early', learn_query_embed=True):
        """
        fusion_type: 'early' or 'late' — determines how query identity is fused
        """
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.fusion_type = fusion_type
        self.learn_query_embed = learn_query_embed

        # Learnable positional or class embeddings per query
        if learn_query_embed:
            self.query_embed = nn.Embedding(num_queries, embed_dim)
        else:
            self.query_embed = SinusoidalPositionalEncoding(max_len=num_queries, d_model=embed_dim)

        if fusion_type == 'late':
            # If concatenating, double the input dimension
            input_dim = embed_dim * 2
        else:
            input_dim = embed_dim

        # Scorer MLP
        self.score_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, memory_tokens):
        """
        memory_tokens: (Q, L, E)
        Returns: fused_tokens: (1, L, E)
        """
        Q, L, E = memory_tokens.shape
        assert Q == self.num_queries and E == self.embed_dim

        # (Q, E) → (Q, 1, E) → (Q, L, E)
        if self.learn_query_embed:
            query_embed = self.query_embed(torch.arange(Q, device=memory_tokens.device))
            query_embed = query_embed.unsqueeze(1).expand(Q, L, E)
        else:
            query_embed = self.query_embed.pe.transpose(1, 0)

        if self.fusion_type == 'early':
            fused_input = memory_tokens + query_embed  # Add class identity directly
        elif self.fusion_type == 'late':
            fused_input = torch.cat([memory_tokens, query_embed], dim=-1)  # Learn how to combine
        else:
            fused_input = memory_tokens

        scores = self.score_mlp(fused_input)  # (Q, L, 1)
        attn_weights = F.softmax(scores, dim=0)  # Softmax over Q

        fused_tokens = (memory_tokens * attn_weights).sum(dim=0, keepdim=True)  # (1, L, E)

        return fused_tokens
