import torch
import torch.nn as nn
import torch.nn.functional as F

class QueryMemoryFusion(nn.Module):
    def __init__(self, num_queries, embed_dim, hidden_dim=128, fusion_type='early'):
        """
        fusion_type: 'early' or 'late' — determines how query identity is fused
        """
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.fusion_type = fusion_type

        # Learnable positional or class embeddings per query
        self.query_embed = nn.Embedding(num_queries, embed_dim)

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
        query_embed = self.query_embed(torch.arange(Q, device=memory_tokens.device))
        query_embed = query_embed.unsqueeze(1).expand(Q, L, E)

        if self.fusion_type == 'early':
            fused_input = memory_tokens + query_embed  # Add class identity directly
        elif self.fusion_type == 'late':
            fused_input = torch.cat([memory_tokens, query_embed], dim=-1)  # Learn how to combine

        scores = self.score_mlp(fused_input)  # (Q, L, 1)
        attn_weights = F.softmax(scores, dim=0)  # Softmax over Q

        fused_tokens = (memory_tokens * attn_weights).sum(dim=0, keepdim=True)  # (1, L, E)

        return fused_tokens
