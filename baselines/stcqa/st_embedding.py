"""
Spatio-Temporal Complex Embedding (ST-TComplEx) module.
Implements Equation (2) and (3) from Dai et al. (KBS 2025):
phi_ST(e_s, r, e_o, t, l) = Re(<e_s, r (x) t (x) l, conj(e_o)>)
"""

import torch
import torch.nn as nn


def complex_mul(a_re: torch.Tensor, a_im: torch.Tensor,
                b_re: torch.Tensor, b_im: torch.Tensor):
    """Element-wise complex multiplication: (a_re + i*a_im) * (b_re + i*b_im)"""
    res_re = a_re * b_re - a_im * b_im
    res_im = a_re * b_im + a_im * b_re
    return res_re, res_im


class STComplExEmbedding(nn.Module):
    """
    Joint Spatio-Temporal Complex Embedding table for entities, relations, timestamps, and locations.
    """
    def __init__(self,
                 num_entities: int = 5897,
                 num_relations: int = 20,
                 num_timestamps: int = 600,
                 num_locations: int = 2500,
                 embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities

        # Entity complex embeddings
        self.ent_re = nn.Embedding(num_entities, embedding_dim)
        self.ent_im = nn.Embedding(num_entities, embedding_dim)
        nn.init.xavier_uniform_(self.ent_re.weight)
        nn.init.xavier_uniform_(self.ent_im.weight)

        # Relation complex embeddings
        self.rel_re = nn.Embedding(num_relations, embedding_dim)
        self.rel_im = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.rel_re.weight)
        nn.init.xavier_uniform_(self.rel_im.weight)

        # Timestamp complex embeddings
        self.time_re = nn.Embedding(num_timestamps, embedding_dim)
        self.time_im = nn.Embedding(num_timestamps, embedding_dim)
        nn.init.xavier_uniform_(self.time_re.weight)
        nn.init.xavier_uniform_(self.time_im.weight)

        # Location complex embeddings
        self.loc_re = nn.Embedding(num_locations, embedding_dim)
        self.loc_im = nn.Embedding(num_locations, embedding_dim)
        nn.init.xavier_uniform_(self.loc_re.weight)
        nn.init.xavier_uniform_(self.loc_im.weight)

    def score_fact(self, s_re, s_im, r_re, r_im, t_re, t_im, l_re, l_im):
        """
        Computes phi_ST score for a batch of query facts against all candidate entities.
        Returns: logits of shape (batch_size, num_entities)
        """
        # Complex product of relation, timestamp, location: rtl = r * t * l
        rt_re, rt_im = complex_mul(r_re, r_im, t_re, t_im)
        rtl_re, rtl_im = complex_mul(rt_re, rt_im, l_re, l_im)

        # s * rtl
        h_re, h_im = complex_mul(s_re, s_im, rtl_re, rtl_im)

        # Inner product with conjugate of all candidate entities:
        # Re(h * conj(E)) = h_re * E_re^T + h_im * E_im^T
        all_e_re = self.ent_re.weight  # (num_entities, D)
        all_e_im = self.ent_im.weight  # (num_entities, D)

        scores = torch.matmul(h_re, all_e_re.t()) + torch.matmul(h_im, all_e_im.t())
        return scores
