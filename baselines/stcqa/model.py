"""
STCQA (Spatio-Temporal Complex Question Answering) Model.
Implements the full neural architecture from Dai et al. (KBS 2025):
- Preprocessing token replacements
- ST-TComplEx embedding layer
- 2-layer Transformer Fusion (4 heads)
- Bidirectional scoring function: max(phi_ST(e_c, W_E q, e, v_t, v_l), phi_ST(e, W_E q, e_c, v_t, v_l))
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from baselines.stcqa.st_embedding import STComplExEmbedding, complex_mul


class STCQAModel(nn.Module):
    def __init__(self,
                 model_name: str = "roberta-base",
                 num_entities: int = 5897,
                 num_relations: int = 20,
                 num_timestamps: int = 600,
                 num_locations: int = 2500,
                 embedding_dim: int = 512,
                 num_transformer_layers: int = 2,
                 nhead: int = 4,
                 freeze_encoder: bool = False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities

        # Text encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_dim = self.encoder.config.hidden_size

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # ST-TComplEx embedding tables
        self.st_embeddings = STComplExEmbedding(
            num_entities=num_entities,
            num_relations=num_relations,
            num_timestamps=num_timestamps,
            num_locations=num_locations,
            embedding_dim=embedding_dim
        )

        # Linear projections for question text: real and imaginary parts
        self.proj_q_re = nn.Linear(hidden_dim, embedding_dim)
        self.proj_q_im = nn.Linear(hidden_dim, embedding_dim)

        # 2-layer Transformer Fusion layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_fusion = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Learnable relation transformation matrix W_E (D x D) per Eq. (4)
        self.W_E = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Default query vectors
        self.default_c_re = nn.Parameter(torch.randn(1, embedding_dim) * 0.02)
        self.default_c_im = nn.Parameter(torch.randn(1, embedding_dim) * 0.02)
        self.default_t_re = nn.Parameter(torch.ones(1, embedding_dim))
        self.default_t_im = nn.Parameter(torch.zeros(1, embedding_dim))
        self.default_l_re = nn.Parameter(torch.ones(1, embedding_dim))
        self.default_l_im = nn.Parameter(torch.zeros(1, embedding_dim))

    def encode_and_fuse(self,
                        input_ids: torch.Tensor,
                        attention_mask: torch.Tensor,
                        c_re: torch.Tensor,
                        t_re: torch.Tensor,
                        l_re: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuses question text encoding with clue embeddings via 2-layer Transformer.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0, :]

        raw_q_re = self.proj_q_re(cls_rep)
        raw_q_im = self.proj_q_im(cls_rep)

        # Stack sequence of tokens: [raw_q, clue_c, clue_t, clue_l] for fusion
        seq_re = torch.stack([raw_q_re, c_re, t_re, l_re], dim=1)  # (B, 4, D)
        fused_seq = self.transformer_fusion(seq_re)
        fused_q_re = fused_seq[:, 0, :]

        # Question relation transformation: W_E * q
        rel_re = self.W_E(fused_q_re)
        rel_im = self.W_E(raw_q_im)

        return rel_re, rel_im

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                central_ids: torch.Tensor = None,
                time_ids: torch.Tensor = None,
                loc_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Bidirectional scoring per Equation (4) in paper:
        max(phi_ST(e_c, W_E q, e, v_t, v_l), phi_ST(e, W_E q, e_c, v_t, v_l))
        """
        batch_size = input_ids.size(0)

        # Retrieve central entity embedding
        if central_ids is not None and (central_ids >= 0).any():
            valid_c = (central_ids >= 0).unsqueeze(-1)
            safe_c = central_ids.clamp(min=0)
            c_re = torch.where(valid_c, self.st_embeddings.ent_re(safe_c), self.default_c_re.expand(batch_size, -1))
            c_im = torch.where(valid_c, self.st_embeddings.ent_im(safe_c), self.default_c_im.expand(batch_size, -1))
        else:
            c_re = self.default_c_re.expand(batch_size, -1)
            c_im = self.default_c_im.expand(batch_size, -1)

        # Retrieve temporal clue embedding
        if time_ids is not None and (time_ids >= 0).any():
            valid_t = (time_ids >= 0).unsqueeze(-1)
            safe_t = time_ids.clamp(min=0)
            t_re = torch.where(valid_t, self.st_embeddings.time_re(safe_t), self.default_t_re.expand(batch_size, -1))
            t_im = torch.where(valid_t, self.st_embeddings.time_im(safe_t), self.default_t_im.expand(batch_size, -1))
        else:
            t_re = self.default_t_re.expand(batch_size, -1)
            t_im = self.default_t_im.expand(batch_size, -1)

        # Retrieve spatial clue embedding
        if loc_ids is not None and (loc_ids >= 0).any():
            valid_l = (loc_ids >= 0).unsqueeze(-1)
            safe_l = loc_ids.clamp(min=0)
            l_re = torch.where(valid_l, self.st_embeddings.loc_re(safe_l), self.default_l_re.expand(batch_size, -1))
            l_im = torch.where(valid_l, self.st_embeddings.loc_im(safe_l), self.default_l_im.expand(batch_size, -1))
        else:
            l_re = self.default_l_re.expand(batch_size, -1)
            l_im = self.default_l_im.expand(batch_size, -1)

        # Fuse question representation
        r_re, r_im = self.encode_and_fuse(input_ids, attention_mask, c_re, t_re, l_re)

        # Forward score: phi_ST(e_c, W_E q, e, v_t, v_l)
        forward_scores = self.st_embeddings.score_fact(c_re, c_im, r_re, r_im, t_re, t_im, l_re, l_im)

        # Reverse score: phi_ST(e, W_E q, e_c, v_t, v_l)
        # Complex product of r, t, l:
        rt_re, rt_im = complex_mul(r_re, r_im, t_re, t_im)
        rtl_re, rtl_im = complex_mul(rt_re, rt_im, l_re, l_im)
        # rtl * conj(e_c)
        c_rtl_re, c_rtl_im = complex_mul(c_re, -c_im, rtl_re, rtl_im)
        # Inner product with candidate entities
        all_e_re = self.st_embeddings.ent_re.weight
        all_e_im = self.st_embeddings.ent_im.weight
        reverse_scores = torch.matmul(c_rtl_re, all_e_re.t()) + torch.matmul(c_rtl_im, all_e_im.t())

        # Bidirectional max
        final_scores = torch.maximum(forward_scores, reverse_scores)
        return final_scores

    def compute_loss(self, logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(target_dist * log_probs, dim=-1).mean()
        return loss

    def predict_topk(self,
                     input_ids: torch.Tensor,
                     attention_mask: torch.Tensor,
                     central_ids: torch.Tensor = None,
                     time_ids: torch.Tensor = None,
                     loc_ids: torch.Tensor = None,
                     k: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(input_ids, attention_mask, central_ids, time_ids, loc_ids)
        topk_scores, topk_indices = torch.topk(logits, k=k, dim=-1)
        return topk_indices, topk_scores
