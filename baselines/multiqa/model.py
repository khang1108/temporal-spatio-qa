"""
MultiQA Baseline for Temporal Knowledge Graph Question Answering.
Implements TComplEx-based temporal scoring with multi-granularity (Year) temporal embeddings.
Reference: Chen et al. (ACL 2023) / Dai et al. (KBS 2025) Section 6.1.1.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class MultiQABaseline(nn.Module):
    """
    MultiQA: Temporal KGQA model combining text representation with TComplEx embeddings.
    Scoring: Re(<e_s, q (x) v_t, conj(e_o)>) = h_re * E_re^T + h_im * E_im^T
    """
    def __init__(self,
                 model_name: str = "roberta-base",
                 num_entities: int = 5897,
                 num_timestamps: int = 600,
                 embedding_dim: int = 512,
                 freeze_encoder: bool = False):
        super().__init__()
        self.num_entities = num_entities
        self.num_timestamps = num_timestamps
        self.embedding_dim = embedding_dim

        # Text encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_dim = self.encoder.config.hidden_size

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Projections to complex question representation: q_re and q_im
        self.proj_re = nn.Linear(hidden_dim, embedding_dim)
        self.proj_im = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(0.1)

        # Complex Entity Embeddings: E_re, E_im
        self.ent_re = nn.Embedding(num_entities, embedding_dim)
        self.ent_im = nn.Embedding(num_entities, embedding_dim)
        nn.init.xavier_uniform_(self.ent_re.weight)
        nn.init.xavier_uniform_(self.ent_im.weight)

        # Complex Timestamp Embeddings: T_re, T_im
        self.time_re = nn.Embedding(num_timestamps, embedding_dim)
        self.time_im = nn.Embedding(num_timestamps, embedding_dim)
        nn.init.xavier_uniform_(self.time_re.weight)
        nn.init.xavier_uniform_(self.time_im.weight)

        # Default query anchor if no explicit central entity
        self.default_subj_re = nn.Parameter(torch.randn(1, embedding_dim) * 0.02)
        self.default_subj_im = nn.Parameter(torch.randn(1, embedding_dim) * 0.02)

        # Default time vector if no explicit timestamp found
        self.default_time_re = nn.Parameter(torch.ones(1, embedding_dim))
        self.default_time_im = nn.Parameter(torch.zeros(1, embedding_dim))

    def get_complex_question(self,
                             input_ids: Optional[torch.Tensor] = None,
                             attention_mask: Optional[torch.Tensor] = None,
                             cls_rep: Optional[torch.Tensor] = None):
        if cls_rep is None:
            assert input_ids is not None and attention_mask is not None
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_rep = outputs.last_hidden_state[:, 0, :]
        q_re = self.dropout(self.proj_re(cls_rep))
        q_im = self.dropout(self.proj_im(cls_rep))
        return q_re, q_im

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                subj_ids: Optional[torch.Tensor] = None,
                time_ids: Optional[torch.Tensor] = None,
                cls_rep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes TComplEx logits over all candidate entities.
        Returns: (batch_size, num_entities)
        """
        batch_size = cls_rep.size(0) if cls_rep is not None else input_ids.size(0)
        q_re, q_im = self.get_complex_question(input_ids, attention_mask, cls_rep=cls_rep)

        # Subject complex embedding
        if subj_ids is not None and (subj_ids >= 0).any():
            valid_mask = (subj_ids >= 0).unsqueeze(-1)
            safe_ids = subj_ids.clamp(min=0)
            s_re = torch.where(valid_mask, self.ent_re(safe_ids), self.default_subj_re.expand(batch_size, -1))
            s_im = torch.where(valid_mask, self.ent_im(safe_ids), self.default_subj_im.expand(batch_size, -1))
        else:
            s_re = self.default_subj_re.expand(batch_size, -1)
            s_im = self.default_subj_im.expand(batch_size, -1)

        # Time complex embedding
        if time_ids is not None and (time_ids >= 0).any():
            valid_mask = (time_ids >= 0).unsqueeze(-1)
            safe_time = time_ids.clamp(min=0)
            t_re = torch.where(valid_mask, self.time_re(safe_time), self.default_time_re.expand(batch_size, -1))
            t_im = torch.where(valid_mask, self.time_im(safe_time), self.default_time_im.expand(batch_size, -1))
        else:
            t_re = self.default_time_re.expand(batch_size, -1)
            t_im = self.default_time_im.expand(batch_size, -1)

        # Complex element-wise product: h = s (x) (q (x) t)
        # First: qt = q (x) t = (q_re*t_re - q_im*t_im) + i(q_re*t_im + q_im*t_re)
        qt_re = q_re * t_re - q_im * t_im
        qt_im = q_re * t_im + q_im * t_re

        # Next: h = s (x) qt = (s_re*qt_re - s_im*qt_im) + i(s_re*qt_im + s_im*qt_re)
        h_re = s_re * qt_re - s_im * qt_im
        h_im = s_re * qt_im + s_im * qt_re

        # Inner product with all candidate entities: Re(h * conj(E)) = h_re * E_re^T + h_im * E_im^T
        all_e_re = self.ent_re.weight  # (num_entities, D)
        all_e_im = self.ent_im.weight  # (num_entities, D)

        logits = torch.matmul(h_re, all_e_re.t()) + torch.matmul(h_im, all_e_im.t())
        return logits

    def compute_loss(self, logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(target_dist * log_probs, dim=-1).mean()
        return loss

    def predict_topk(self,
                     input_ids: Optional[torch.Tensor] = None,
                     attention_mask: Optional[torch.Tensor] = None,
                     subj_ids: Optional[torch.Tensor] = None,
                     time_ids: Optional[torch.Tensor] = None,
                     cls_rep: Optional[torch.Tensor] = None,
                     k: int = 10) -> torch.Tensor:
        logits = self.forward(input_ids, attention_mask, subj_ids, time_ids, cls_rep=cls_rep)
        _, topk_indices = torch.topk(logits, k=k, dim=-1)
        return topk_indices

    def load_checkpoint(self,
                        checkpoint_path: Optional[str] = None,
                        url: Optional[str] = None,
                        device: Optional[torch.device] = None,
                        force_download: bool = False):
        """
        Checks whether .pt file exists; if not, downloads it, then loads weights into model.
        """
        from src.checkpoint_utils import load_model_checkpoint
        return load_model_checkpoint(
            self,
            model_name="multiqa",
            checkpoint_path=checkpoint_path,
            url=url,
            device=device,
            force_download=force_download
        )

