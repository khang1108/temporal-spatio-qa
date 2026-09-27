"""
RoBERTa-base PLM Baseline for STKGQA.
Following Dai et al. (KBS 2025) Section 6.1.1:
- RoBERTa-base encoder (768-d)
- Learnable linear projection layer (768 -> 512)
- Entity embeddings (num_entities x 512)
- Scoring function: Dot product over all entities
- Loss: Cross-entropy over target answers
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class RoBERTaBaseline(nn.Module):
    def __init__(self,
                 model_name: str = "roberta-base",
                 num_entities: int = 5897,
                 embedding_dim: int = 512,
                 freeze_encoder: bool = False):
        super().__init__()
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim

        # RoBERTa encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_dim = self.encoder.config.hidden_size  # 768 for roberta-base

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Projection layer: maps 768-d question representation to 512-d entity space
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(0.1)
        )

        # Entity embedding table in the STKG
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                cls_rep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes logits over all candidate entities.
        Returns tensor of shape (batch_size, num_entities).
        """
        if cls_rep is None:
            assert input_ids is not None and attention_mask is not None, "Provide input_ids/attention_mask or cls_rep"
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_rep = outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)

        # Project to 512 dimensions
        q_proj = self.projection(cls_rep)  # (batch_size, 512)

        # Dot product with all entity embeddings
        all_entity_emb = self.entity_embeddings.weight  # (num_entities, 512)
        logits = torch.matmul(q_proj, all_entity_emb.t())  # (batch_size, num_entities)

        return logits

    def compute_loss(self, logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy loss against target answer distribution.
        Handles multi-answer targets using soft cross-entropy / KL divergence with log-softmax.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(target_dist * log_probs, dim=-1).mean()
        return loss

    def predict_topk(self,
                     input_ids: Optional[torch.Tensor] = None,
                     attention_mask: Optional[torch.Tensor] = None,
                     cls_rep: Optional[torch.Tensor] = None,
                     k: int = 10) -> torch.Tensor:
        """
        Returns top-k entity IDs for each sample in the batch.
        """
        logits = self.forward(input_ids, attention_mask, cls_rep=cls_rep)
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
            model_name="roberta",
            checkpoint_path=checkpoint_path,
            url=url,
            device=device,
            force_download=force_download
        )

