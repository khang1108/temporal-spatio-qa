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
from src.models.stcqa.st_embedding import STComplExEmbedding, complex_mul


class STCQAModel(nn.Module):
    def __init__(self,
                 model_name: str = "roberta-base",
                 num_entities: int = 5897,
                 num_relations: int = 32,
                 num_timestamps: int = 170,
                 num_locations: int = 1352,
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
                        input_ids: Optional[torch.Tensor],
                        attention_mask: Optional[torch.Tensor],
                        c_re: torch.Tensor,
                        t_re: torch.Tensor,
                        l_re: torch.Tensor,
                        cls_rep: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuses question text encoding with clue embeddings via 2-layer Transformer.
        """
        if cls_rep is None:
            assert input_ids is not None and attention_mask is not None
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
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                central_ids: Optional[torch.Tensor] = None,
                time_ids: Optional[torch.Tensor] = None,
                loc_ids: Optional[torch.Tensor] = None,
                cls_rep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Bidirectional scoring per Equation (4) in paper:
        max(phi_ST(e_c, W_E q, e, v_t, v_l), phi_ST(e, W_E q, e_c, v_t, v_l))
        """
        batch_size = cls_rep.size(0) if cls_rep is not None else input_ids.size(0)

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
        r_re, r_im = self.encode_and_fuse(input_ids, attention_mask, c_re, t_re, l_re, cls_rep=cls_rep)

        # Forward score: phi_ST(e_c, W_E q, e, v_t, v_l)
        forward_scores = self.st_embeddings.score_fact(c_re, c_im, r_re, r_im, t_re, t_im, l_re, l_im)

        # Reverse score: phi_ST(e, W_E q, e_c, v_t, v_l)
        reverse_scores = self.st_embeddings.score_reverse_fact(c_re, c_im, r_re, r_im, t_re, t_im, l_re, l_im)

        # Bidirectional max
        final_scores = torch.maximum(forward_scores, reverse_scores)
        return final_scores

    def compute_loss(self, logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(target_dist * log_probs, dim=-1).mean()
        return loss

    def predict_topk(self,
                     input_ids: Optional[torch.Tensor] = None,
                     attention_mask: Optional[torch.Tensor] = None,
                     central_ids: Optional[torch.Tensor] = None,
                     time_ids: Optional[torch.Tensor] = None,
                     loc_ids: Optional[torch.Tensor] = None,
                     cls_rep: Optional[torch.Tensor] = None,
                     k: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(input_ids, attention_mask, central_ids, time_ids, loc_ids, cls_rep=cls_rep)
        topk_scores, topk_indices = torch.topk(logits, k=k, dim=-1)
        return topk_indices, topk_scores

    def load_pretrained_stkg(self, checkpoint_path: str, freeze: bool = True):
        """
        Loads pre-trained ST-TComplEx embeddings from STKG link prediction stage (Section 5.3 & Appendix B).
        Optionally freezes the embeddings during QA fine-tuning.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        with torch.no_grad():
            n_ent = min(self.st_embeddings.num_entities, sd["ent_re.weight"].size(0))
            self.st_embeddings.ent_re.weight[:n_ent].copy_(sd["ent_re.weight"][:n_ent])
            self.st_embeddings.ent_im.weight[:n_ent].copy_(sd["ent_im.weight"][:n_ent])

            n_rel = min(self.st_embeddings.num_relations, sd["rel_re.weight"].size(0))
            self.st_embeddings.rel_re.weight[:n_rel].copy_(sd["rel_re.weight"][:n_rel])
            self.st_embeddings.rel_im.weight[:n_rel].copy_(sd["rel_im.weight"][:n_rel])

            n_time = min(self.st_embeddings.num_timestamps, sd["time_re.weight"].size(0))
            self.st_embeddings.time_re.weight[:n_time].copy_(sd["time_re.weight"][:n_time])
            self.st_embeddings.time_im.weight[:n_time].copy_(sd["time_im.weight"][:n_time])

            n_loc = min(self.st_embeddings.num_locations, sd["loc_re.weight"].size(0))
            self.st_embeddings.loc_re.weight[:n_loc].copy_(sd["loc_re.weight"][:n_loc])
            self.st_embeddings.loc_im.weight[:n_loc].copy_(sd["loc_im.weight"][:n_loc])

        if freeze:
            for param in self.st_embeddings.parameters():
                param.requires_grad = False
            print(f"[STCQA] Loaded and FROZEN pre-trained STKG embeddings from {checkpoint_path}")
        else:
            print(f"[STCQA] Loaded pre-trained STKG embeddings from {checkpoint_path} (trainable)")

    def load_checkpoint(self,
                        checkpoint_path: Optional[str] = None,
                        url: Optional[str] = None,
                        device: Optional[torch.device] = None,
                        force_download: bool = False):
        """
        Checks whether .pt file exists; if not, downloads it, then loads weights into model.
        """
        from src.utils.checkpoint import load_model_checkpoint
        return load_model_checkpoint(
            self,
            model_name="stcqa",
            checkpoint_path=checkpoint_path,
            url=url,
            device=device,
            force_download=force_download
        )


