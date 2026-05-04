"""
Early-ViT — Early Action Recognition with Action Prototypes
Camporese, Bergamo, Lin, Tighe, Modolo (AWS AI Labs), ECCV 2024
arXiv:2312.06598

Variante "encoder image" : 1 frame par segment, donc chaque frame du
dataloader est un segment t. Format des entrées : (B, T, C, H, W).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16


class CausalTransformerDecoder(nn.Module):
    """T-Dec-B du papier : 6 blocs Transformer-encoder avec masque causal."""
    def __init__(self, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=1024, dropout=0.1, max_len=32):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    @staticmethod
    def _causal_mask(T, device):
        return torch.triu(
            torch.full((T, T), float("-inf"), device=device), diagonal=1
        )

    def forward(self, z_enc):
        # z_enc : (B, T, D)
        B, T, _ = z_enc.shape
        x = z_enc + self.pos_emb[:, :T]
        mask = self._causal_mask(T, z_enc.device)
        return self.layers(x, mask=mask, is_causal=True)


class EarlyVit(nn.Module):
    """
    Early-ViT — version "encodeur image".
    Chaque segment t est UNE frame, encodée par un ViT_b_16 from-scratch.
    Entrées : (B, T, C, H, W).
    """

    def __init__(
        self,
        num_classes: int,
        d_enc: int = 768,
        d_model: int = 256,
        num_decoder_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_segments: int = 32,
        label_smoothing: float = 0.1,
        pretrained: bool = False,    # gardé pour compat config mais non utilisé
    ):
        super().__init__()
        self.K = num_classes
        self.D = d_model
        self.label_smoothing = label_smoothing

        # ---- Encodeur ViT image, from-scratch ----
        encoder = vit_b_16(weights=None)
        encoder.heads = nn.Identity()  # type: ignore # on n'utilise pas la tête de classification du ViT
        self.encoder = encoder

        # Projection vers la dimension du décodeur
        self.proj = nn.Linear(d_enc, d_model)

        # Décodeur causal T-Dec-B
        self.decoder = CausalTransformerDecoder(
            d_model=d_model, nhead=nhead, num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            max_len=max_segments,
        )

        # Tête de classification (Eq. 1)
        self.classifier = nn.Linear(d_model, num_classes)

        # Banque de prototypes P ∈ R^(K × D)
        self.prototypes = nn.Parameter(torch.randn(num_classes, d_model) * 0.02)

        # Future Proto Predictor f(·) — MLP
        self.future_proto_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def encode_clips(self, clips: torch.Tensor) -> torch.Tensor:
        """
        clips : (B, T, C, H, W) — T frames par vidéo
        retourne : (B, T, D)
        """
        B, T, C, H, W = clips.shape
        x = clips.reshape(B * T, C, H, W)    # (B*T, 3, 224, 224)
        feats = self.encoder(x)              # (B*T, 768)
        feats = self.proj(feats)             # (B*T, 256)
        return feats.reshape(B, T, self.D)   # (B, T, 256)

    def forward(self, clips: torch.Tensor):
        """
        clips : (B, T, C, H, W)
        Renvoie directement les logits (B, K) — prédiction au dernier segment,
        compatible avec CrossEntropyLoss standard.
        
        Les sorties intermédiaires (z, z_last) sont stockées dans self._cache
        si tu en as besoin plus tard pour les pertes prototypes.
        """
        z_enc = self.encode_clips(clips)            # (B, T, D)
        z = self.decoder(z_enc)                     # (B, T, D)
        logits_all = self.classifier(z)             # (B, T, K)
        
        # On stocke les sorties intermédiaires au cas où (utile pour debug)
        self._cache = {
            "logits_all": logits_all,
            "z": z,
            "z_last": z[:, -1],
        }
        
        # Renvoie uniquement les logits au dernier segment : (B, K)
        return logits_all[:, -1]

    @torch.no_grad()
    def online_step(self, new_frame: torch.Tensor, cache_z_enc=None):
        """
        new_frame  : (B, C, H, W) — nouvelle frame du segment t
        cache_z_enc: (B, t-1, D) ou None
        """
        feats = self.proj(self.encoder(new_frame)).unsqueeze(1)  # (B, 1, D)
        z_enc = feats if cache_z_enc is None else torch.cat(
            [cache_z_enc, feats], dim=1
        )
        z = self.decoder(z_enc)
        logits = self.classifier(z[:, -1])
        return logits, z_enc
# ======================================================================
#   PERTES — Section 3.2 et 3.3 du papier
# ======================================================================
 
def loss_proto(z_last: torch.Tensor,
               prototypes: torch.Tensor,
               y: torch.Tensor) -> torch.Tensor:
    """
    L_proto (Eq. 2-3) : entraîne la banque de prototypes.
 
    Similarité = -||P(k) - sg[z(T)]||_2          (Eq. 2)
    L_proto = cross-entropy(softmax(s), one_hot(y))  (Eq. 3)
 
    Le stop-grad sur z(T) sépare l'objectif des prototypes des autres.
    """
    z_t = z_last.detach()                            # sg[z(T)]
    # ||P(k) - z||_2 pour chaque k -> (B, K)
    dist = torch.cdist(z_t, prototypes, p=2)         # (B, K)
    s = -dist                                        # similarité (Eq. 2)
    return F.cross_entropy(s, y)
 
 
def loss_reg(z_all: torch.Tensor,
             prototypes: torch.Tensor,
             y: torch.Tensor,
             future_proto_predictor: nn.Module) -> torch.Tensor:
    """
    L_reg (Eq. 4-5) : régularise le décodeur en demandant à chaque z(t)
    de prédire (via f) le prototype de la classe vraie.
 
    s_reg(t,k) = -||sg[P(k)] - f(z(t))||_2           (Eq. 4)
    L_reg = moyenne_t cross-entropy(softmax_k(s_reg), one_hot(y))  (Eq. 5)
 
    Stop-grad sur P pour ne pas court-circuiter L_proto.
    """
    B, T, D = z_all.shape
    P = prototypes.detach()                          # sg[P]
    # f appliqué indépendamment à chaque z(t)
    f_z = future_proto_predictor(z_all)              # (B, T, D)
    # ||P(k) - f(z(t))||_2 pour chaque (b, t, k)
    dist = torch.cdist(f_z.reshape(B * T, D), P, p=2)  # (B*T, K)
    s_reg = -dist
    y_rep = y.repeat_interleave(T)                   # (B*T,)
    return F.cross_entropy(s_reg, y_rep)
 
 
def loss_clf_at(logits_t: torch.Tensor, y: torch.Tensor,
                label_smoothing: float = 0.1) -> torch.Tensor:
    """L_clf(x, y, t) — cross-entropy classique pour un t donné (Eq. 1)."""
    return F.cross_entropy(logits_t, y, label_smoothing=label_smoothing)
 
 
def loss_dyn(logits: torch.Tensor, y: torch.Tensor,
             epoch: int, e_star: int = 15,
             label_smoothing: float = 0.1) -> torch.Tensor:
    """
    L_dyn (Eq. 6-8) : la perte dynamique du papier.
        - epoch <= e* : L_ol = L_clf(T)         (only-last)
        - epoch >  e* : L_all = (1/T) Σ_t L_clf(t)
 
    Démarre en optimisant la prédiction sur la vidéo entière (haute
    accuracy finale), puis bascule sur toutes les étapes pour bien
    prédire en début de vidéo aussi.
    """
    if epoch <= e_star:
        # L_ol : loss uniquement sur le dernier token z(T)
        return loss_clf_at(logits[:, -1], y, label_smoothing)
    else:
        # L_all : moyenne sur tous les pas de temps
        B, T, K = logits.shape
        return loss_clf_at(
            logits.reshape(B * T, K),
            y.repeat_interleave(T),
            label_smoothing,
        )
 
 
def early_vit_total_loss(outputs: dict, prototypes: torch.Tensor,
                         future_proto_predictor: nn.Module,
                         y: torch.Tensor, epoch: int,
                         e_star: int = 15,
                         label_smoothing: float = 0.1) -> dict:
    """
    L_tot = L_dyn + L_proto + L_reg                  (Eq. 9)
 
    À utiliser dans la boucle d'entraînement :
        out = model(clips)
        losses = early_vit_total_loss(out, model.prototypes,
                                      model.future_proto_predictor,
                                      labels, epoch)
        losses['total'].backward()
    """
    l_dyn  = loss_dyn(outputs["logits"], y, epoch, e_star, label_smoothing)
    l_pro  = loss_proto(outputs["z_last"], prototypes, y)
    l_reg  = loss_reg(outputs["z"], prototypes, y, future_proto_predictor)
    return {
        "L_dyn":   l_dyn,
        "L_proto": l_pro,
        "L_reg":   l_reg,
        "total":   l_dyn + l_pro + l_reg,
    }
 
 