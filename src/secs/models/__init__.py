from secs.models.base import HFCausalLMEncoder, ModalityEncoder
from secs.models.encoders import (
    CNmrTransformerEncoder,
    GraphGCNEncoder,
    GraphGINEncoder,
    HNmrCNNEncoder,
    HNmrConvAttentionEncoder,
    HNmrDilatedCNNEncoder,
    HsqcCNNEncoder,
    IrCNNEncoder,
    MolformerEncoder,
)
from secs.models.heads import ProjectionHead
from secs.models.model import MolBind
from secs.models.registry import available_encoders, register_encoder, resolve_encoder

__all__ = [
    "CNmrTransformerEncoder",
    "GraphGCNEncoder",
    "GraphGINEncoder",
    "HFCausalLMEncoder",
    "HNmrCNNEncoder",
    "HNmrConvAttentionEncoder",
    "HNmrDilatedCNNEncoder",
    "HsqcCNNEncoder",
    "IrCNNEncoder",
    "ModalityEncoder",
    "MolBind",
    "MolformerEncoder",
    "ProjectionHead",
    "SECSModule",
    "available_encoders",
    "register_encoder",
    "resolve_encoder",
]


def __getattr__(name: str):
    if name == "SECSModule":
        from secs.models.lightning_module import SECSModule

        return SECSModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
