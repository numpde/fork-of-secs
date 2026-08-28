from torch import Tensor
from transformers import AutoConfig, AutoModel

from secs.models.base import HFCausalLMEncoder
from secs.models.registry import register_encoder

MOLFORMER_CHECKPOINT = "ibm-research/MoLFormer-XL-both-10pct"
MOLFORMER_REVISION = "7b12d946c181a37f6012b9dc3b002275de070314"


@register_encoder("smiles", "molformer", default=True)
class MolformerEncoder(HFCausalLMEncoder):
    """MoLFormer-XL over tokenized SMILES, read out from its pooler."""

    output_dim = 768

    def __init__(self, freeze_encoder: bool = False, pretrained: bool = True, **kwargs) -> None:
        super().__init__(MOLFORMER_CHECKPOINT, freeze_encoder, pretrained, **kwargs)

    def _initialize_encoder(self) -> None:
        self.encoder = (
            # Load IBM's trained parameters when pretrained weights are requested.
            AutoModel.from_pretrained
            if self.pretrained
            # Otherwise initialize a new model from the pinned definition and configuration.
            else lambda model_name, **kw: AutoModel.from_config(
                AutoConfig.from_pretrained(model_name, **kw),
                trust_remote_code=kw["trust_remote_code"],
                code_revision=kw["revision"],
            )
        )(
            self.model_name,
            trust_remote_code=True,
            revision=MOLFORMER_REVISION,
            deterministic_eval=True,
        )
        if self.frozen:
            self.freeze()

    def forward(self, x: tuple[Tensor, ...]) -> Tensor:
        token_ids, attention_mask = x[0], x[1]
        output = self.encoder(input_ids=token_ids, attention_mask=attention_mask)
        return output.pooler_output
