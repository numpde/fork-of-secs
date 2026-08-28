from transformers import AutoTokenizer

from secs.models.encoders.smiles.molformer import MOLFORMER_CHECKPOINT, MOLFORMER_REVISION

# The checkpoint's embedding table was trained with this tokenizer revision.
# Pin it so later tokenizer changes cannot assign different token IDs.
SMILES_TOKENIZER = AutoTokenizer.from_pretrained(
    MOLFORMER_CHECKPOINT,
    revision=MOLFORMER_REVISION,
    trust_remote_code=True,
)
