from fairseq.models import register_model_architecture
from fairseq.models.transformer.transformer_legacy import base_architecture


@register_model_architecture("transformer", "transformer_wmt_en_de_t2t")
def transformer_wmt_en_de_t2t(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    base_architecture(args)
