import torch
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig


def load_fastconformer(
    model_name: str = "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi",
    att_context_size=(70, 1),  # ~80ms worst
    decoder_type: str = "rnnt",
    device: str = "cuda:0",
):
    """Load cache-aware streaming FastConformer Hybrid model (fp32)."""
    dev = torch.device(device)
    asr = EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name=model_name)
    asr = asr.to(dev, dtype=torch.float32)
    asr.eval()

    if hasattr(asr.encoder, "set_default_att_context_size"):
        asr.encoder.set_default_att_context_size(list(att_context_size))
    else:
        raise RuntimeError("Model does not support multiple lookaheads.")

    if hasattr(asr, "change_decoding_strategy"):
        # Align with NVIDIA streaming example: disable fused batching in RNNT decoding for streaming
        if decoder_type == "rnnt":
            try:
                rnnt_cfg = RNNTDecodingConfig(fused_batch_size=-1)
                asr.change_decoding_strategy(rnnt_cfg, decoder_type=decoder_type)
            except Exception:
                asr.change_decoding_strategy(decoder_type=decoder_type)
        else:
            asr.change_decoding_strategy(decoder_type=decoder_type)

    torch.set_float32_matmul_precision("high")
    return asr


