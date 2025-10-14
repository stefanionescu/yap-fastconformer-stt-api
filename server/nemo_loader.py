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
        # Force non-fused greedy RNNT for streaming; avoid fused "blank-as-pad" path
        if decoder_type == "rnnt":
            try:
                rnnt_cfg = RNNTDecodingConfig(strategy="greedy")
                # Best effort: explicitly disable fused batching knobs if present on this config
                try:
                    setattr(rnnt_cfg, "fused_batch_size", -1)
                except Exception:
                    pass
                asr.change_decoding_strategy(rnnt_cfg, decoder_type=decoder_type)
            except Exception:
                # Fallback to any signature that accepts decoding_cfg as a dict
                try:
                    asr.change_decoding_strategy(decoding_cfg={"strategy": "greedy"}, decoder_type=decoder_type)
                except Exception:
                    asr.change_decoding_strategy(decoder_type=decoder_type)

            # Also flip decoder-level fused switches when available
            dec = getattr(asr, "decoding", None)
            if dec is not None:
                if hasattr(dec, "blank_as_pad"):
                    try:
                        dec.blank_as_pad = False
                    except Exception:
                        pass
                if hasattr(dec, "greedy_fused_batch_size"):
                    try:
                        dec.greedy_fused_batch_size = -1
                    except Exception:
                        pass
                if hasattr(dec, "fused_batch_size"):
                    try:
                        dec.fused_batch_size = -1
                    except Exception:
                        pass
                if hasattr(dec, "use_gpu_kernels"):
                    try:
                        dec.use_gpu_kernels = False
                    except Exception:
                        pass
        else:
            asr.change_decoding_strategy(decoder_type=decoder_type)

    torch.set_float32_matmul_precision("high")
    return asr


