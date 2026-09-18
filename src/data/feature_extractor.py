import torch
import torchaudio


def load_audio_mono_16k(path: str) -> torch.Tensor:
    """Load audio via soundfile (torchaudio.load needs torchcodec, which is
    broken in this venv), downmix to mono, resample to 16 kHz.

    Returns: waveform [1, num_samples]
    """
    import soundfile as sf
    wav_np, sr = sf.read(path, dtype="float32")
    wav = torch.from_numpy(wav_np)
    if wav.dim() == 2:  # [T, C] from soundfile
        wav = wav.mean(dim=1)
    wav = wav.unsqueeze(0)  # [1, T]
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    return wav

_NEMO_PREPROCESSOR = None


def _get_nemo_preprocessor():
    """Lazily build (once per process) the exact feature front-end the
    pretrained FastConformer checkpoint was trained with. Must NOT reuse
    extract_fbank_features's Kaldi-style fbank: different mel-filterbank
    implementation and windowing (povey vs hann) would feed the frozen
    FastConformer encoder out-of-distribution features, confounding any
    "does FastConformer scale" comparison with an input-mismatch bug.
    Config values below are copied verbatim from
    stt_en_fastconformer_hybrid_large_streaming_480ms's cfg.preprocessor
    (dither forced to 0.0 for eval determinism; negligible vs. the
    checkpoint's own 1e-5)."""
    global _NEMO_PREPROCESSOR
    if _NEMO_PREPROCESSOR is None:
        from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
        _NEMO_PREPROCESSOR = AudioToMelSpectrogramPreprocessor(
            sample_rate=16000,
            window_size=0.025,
            window_stride=0.01,
            window="hann",
            normalize="NA",
            n_fft=512,
            features=80,
            dither=0.0,
            pad_to=0,
            frame_splicing=1,
        )
        _NEMO_PREPROCESSOR.eval()
    return _NEMO_PREPROCESSOR


def extract_nemo_fbank_features(waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """Extracts 80-dim log-mel features matching NeMo's FastConformer
    preprocessing exactly (see _get_nemo_preprocessor). Same [num_frames, 80]
    shape/convention as extract_fbank_features so it drops into the same
    dataset/collate pipeline; only the numeric content differs.

    Args:
        waveform: Tensor of shape (1, num_samples) or (num_samples,)
        sample_rate: Sampling rate, expected 16000

    Returns:
        Tensor of shape (num_frames, 80)
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    assert sample_rate == 16000, "NeMo preprocessor here is hardcoded for 16kHz"

    preprocessor = _get_nemo_preprocessor()
    length = torch.tensor([waveform.size(1)], dtype=torch.long)
    with torch.no_grad():
        feats, _ = preprocessor(input_signal=waveform, length=length)  # [1, 80, T]
    return feats.squeeze(0).transpose(0, 1).contiguous()  # [T, 80]


def extract_wavlm_waveform_features(waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """Returns a per-utterance zero-mean/unit-variance normalized RAW waveform
    for WavLM's own CNN feature encoder — WavLM (unlike the fbank-based
    backbones) does not take an external mel filterbank at all, so this is
    NOT a drop-in numeric variant of extract_fbank_features, it's a different
    kind of "feature" entirely (raw samples, not frames).

    Normalization matches HuggingFace's Wav2Vec2FeatureExtractor for
    microsoft/wavlm-large exactly (`do_normalize=True`): confirmed via
    `AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")`.
    HF's own `zero_mean_unit_var_norm` normalizes each array independently
    before batching/padding, so doing it here per-item (before this project's
    own padding in collate_fn) is numerically identical to what the
    checkpoint's own preprocessor would produce, not an approximation of it.

    Returns a 1-D tensor [T] of samples (not [T, 80] frames) — the dataset/
    collate pipeline only ever assumes `f.size(0)` is "this example's
    length" for sorting/padding, so a 1-D return is a legitimate signature
    here, not a special case; the model wrapper (WavLMBackbone) is the only
    place that interprets this tensor's meaning (samples vs. frames).

    Args:
        waveform: Tensor of shape (1, num_samples) or (num_samples,)
        sample_rate: Sampling rate, expected 16000

    Returns:
        Tensor of shape (num_samples,)
    """
    if waveform.dim() == 2:
        waveform = waveform.squeeze(0)
    assert sample_rate == 16000, "WavLM here is hardcoded for 16kHz"

    x = waveform.float()
    mean = x.mean()
    var = x.var(unbiased=False)
    x = (x - mean) / torch.sqrt(var + 1e-7)
    return x.contiguous()


def extract_fbank_features(waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """Extracts 80-dim log-mel filterbank features matching icefall.
    
    Args:
        waveform: Tensor of shape (1, num_samples) or (num_samples,)
        sample_rate: Sampling rate, expected 16000
        
    Returns:
        Tensor of shape (num_frames, 80)
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
        
    # Standard Kaldi fbank config used in icefall
    # 25ms window, 10ms shift, 80 mel bins
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_shift=10.0,
        frame_length=25.0,
        dither=0.0,
        energy_floor=1.0,  # icefall often uses energy_floor=1.0 for fbank? Or defaults. kaldi.fbank defaults are usually close enough.
        sample_frequency=sample_rate
    )
    return fbank


def feature_extractor_for_config(config: dict):
    """Single source of truth for "which fbank frontend does this config's
    backbone need". Every entrypoint that extracts features for a ZipCount
    config (training, validation, or any of the eval/probe/diagnosis scripts)
    must go through this instead of importing extract_fbank_features
    directly, or a FastConformer checkpoint silently gets fed out-of-
    distribution Kaldi-style features and produces a meaningless score with
    no error (this happened in an earlier draft of the FastConformer
    backbone-swap experiment, caught by review before any benchmark numbers
    were reported)."""
    encoder_type = config.get("model", {}).get("encoder", {}).get("type")
    if encoder_type == "fastconformer":
        return extract_nemo_fbank_features
    if encoder_type == "wavlm":
        return extract_wavlm_waveform_features
    return extract_fbank_features
