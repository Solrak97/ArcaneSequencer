from pathlib import Path

import torch
import torchaudio
import bigvgan

from src.utils.data import create_dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


model = bigvgan.BigVGAN.from_pretrained(
    "nvidia/bigvgan_v2_22khz_80band_fmax8k_256x",
    use_cuda_kernel=False,
    proxies=None,
    resume_download=False,
).to(device)

model.remove_weight_norm()
model.eval()

h = model.h  # this is what we pass to the dataset

# 2) Dataloader that uses BigVGAN-compatible mels
loader = create_dataloader(
    root_dir="./data/wav",      # your folder with audio
    batch_size=1,
    segment_seconds=15.0,
    sample_rate=h.sampling_rate,
    n_fft=h.n_fft,
    hop_length=h.hop_size,
    n_mels=h.num_mels,
    return_waveform=True,
    num_workers=0,
    use_bigvgan_mel=True,
    bigvgan_h=h,
    device=device,
)

batch = next(iter(loader))
mel = batch["mel"]          # [B, 1, num_mels, T]
wav = batch["waveform"]     # [B, 1, segment_samples]
print("mel shape:", mel.shape)
print("waveform shape:", wav.shape)

# 3) Run through BigVGAN
mel_for_vocoder = mel.squeeze(1).to(device)  # [B, num_mels, T]

with torch.inference_mode():
    audio_hat = model(mel_for_vocoder)       # [B, 1, samples]

out_sr = h.sampling_rate
torchaudio.save("outputs/bigvgan_original_chunk.wav", wav[0].cpu(), out_sr)
torchaudio.save("outputs/bigvgan_from_mel.wav", audio_hat[0].cpu(), out_sr)
print("Saved bigvgan_original_chunk.wav and bigvgan_from_mel.wav")