# data.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa


class MusicChunkDataset(Dataset):
    """
    Loads audio files and returns fixed-length waveform chunks and mel spectrograms.

    If `use_bigvgan_mel=True`, mel spectrograms are computed using BigVGAN's
    mel_spectrogram config (via bigvgan.mel_spectrogram), so they match exactly
    what the BigVGAN vocoder expects.
    """

    def __init__(
        self,
        root_dir: str | Path,
        sample_rate: int = 22050,
        segment_seconds: float = 2.0,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        mono: bool = True,
        return_waveform: bool = True,
        extensions: Optional[List[str]] = None,
        # BigVGAN-specific options
        use_bigvgan_mel: bool = False,
        bigvgan_h: Any | None = None,
        device: str | torch.device = "cpu",
    ) -> None:

        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.segment_seconds = segment_seconds
        self.segment_samples = int(segment_seconds * sample_rate)
        self.mono = mono
        self.return_waveform = return_waveform
        self.use_bigvgan_mel = use_bigvgan_mel
        self.bigvgan_h = bigvgan_h
        self.device = torch.device(device)

        if extensions is None:
            extensions = [".wav", ".flac", ".mp3", ".ogg", ".m4a"]
        self.extensions = {ext.lower() for ext in extensions}

        # Find all files
        self.files: List[Path] = []
        for path in self.root_dir.rglob("*"):
            if path.suffix.lower() in self.extensions:
                self.files.append(path)

        if not self.files:
            raise ValueError(f"No audio files found in {self.root_dir}")

        if self.use_bigvgan_mel:
            # We will use bigvgan.mel_spectrogram in _wav_to_logmel
            if self.bigvgan_h is None:
                raise ValueError(
                    "use_bigvgan_mel=True but bigvgan_h is None. "
                    "Pass model.h from a loaded BigVGAN model."
                )
            # Ensure dataset sample_rate matches BigVGAN's expectation
            self.sample_rate = int(self.bigvgan_h.sampling_rate)
        else:
            # Standard torchaudio mel config (your original settings)
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                n_mels=n_mels,
                center=True,
                pad_mode="reflect",
                power=2.0,  # magnitude^2
            )
            self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def __len__(self) -> int:
        return len(self.files)

    def _load_audio(self, path: Path) -> torch.Tensor:
        """
        Load audio using librosa instead of torchaudio.load, to avoid torchcodec.
        Returns [1, num_samples] (mono) or [C, num_samples].
        """
        y, sr = librosa.load(
            path,
            sr=self.sample_rate,   # resample to dataset sample_rate
            mono=self.mono,        # True -> [N], False -> [C, N]
        )
        if self.mono:
            y = torch.from_numpy(y).unsqueeze(0)  # [1, N]
        else:
            y = torch.from_numpy(y)               # [C, N]
        return y

    def _random_chunk(self, waveform: torch.Tensor) -> torch.Tensor:
        num_samples = waveform.shape[-1]

        if num_samples < self.segment_samples:
            pad_amount = self.segment_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            return waveform

        max_start = num_samples - self.segment_samples
        start = torch.randint(low=0, high=max_start + 1, size=(1,)).item()
        end = start + self.segment_samples
        return waveform[..., start:end]

    def _wav_to_logmel(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: [1, num_samples]

        If use_bigvgan_mel:
            - Uses bigvgan.mel_spectrogram with bigvgan_h hyperparams.
            - Returns mel in the same scale BigVGAN expects (no extra dB transform).

        Else:
            - Uses torchaudio MelSpectrogram + AmplitudeToDB (your original pipeline).
        """
        if self.use_bigvgan_mel:
            import bigvgan

            # move to device expected by bigvgan.mel_spectrogram
            wav = wav.to(self.device)

            h = self.bigvgan_h
            mel = bigvgan.mel_spectrogram(
                wav,
                n_fft=h.n_fft,
                num_mels=h.num_mels,
                sampling_rate=h.sampling_rate,
                hop_size=h.hop_size,
                win_size=h.win_size,
                fmin=h.fmin,
                fmax=h.fmax,
                center=False,
            )
            # mel: [1, num_mels, T]
            return mel.cpu()

        # Your original torchaudio-based mel
        mel = self.mel_transform(wav)      # [1, n_mels, T]
        mel_db = self.db_transform(mel)    # [1, n_mels, T]
        return mel_db

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx % len(self.files)]
        waveform = self._load_audio(path)          # [1, N] or [C, N]
        chunk = self._random_chunk(waveform)       # [1, segment_samples]
        mel = self._wav_to_logmel(chunk)           # [1, n_mels, T]

        sample: Dict[str, Any] = {
            "mel": mel,
            "path": str(path),
        }

        if self.return_waveform:
            sample["waveform"] = chunk

        return sample


def create_dataloader(
    root_dir: str | Path,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    **dataset_kwargs: Any,
) -> DataLoader:

    dataset = MusicChunkDataset(root_dir=root_dir, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
