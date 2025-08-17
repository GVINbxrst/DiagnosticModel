"""Advanced feature engineering & embeddings for raw current phases.

Provides:
 - compute_base_features: RMS, FFT magnitude, top harmonics, spectral entropy
 - Autoencoder1D: simple 1D CNN autoencoder for embeddings
 - EmbeddingExtractor: orchestrates training & inference, stores embedding vectors

Embedding shape kept modest (e.g. 32) to fit JSONB storage in Feature.extra['embedding'].
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from src.utils.logger import get_logger
from src.config.settings import get_settings
import json, hashlib, os, inspect, datetime, joblib

logger = get_logger(__name__)

# ---------- Hand‑crafted features ----------

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0

def spectral_entropy(mag: np.ndarray, eps: float = 1e-12) -> float:
    p = mag / (mag.sum() + eps)
    ent = -(p * np.log2(p + eps)).sum()
    return float(ent / math.log2(len(p)+1e-9))  # normalized

def top_harmonics(signal: np.ndarray, sample_rate: int, n: int = 5) -> List[Tuple[float,float]]:
    if signal.size == 0:
        return []
    # next pow2
    n_fft = 1 << (signal.size - 1).bit_length()
    fft_vals = np.fft.rfft(signal, n_fft)
    mag = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(n_fft, d=1.0/sample_rate)
    # exclude DC for harmonics ranking
    idx = np.argsort(mag[1:])[-n:][::-1] + 1 if len(mag) > 1 else []
    return [(float(freqs[i]), float(mag[i])) for i in idx]

def compute_base_features(phase_a: Optional[np.ndarray], phase_b: Optional[np.ndarray], phase_c: Optional[np.ndarray], sample_rate: int) -> Dict:
    phases = {'a': phase_a, 'b': phase_b, 'c': phase_c}
    out: Dict[str, Dict] = {}
    for k, arr in phases.items():
        if arr is None or arr.size == 0:
            out[k] = None
            continue
        # detrend simple
        x = arr - np.mean(arr)
        rms_v = rms(x)
        n_fft = 1 << (x.size - 1).bit_length()
        spec = np.abs(np.fft.rfft(x, n_fft))
        entropy = spectral_entropy(spec)
        harms = top_harmonics(x, sample_rate, n=5)
        out[k] = {
            'rms': rms_v,
            'spectral_entropy': entropy,
            'harmonics': harms[:5],
        }
    return out

# ---------- Autoencoder (PyTorch) ----------
try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False

EMBED_DIM = 32

class Autoencoder1D(nn.Module):  # type: ignore
    def __init__(self, emb_dim: int = EMBED_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(emb_dim)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 4, stride=2, padding=1)
        )

    def forward(self, x):  # x: (B,1,L)
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

@dataclass
class EmbeddingResult:
    embedding: List[float]
    loss: float

class EmbeddingExtractor:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.model: Optional[Autoencoder1D] = None
        self.device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'

    def _prepare_model(self):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch недоступен – установите torch для автоэнкодера.")
        if self.model is None:
            self.model = Autoencoder1D().to(self.device)
            # Попытка загрузки сохранённых весов
            try:
                st = get_settings()
                weights = st.models_path / 'embeddings' / 'autoencoder_weights.pt'
                if weights.exists():
                    self.model.load_state_dict(torch.load(weights, map_location=self.device))  # type: ignore
                    logger.info("Загружены кешированные веса автоэнкодера")
            except Exception as e:  # pragma: no cover
                logger.debug(f"Не удалось загрузить веса автоэнкодера: {e}")
        return self.model

    def train_autoencoder(self, signals: List[np.ndarray], epochs: int = 3, batch_size: int = 16) -> None:
        model = self._prepare_model()
        model.train()
        # pad / trim to max len (power of two cap 4096)
        max_len = min(4096, max(len(s) for s in signals))
        proc = []
        for s in signals:
            x = s[:max_len]
            if len(x) < max_len:
                x = np.pad(x, (0, max_len-len(x)))
            proc.append(x.astype(np.float32))
        data = torch.tensor(proc).unsqueeze(1)  # (N,1,L)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        for ep in range(epochs):
            perm = torch.randperm(data.size(0))
            total = 0.0
            for i in range(0, data.size(0), batch_size):
                idx = perm[i:i+batch_size]
                batch = data[idx].to(self.device)
                optimizer.zero_grad()
                recon, _ = model(batch)
                # match shapes (decoder may overshoot length) – crop
                if recon.size(-1) != batch.size(-1):
                    recon = recon[..., :batch.size(-1)]
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                total += loss.item() * batch.size(0)
            logger.info(f"Autoencoder epoch {ep+1}: loss={total/len(data):.6f}")
        try:
            self._write_manifest(total/len(data))
            # Сохраняем веса для повторного использования
            try:
                st = get_settings()
                out_dir = st.models_path / 'embeddings'
                os.makedirs(out_dir, exist_ok=True)
                torch.save(model.state_dict(), out_dir / 'autoencoder_weights.pt')  # type: ignore
            except Exception as se:  # pragma: no cover
                logger.debug(f"save ae weights failed: {se}")
        except Exception as e:  # pragma: no cover
            logger.warning(f"Не удалось записать embedding manifest: {e}")

    def embed_signal(self, signal: np.ndarray) -> EmbeddingResult:
        if self.model is None:
            raise ValueError("Модель автоэнкодера не обучена")
        self.model.eval()
        with torch.no_grad():  # type: ignore
            L = min(4096, len(signal))
            x = signal[:L]
            if len(x) < L:
                x = np.pad(x, (0, L-len(x)))
            tens = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            recon, z = self.model(tens)
            if recon.size(-1) != tens.size(-1):
                recon = recon[..., :tens.size(-1)]
            loss = torch.nn.functional.mse_loss(recon, tens).item()
            # global average over channels & sequence
            emb = z.mean(dim=-1).squeeze().cpu().numpy()  # (64) before pool, but after encoder it's (64, emb_dim) -> we pooled
            emb_vec = emb.flatten()[:EMBED_DIM]
            return EmbeddingResult(embedding=[float(v) for v in emb_vec], loss=loss)

    def _write_manifest(self, final_loss: float):
        st = get_settings()
        model_dir = st.models_path / 'embeddings'
        os.makedirs(model_dir, exist_ok=True)
        code_src = inspect.getsource(Autoencoder1D)
        code_hash = hashlib.sha256(code_src.encode('utf-8')).hexdigest()[:12]
        manifest = {
            'model_family': 'autoencoder1d',
            'embedding_dim': EMBED_DIM,
            'sample_rate': self.sample_rate,
            'device': self.device,
            'code_hash': code_hash,
            'trained_at': datetime.datetime.utcnow().isoformat() + 'Z',
            'final_loss': final_loss,
            'version': datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
        }
        (model_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

__all__ = [
    'compute_base_features', 'EmbeddingExtractor', 'Autoencoder1D', 'EmbeddingResult'
]
