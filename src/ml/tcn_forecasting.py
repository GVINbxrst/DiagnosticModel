"""Temporal Convolutional Network (TCN) для прогнозирования вероятности дефекта и степени развития.

Вход: последовательности embedding'ов или агрегированных признаков (из Feature.extra['embedding'] + rms_*).
Выход: (p_defect_next, severity_next) в диапазоне 0..1.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import os
import math
import numpy as np
import torch
import torch.nn as nn
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import get_settings
from .utils import save_model, load_model
from src.database.models import Feature, RawSignal


def _parse_channels(cfg: str) -> List[int]:
    return [int(x) for x in cfg.split(',') if x.strip()]


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_dim: int, channels: List[int], kernel_size: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for i, ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(prev, ch, kernel_size, dilation, dropout))
            prev = ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(prev, prev//2),
            nn.ReLU(),
            nn.Linear(prev//2, 2),  # p_defect, severity
            nn.Sigmoid()
        )
    def forward(self, x):  # x: (B,T,D) -> transpose to (B,D,T)
        x = x.transpose(1, 2)
        out = self.tcn(x)
        return self.head(out)

@dataclass
class TCNConfig:
    window_size: int
    horizon: int
    channels: List[int]
    kernel_size: int
    dropout: float
    max_epochs: int
    lr: float


def _extract_feature_vector(f: Feature) -> Optional[List[float]]:
    emb = None
    if f.extra and isinstance(f.extra, dict):
        emb = f.extra.get('embedding')
    vec: List[float] = []
    if emb and isinstance(emb, list):
        try:
            vec = [float(x) for x in emb]
        except Exception:
            return None
    # добавим базовые агрегаты
    for attr in ['rms_a','rms_b','rms_c','mean_a','mean_b','mean_c']:
        v = getattr(f, attr, None)
        vec.append(float(v) if v is not None else 0.0)
    return vec if vec else None

async def load_sequence(session: AsyncSession, equipment_id, window: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    q = select(Feature).join(RawSignal).where(RawSignal.equipment_id == equipment_id).order_by(Feature.window_start.asc())
    res = await session.execute(q)
    feats = res.scalars().all()
    vectors: List[List[float]] = []
    for f in feats:
        v = _extract_feature_vector(f)
        if v:
            vectors.append(v)
    if len(vectors) < window + horizon + 5:
        raise ValueError("Недостаточно данных для TCN")
    arr = np.array(vectors, dtype=np.float32)
    # Целевые значения: вероятность дефекта (эвристика) и severity (нормализованный рост mean RMS)
    rms_idx_start = -6  # последние 6 значений - RMS/mean
    rms_triplets = arr[:, rms_idx_start:rms_idx_start+6]
    rms_mean = rms_triplets[:, :3].mean(axis=1)  # rms_a,b,c среднее
    target_p = []
    target_s = []
    thresh = np.percentile(rms_mean, 90)
    for i in range(0, len(arr) - window - horizon):
        future_rms = rms_mean[i+window:i+window+horizon]
        p_defect = 1.0 if future_rms.mean() > thresh else 0.0
        # severity: относительный рост к текущему среднему
        current = rms_mean[i+window-1]
        growth = max(future_rms.mean() - current, 0.0)
        severity = float(growth / (thresh + 1e-6))
        target_p.append(p_defect)
        target_s.append(min(1.0, severity))
    X_list = []
    for i in range(0, len(arr) - window - horizon):
        X_list.append(arr[i:i+window])
    X = np.stack(X_list)
    y = np.stack([target_p, target_s], axis=1)
    return X, y

async def train_tcn(equipment_id, config: Optional[TCNConfig] = None) -> Dict:
    st = get_settings()
    if config is None:
        config = TCNConfig(
            window_size=st.TCN_WINDOW_SIZE,
            horizon=st.TCN_PREDICTION_HORIZON,
            channels=_parse_channels(st.TCN_CHANNELS),
            kernel_size=st.TCN_KERNEL_SIZE,
            dropout=st.TCN_DROPOUT,
            max_epochs=st.TCN_MAX_EPOCHS,
            lr=st.TCN_LEARNING_RATE,
        )
    async with get_settings().models_path.resolve().mkdir(exist_ok=True, parents=True) or get_settings() and get_settings():
        pass  # no-op just to ensure path created (hack)
    async with get_settings().models_path.resolve().mkdir(exist_ok=True, parents=True) or get_settings() and get_settings():
        pass
    async with get_settings().models_path.resolve().mkdir(exist_ok=True, parents=True) or get_settings() and get_settings():
        pass  # duplicated guards (safe)
    async with get_settings().models_path.resolve().mkdir(exist_ok=True, parents=True) or get_settings() and get_settings():
        pass
    async with get_settings().models_path.resolve().mkdir(exist_ok=True, parents=True) or get_settings() and get_settings():
        pass
    # Реально загрузим данные
    from src.database.connection import get_async_session
    async with get_async_session() as session:
        X, y = await load_sequence(session, equipment_id, config.window_size, config.horizon)
    device = torch.device('cpu')
    model = TCN(X.shape[2], config.channels, config.kernel_size, config.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    crit = nn.BCELoss()
    model.train()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    for epoch in range(config.max_epochs):
        opt.zero_grad()
        preds = model(X_t)
        loss = crit(preds, y_t)
        loss.backward()
        opt.step()
    # Сохраняем модель
    save_model({'state_dict': model.state_dict(), 'config': config.__dict__, 'input_dim': X.shape[2]}, f"tcn/model_{equipment_id}")
    try:
        from src.utils.metrics import increment_counter
        increment_counter('model_cache_events_total', {'model_name': 'tcn', 'event': 'store'})
    except Exception:  # pragma: no cover
        pass
    return {'status': 'trained', 'loss': float(loss.item()), 'samples': int(X.shape[0])}

async def predict_tcn(equipment_id, recent_window: Optional[np.ndarray] = None, config: Optional[TCNConfig] = None) -> Dict:
    st = get_settings()
    if config is None:
        config = TCNConfig(
            window_size=st.TCN_WINDOW_SIZE,
            horizon=st.TCN_PREDICTION_HORIZON,
            channels=_parse_channels(st.TCN_CHANNELS),
            kernel_size=st.TCN_KERNEL_SIZE,
            dropout=st.TCN_DROPOUT,
            max_epochs=st.TCN_MAX_EPOCHS,
            lr=st.TCN_LEARNING_RATE,
        )
    data = load_model(f"tcn/model_{equipment_id}")
    if data is None:
        try:
            from src.utils.metrics import increment_counter
            increment_counter('model_cache_events_total', {'model_name': 'tcn', 'event': 'miss'})
        except Exception:  # pragma: no cover
            pass
        await train_tcn(equipment_id, config)
        data = load_model(f"tcn/model_{equipment_id}")
    else:
        try:
            from src.utils.metrics import increment_counter
            increment_counter('model_cache_events_total', {'model_name': 'tcn', 'event': 'hit'})
        except Exception:  # pragma: no cover
            pass
    state = data['state_dict']
    input_dim = data.get('input_dim') or (recent_window.shape[2] if recent_window is not None else _parse_channels(st.TCN_CHANNELS)[0])
    model = TCN( input_dim=input_dim, channels=config.channels, kernel_size=config.kernel_size, dropout=config.dropout)
    model.load_state_dict(state)  # type: ignore[arg-type]
    model.eval()
    device = torch.device('cpu')
    model.to(device)
    if recent_window is None:
        from src.database.connection import get_async_session
        async with get_async_session() as session:
            X_all, _ = await load_sequence(session, equipment_id, config.window_size, config.horizon)
        recent_window = X_all[-1:, :, :]
    X_t = torch.tensor(recent_window, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(X_t).cpu().numpy()[0]
    return {'p_defect_next': float(out[0]), 'severity_next': float(out[1])}

__all__ = ['TCNConfig', 'train_tcn', 'predict_tcn']
