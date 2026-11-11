"""Regras de confiança para CSVs FIRMS históricos e NRT."""
from __future__ import annotations

import pandas as pd

from .ingest import validation_masks


def firms_validation_masks(frame: pd.DataFrame, product: str) -> dict[str, pd.Series]:
    masks = validation_masks(frame, product)
    if product == "viirs-jpss1" and "confidence" in frame:
        # O histórico contém l/n/h; o arquivo NRT usa low/nominal/high.
        confidence = frame.confidence.astype(str).str.strip().str.lower()
        masks["confidence_invalid"] = ~confidence.isin({"l", "n", "h", "low", "nominal", "high"})
    return masks
