from __future__ import annotations
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseGenerator(ABC):
    name: str = "base_generator"

    def __init__(self):
        self._is_fitted: bool = False
        self._n_snps: int | None = None

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseGenerator":
        raise NotImplementedError

    @abstractmethod
    def generate(self, n_samples: int) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        self._check_is_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, path: str | Path) -> "BaseGenerator":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No saved generator state at path {path}")
        with open(path, "rb") as f:
            state = pickle.load(f)

        for key, value in state.items():
            setattr(self, key, value)

        return self

    def _mark_fitted(self, n_snps: int) -> None:
        self._is_fitted = True
        self._n_snps = n_snps

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} has not been trained yet. Call fit(X).")

    def _validate_output_shape(self, X: np.ndarray, n_samples: int) -> None:
        expected = (n_samples, self._n_snps)
        if X.shape != expected:
            raise ValueError(
                f"{self.name}.generate() produced shape {X.shape}, expected {expected}"
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return f"<{self.__class__.__name__} name={self.name!r} status={status}>"
