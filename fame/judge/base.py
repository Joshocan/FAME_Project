from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class JudgeClient:
    model: str
    base_url: str
    api_key_env: str
    temperature: float
    max_tokens: int
    timeout_s: int

    def _get_api_key(self) -> str:
        return os.getenv(self.api_key_env, "").strip()

    def generate(self, prompt: str, *, system: Optional[str] = None, temperature: Optional[float] = None) -> str:
        raise NotImplementedError
