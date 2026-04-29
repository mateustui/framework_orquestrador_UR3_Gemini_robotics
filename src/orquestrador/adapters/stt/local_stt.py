from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class LocalSTT:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,
        vad_filter: bool = False,
    ):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Dependencia ausente para STT local. Instale: pip install '.[stt]'"
            ) from exc

        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._beam_size = max(1, int(beam_size))
        self._vad_filter = bool(vad_filter)

    def transcribe_array(self, audio: "np.ndarray", sample_rate: int, language: str = "pt") -> str:
        if sample_rate != 16000:
            raise RuntimeError("MIC_SAMPLE_RATE deve ser 16000 para transcricao rapida sem resample.")

        segments, _ = self._model.transcribe(
            audio,
            language=language,
            beam_size=self._beam_size,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=False,
            vad_filter=self._vad_filter,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
