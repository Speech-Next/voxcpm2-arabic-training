"""Per-utterance metrics: ASR CER / WER, UTMOS, DNSMOS, WavLM-SV speaker similarity."""

from __future__ import annotations

import re
import unicodedata

import numpy as np

SR = 16000
WHISPER = "/workspace/models/faster-whisper-large-v3"
DNSMOS_DIR = "/workspace/models/dnsmos"
SV_MODEL = "microsoft/wavlm-base-plus-sv"
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")  # harakat, Quranic marks, tatweel


def normalize_ar(s: str) -> str:
    """For CER / WER: diacritics, tatweel and punctuation removed; alef / ya / ta-marbuta / hamza seats unified; ASR
    event tags (<hesitation>) dropped."""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"<[^>]*>", " ", s)
    s = _DIACRITICS.sub("", s)
    s = re.sub("[إأآٱ]", "ا", s)
    s = s.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    s = s.replace("ک", "ك").replace("ی", "ي")
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def error_rates(ref: str, hyp: str) -> dict:
    import jiwer

    r, h = normalize_ar(ref), normalize_ar(hyp) or " "
    return {"cer": jiwer.cer(r, h), "wer": jiwer.wer(r, h), "babble": len(h) > 2 * len(r) + 10}


class WhisperASR:
    name = "whisper-large-v3"

    def __init__(self, device: str = "cuda"):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(WHISPER, device=device, compute_type="float16")

    def __call__(self, path: str) -> str:
        segs, _ = self.model.transcribe(path, language="ar", beam_size=1, condition_on_previous_text=False)
        return " ".join(s.text for s in segs).strip()


class CohereASR:
    """Cohere Transcribe Arabic (2B Conformer encoder-decoder, dialect-trained): the second ASR, so that a change in one
    ASR's bias is not read as a change in the speech. (The lahgtna corpus's transcripts were checked with it.)"""

    name = "cohere-transcribe-arabic"
    REPO = "CohereLabs/cohere-transcribe-arabic-07-2026"

    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(self.REPO)
        self.model = CohereAsrForConditionalGeneration.from_pretrained(self.REPO, torch_dtype=torch.bfloat16).to(device).eval()

    def __call__(self, audio: np.ndarray) -> str:
        inputs = self.processor(audio, sampling_rate=SR, return_tensors="pt", language="ar")
        inputs.to(self.model.device, dtype=self.model.dtype)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=256)
        text = self.processor.decode(out, skip_special_tokens=True)  # a list: one text per (batch) input
        return (text[0] if isinstance(text, list) else text).strip()


class DNSMOS:
    """Microsoft DNSMOS (dnsmos_local.py logic), non-personalized P.835 + P.808."""

    INPUT_LENGTH = 9.01

    def __init__(self):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.p835 = ort.InferenceSession(f"{DNSMOS_DIR}/sig_bak_ovr.onnx", opts, providers=["CPUExecutionProvider"])
        self.p808 = ort.InferenceSession(f"{DNSMOS_DIR}/model_v8.onnx", opts, providers=["CPUExecutionProvider"])

    @staticmethod
    def _poly(sig, bak, ovr):
        p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        return p_sig(sig), p_bak(bak), p_ovr(ovr)

    @staticmethod
    def _melspec(audio):
        import librosa

        mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=321, hop_length=160, n_mels=120)
        return ((librosa.power_to_db(mel, ref=np.max) + 40) / 40).T

    def __call__(self, audio: np.ndarray) -> dict:
        n = int(self.INPUT_LENGTH * SR)
        while len(audio) < n:
            audio = np.append(audio, audio)
        rows = []
        for i in range(int(np.floor(len(audio) / SR) - self.INPUT_LENGTH) + 1):
            seg = audio[int(i * SR) : int((i + self.INPUT_LENGTH) * SR)]
            if len(seg) < n:
                continue
            sig, bak, ovr = self.p835.run(None, {"input_1": seg.astype("float32")[np.newaxis, :]})[0][0]
            p808 = self.p808.run(None, {"input_1": self._melspec(seg[:-160]).astype("float32")[np.newaxis]})[0][0][0]
            rows.append((*self._poly(sig, bak, ovr), p808))
        sig, bak, ovr, p808 = np.mean(rows, axis=0)
        return {"dnsmos_sig": float(sig), "dnsmos_bak": float(bak), "dnsmos_ovrl": float(ovr), "dnsmos_p808": float(p808)}


class UTMOS:
    def __init__(self, device: str = "cuda"):
        import torch

        self.torch, self.device = torch, device
        self.model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()

    def __call__(self, audio: np.ndarray) -> float:
        with self.torch.inference_mode():
            return float(self.model(self.torch.from_numpy(audio).unsqueeze(0).to(self.device), SR).item())


class SpeakerEmbedder:
    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.torch, self.device = torch, device
        self.fe = AutoFeatureExtractor.from_pretrained(SV_MODEL)
        self.model = WavLMForXVector.from_pretrained(SV_MODEL).to(device).eval()

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        inputs = self.fe(audio, sampling_rate=SR, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            emb = self.model(**inputs).embeddings[0]
        return self.torch.nn.functional.normalize(emb, dim=-1).float().cpu().numpy()


def load16k(path: str) -> np.ndarray:
    import librosa

    return librosa.load(path, sr=SR)[0]
