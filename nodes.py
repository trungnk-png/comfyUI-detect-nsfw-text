"""
Detect NSFW Text — ComfyUI node.

String input goes to the Qwen prompt guard. The prompt policy is unchanged.
"""

import json
import re
import threading

import comfy.model_management
import comfy.sd
import folder_paths


DEFAULT_CLIP = "qwen3vl_8b_int8_convrot.safetensors"
_cached_clip = None
_cached_path = None
_load_lock = threading.Lock()

_PROMPT_POLICY = """Evaluate this prompt for workplace safety. Prompt must be in English (otherwise VERDICT: NOT_ENGLISH).
Safety policy: NSFW = nudity, sexual requests, erotic content, exposed body parts; clothed people/everyday scenes = SAFE.
Reply with exactly two lines:
VERDICT: SAFE, VERDICT: NSFW, or VERDICT: NOT_ENGLISH
REASON: brief explanation.
"""


class NSFWDetectedError(RuntimeError):
    pass


class UnsupportedLanguageError(ValueError):
    pass


def _load_clip(clip_name):
    global _cached_clip, _cached_path
    path = folder_paths.get_full_path("text_encoders", clip_name)
    if path is None:
        path = folder_paths.get_full_path("clip", clip_name)
    if path is None:
        raise FileNotFoundError(f"Qwen NSFW Guard: text encoder not found: {clip_name}")
    with _load_lock:
        if _cached_clip is None or _cached_path != path:
            loaded = comfy.sd.load_clip(ckpt_paths=[path])
            _cached_clip, _cached_path = loaded, path
        return _cached_clip


def _check(clip, query, source):
    tokens = clip.tokenize(query)
    generated = clip.generate(tokens, do_sample=False, max_length=32, temperature=0.0)
    response = clip.tokenizer.decode(generated).strip()
    verdicts = re.findall(r"^\s*VERDICT:\s*(NSFW|SAFE|NOT[ _]ENGLISH|UNSUPPORTED[ _]LANGUAGE)\b", response, re.MULTILINE | re.IGNORECASE)
    if any(re.match(r"^(NOT[ _]ENGLISH|UNSUPPORTED[ _]LANGUAGE)$", v, re.IGNORECASE) for v in verdicts):
        comfy.model_management.interrupt_current_processing(True)
        raise UnsupportedLanguageError("language not supported, only support english at this time")
    if any(verdict.upper() == "NSFW" for verdict in verdicts):
        comfy.model_management.interrupt_current_processing(True)
        raise NSFWDetectedError(f"Qwen NSFW Guard blocked {source}: {response}")
    if len(verdicts) != 1 or verdicts[0].upper() != "SAFE":
        raise RuntimeError(f"Qwen NSFW Guard could not classify {source}: {response!r}")
    return {"source": source, "verdict": "SAFE", "response": response}


def _moderate_prompt(clip_name, clip=None, prompt=None):
    if prompt is None or not str(prompt).strip():
        raise ValueError("Qwen NSFW Guard requires a prompt.")
    clip = clip if clip is not None else _load_clip(clip_name)
    checks = [
        _check(
            clip,
            _PROMPT_POLICY + "\nClassify this prompt (JSON string):\n" + json.dumps(prompt),
            "prompt",
        )
    ]
    return json.dumps({"verdict": "SAFE", "checks": checks}, ensure_ascii=False)


def _log_error(exc):
    print(f"[Detect NSFW Text] {exc}", flush=True)


class DetectNSFWText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "clip_name": ("STRING", {"default": DEFAULT_CLIP}),
            },
            "optional": {
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "report")
    FUNCTION = "check"
    CATEGORY = "safety"
    OUTPUT_NODE = True

    def check(self, prompt, clip_name=DEFAULT_CLIP, clip=None):
        try:
            report = _moderate_prompt(clip_name, clip=clip, prompt=prompt)
        except (NSFWDetectedError, UnsupportedLanguageError, RuntimeError, FileNotFoundError, ValueError) as exc:
            _log_error(exc)
            raise
        return {"ui": {"text": [report]}, "result": (prompt, report)}


NODE_CLASS_MAPPINGS = {
    "DetectNSFWText": DetectNSFWText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DetectNSFWText": "Detect NSFW Text",
}
