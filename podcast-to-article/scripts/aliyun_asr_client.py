from __future__ import annotations

import json
import mimetypes
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from utils import build_youtube_timestamp_url, resolve_setting
except ImportError:  # pragma: no cover - package import path
    from .utils import build_youtube_timestamp_url, resolve_setting


class AliyunAsrError(RuntimeError):
    """Raised when Aliyun DashScope ASR cannot produce a usable transcript."""


@dataclass
class AliyunAsrConfig:
    api_key: str
    model: str = "fun-asr"
    api_base: str = "https://dashscope.aliyuncs.com/api/v1"
    poll_interval_sec: float = 3.0
    timeout_sec: int = 900
    language_hints: tuple[str, ...] = ("zh", "en")
    yt_dlp_bin: str = "yt-dlp"
    audio_dir: Optional[Path] = None
    keep_audio: bool = False


def _bool_setting(value: Optional[str], *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def parse_aliyun_asr_config(start_path: Optional[Path] = None) -> Optional[AliyunAsrConfig]:
    api_key = resolve_setting(
        ("ALIYUN_API_KEY", "DASHSCOPE_API_KEY"),
        start_path=start_path,
    )
    if not api_key:
        return None

    model = resolve_setting(("ALIYUN_ASR_MODEL",), start_path=start_path) or "fun-asr"
    api_base = resolve_setting(("ALIYUN_ASR_API_BASE",), start_path=start_path) or "https://dashscope.aliyuncs.com/api/v1"
    poll_interval = resolve_setting(("ALIYUN_ASR_POLL_INTERVAL_SEC",), start_path=start_path)
    timeout = resolve_setting(("ALIYUN_ASR_TIMEOUT_SEC",), start_path=start_path)
    language_hints = resolve_setting(("ALIYUN_ASR_LANGUAGE_HINTS",), start_path=start_path)
    yt_dlp_bin = resolve_setting(("YT_DLP_BIN",), start_path=start_path) or "yt-dlp"
    audio_dir = resolve_setting(("ALIYUN_ASR_AUDIO_DIR",), start_path=start_path)
    keep_audio = _bool_setting(resolve_setting(("ALIYUN_ASR_KEEP_AUDIO",), start_path=start_path), default=False)

    hints = tuple(part.strip() for part in (language_hints or "zh,en").split(",") if part.strip())
    return AliyunAsrConfig(
        api_key=api_key,
        model=model,
        api_base=api_base.rstrip("/"),
        poll_interval_sec=float(poll_interval) if poll_interval else 3.0,
        timeout_sec=int(timeout) if timeout else 900,
        language_hints=hints or ("zh", "en"),
        yt_dlp_bin=yt_dlp_bin,
        audio_dir=Path(audio_dir).expanduser() if audio_dir else None,
        keep_audio=keep_audio,
    )


def aliyun_asr_is_enabled(start_path: Optional[Path] = None) -> bool:
    backend = resolve_setting(("TRANSCRIBER_BACKEND",), start_path=start_path)
    fallback = resolve_setting(
        ("TRANSCRIBER_FALLBACK_BACKEND", "TRANSCRIPT_FALLBACK_BACKEND"),
        start_path=start_path,
    )
    fallback_enabled = _bool_setting(resolve_setting(("ALIYUN_ASR_FALLBACK",), start_path=start_path), default=True)
    if backend == "aliyun_asr" or fallback == "aliyun_asr":
        return True
    return fallback_enabled and parse_aliyun_asr_config(start_path) is not None


def aliyun_asr_is_preferred(start_path: Optional[Path] = None) -> bool:
    return resolve_setting(("TRANSCRIBER_BACKEND",), start_path=start_path) == "aliyun_asr"


def _request_json(
    url: str,
    *,
    method: str,
    headers: dict[str, str],
    payload: Optional[dict[str, Any]] = None,
    timeout: int = 60,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise AliyunAsrError(f"Aliyun ASR HTTP error {exc.code}: {detail}") from exc
    except URLError as exc:
        raise AliyunAsrError(f"Failed to reach Aliyun ASR: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AliyunAsrError(f"Aliyun ASR returned non-JSON response from {url}") from exc


def _request_bytes(
    url: str,
    *,
    method: str,
    headers: dict[str, str],
    data: bytes,
    timeout: int = 300,
) -> bytes:
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise AliyunAsrError(f"Aliyun ASR upload HTTP error {exc.code}: {detail}") from exc
    except URLError as exc:
        raise AliyunAsrError(f"Failed to upload Aliyun ASR input file: {exc}") from exc


def _multipart_form_data(
    fields: dict[str, str],
    *,
    file_field: str,
    file_path: Path,
) -> tuple[str, bytes]:
    boundary = f"----pod2article-{time.time_ns()}"
    body = bytearray()

    def add(value: bytes) -> None:
        body.extend(value)
        body.extend(b"\r\n")

    for name, value in fields.items():
        add(f"--{boundary}".encode("utf-8"))
        add(f'Content-Disposition: form-data; name="{name}"'.encode("utf-8"))
        add(b"")
        add(str(value).encode("utf-8"))

    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    add(f"--{boundary}".encode("utf-8"))
    add(
        (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{file_path.name}"'
        ).encode("utf-8")
    )
    add(f"Content-Type: {content_type}".encode("utf-8"))
    add(b"")
    body.extend(file_path.read_bytes())
    body.extend(b"\r\n")
    add(f"--{boundary}--".encode("utf-8"))
    return f"multipart/form-data; boundary={boundary}", bytes(body)


def _coerce_ms(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.isdigit():
            return int(cleaned)
    return None


def _extract_sentences(payload: Any) -> list[dict[str, Any]]:
    sentences: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        maybe_sentences = payload.get("sentences")
        if isinstance(maybe_sentences, list):
            sentences.extend(item for item in maybe_sentences if isinstance(item, dict))
        for key in ("transcripts", "transcription", "results", "result", "output"):
            value = payload.get(key)
            if isinstance(value, (dict, list)):
                sentences.extend(_extract_sentences(value))
    elif isinstance(payload, list):
        for item in payload:
            sentences.extend(_extract_sentences(item))
    return sentences


def _first_ms(*values: Any) -> Optional[int]:
    for value in values:
        coerced = _coerce_ms(value)
        if coerced is not None:
            return coerced
    return None


def normalize_aliyun_asr_payload(payload: dict[str, Any], *, video_id: str) -> dict[str, Any]:
    transcript: list[dict[str, Any]] = []
    for sentence in _extract_sentences(payload):
        text = str(sentence.get("text") or sentence.get("sentence") or "").strip()
        if not text:
            continue
        start_ms = _first_ms(
            sentence.get("begin_time"),
            sentence.get("start_time"),
            sentence.get("start"),
            sentence.get("beginTime"),
            sentence.get("startTime"),
        )
        end_ms = _first_ms(
            sentence.get("end_time"),
            sentence.get("end"),
            sentence.get("endTime"),
        )
        if start_ms is None:
            continue
        item: dict[str, Any] = {
            "snippet": text,
            "start_ms": start_ms,
            "start_sec": start_ms // 1000,
            "url": build_youtube_timestamp_url(video_id, start_ms // 1000),
        }
        if end_ms is not None:
            item["end_ms"] = end_ms
            item["end_sec"] = end_ms // 1000
        transcript.append(item)

    if not transcript:
        raise AliyunAsrError("Aliyun ASR result did not include timestamped sentence segments.")

    transcript.sort(key=lambda item: int(item["start_ms"]))
    return {
        "provider": "aliyun_asr",
        "engine": "dashscope_asr",
        "transcript": transcript,
        "raw_result": payload,
    }


class AliyunAsrClient:
    def __init__(self, config: AliyunAsrConfig) -> None:
        self.config = config

    @classmethod
    def from_environment(cls, start_path: Optional[Path] = None) -> "AliyunAsrClient":
        config = parse_aliyun_asr_config(start_path)
        if config is None:
            raise AliyunAsrError(
                "Missing Aliyun ASR credentials. Set ALIYUN_API_KEY or DASHSCOPE_API_KEY."
            )
        return cls(config)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def extract_youtube_audio_url(self, video_id: str) -> str:
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        command = [
            self.config.yt_dlp_bin,
            "--no-playlist",
            "--no-warnings",
            "--skip-download",
            "-f",
            "ba/bestaudio/best",
            "--get-url",
            video_url,
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
        except FileNotFoundError as exc:
            raise AliyunAsrError("yt-dlp is required for Aliyun ASR fallback but was not found.") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise AliyunAsrError(f"yt-dlp failed to resolve YouTube audio URL: {detail}") from exc
        except subprocess.TimeoutExpired as exc:
            raise AliyunAsrError("yt-dlp timed out while resolving YouTube audio URL.") from exc

        audio_url = next((line.strip() for line in completed.stdout.splitlines() if line.strip()), "")
        if not audio_url.startswith(("http://", "https://")):
            raise AliyunAsrError("yt-dlp did not return a usable audio URL.")
        return audio_url

    def download_youtube_audio(self, video_id: str, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        output_template = output_dir / f"{video_id}.%(ext)s"
        command = [
            self.config.yt_dlp_bin,
            "--no-playlist",
            "--no-warnings",
            "-f",
            "ba[ext=m4a]/bestaudio[ext=m4a]/ba/best",
            "-o",
            str(output_template),
            video_url,
        ]
        before = set(output_dir.glob(f"{video_id}.*"))
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=900,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
        except FileNotFoundError as exc:
            raise AliyunAsrError("yt-dlp is required for Aliyun ASR fallback but was not found.") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise AliyunAsrError(f"yt-dlp failed to download YouTube audio: {detail}") from exc
        except subprocess.TimeoutExpired as exc:
            raise AliyunAsrError("yt-dlp timed out while downloading YouTube audio.") from exc

        candidates = [
            path
            for path in output_dir.glob(f"{video_id}.*")
            if path not in before and path.is_file() and path.suffix not in {".part", ".ytdl"}
        ]
        if not candidates:
            candidates = [
                path
                for path in output_dir.glob(f"{video_id}.*")
                if path.is_file() and path.suffix not in {".part", ".ytdl"}
            ]
        if not candidates:
            raise AliyunAsrError("yt-dlp finished but no downloaded audio file was found.")
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def get_upload_policy(self) -> dict[str, Any]:
        query = urlencode({"action": "getPolicy", "model": self.config.model})
        response = _request_json(
            f"{self.config.api_base}/uploads?{query}",
            method="GET",
            headers=self._headers(),
            timeout=60,
        )
        policy = response.get("data") if isinstance(response.get("data"), dict) else response
        if not isinstance(policy, dict):
            raise AliyunAsrError(f"Aliyun upload policy response is not usable: {response}")
        required = ("upload_host", "upload_dir", "oss_access_key_id", "policy", "signature")
        missing = [key for key in required if not policy.get(key)]
        if missing:
            raise AliyunAsrError(f"Aliyun upload policy is missing fields {missing}: {response}")
        return policy

    def upload_file_and_get_url(self, file_path: Path) -> str:
        policy = self.get_upload_policy()
        upload_dir = str(policy["upload_dir"]).rstrip("/")
        object_key = f"{upload_dir}/{file_path.name}"
        fields = {
            "OSSAccessKeyId": str(policy["oss_access_key_id"]),
            "policy": str(policy["policy"]),
            "Signature": str(policy["signature"]),
            "key": object_key,
            "x-oss-object-acl": str(policy.get("x_oss_object_acl") or "private"),
            "x-oss-forbid-overwrite": str(policy.get("x_oss_forbid_overwrite") or "true"),
            "success_action_status": "200",
        }
        content_type, body = _multipart_form_data(fields, file_field="file", file_path=file_path)
        _request_bytes(
            str(policy["upload_host"]),
            method="POST",
            headers={"Content-Type": content_type},
            data=body,
            timeout=600,
        )
        return f"oss://{object_key}"

    def submit_task(self, audio_url: str) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": {"file_urls": [audio_url]},
            "parameters": {"language_hints": list(self.config.language_hints)},
        }
        headers = {**self._headers(), "X-DashScope-Async": "enable"}
        if audio_url.startswith("oss://"):
            headers["X-DashScope-OssResourceResolve"] = "enable"
        response = _request_json(
            f"{self.config.api_base}/services/audio/asr/transcription",
            method="POST",
            headers=headers,
            payload=payload,
            timeout=60,
        )
        task_id = response.get("output", {}).get("task_id") if isinstance(response.get("output"), dict) else None
        if not isinstance(task_id, str) or not task_id:
            raise AliyunAsrError(f"Aliyun ASR did not return a task id: {response}")
        return task_id

    def wait_for_task(self, task_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.config.timeout_sec
        last_payload: dict[str, Any] = {}
        while time.monotonic() < deadline:
            payload = _request_json(
                f"{self.config.api_base}/tasks/{task_id}",
                method="GET",
                headers={**self._headers(), "X-DashScope-Async": "enable"},
                timeout=60,
            )
            last_payload = payload
            output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
            status = output.get("task_status") or payload.get("task_status")
            if status == "SUCCEEDED":
                return payload
            if status in {"FAILED", "CANCELED", "UNKNOWN"}:
                message = output.get("message") or output.get("error_message") or payload
                raise AliyunAsrError(f"Aliyun ASR task {task_id} failed: {message}")
            time.sleep(max(self.config.poll_interval_sec, 0.5))
        raise AliyunAsrError(f"Aliyun ASR task {task_id} timed out: {last_payload}")

    def fetch_result(self, task_payload: dict[str, Any]) -> dict[str, Any]:
        output = task_payload.get("output") if isinstance(task_payload.get("output"), dict) else {}
        results = output.get("results")
        if not isinstance(results, list) or not results:
            raise AliyunAsrError(f"Aliyun ASR task succeeded but returned no result list: {task_payload}")
        first = results[0]
        if not isinstance(first, dict):
            raise AliyunAsrError(f"Aliyun ASR result entry is not an object: {first}")
        result_url = first.get("transcription_url") or first.get("url")
        if not isinstance(result_url, str) or not result_url.startswith(("http://", "https://")):
            raise AliyunAsrError(f"Aliyun ASR result did not include transcription_url: {first}")
        return _request_json(result_url, method="GET", headers={}, timeout=120)

    def transcribe_youtube_video(self, video_id: str) -> dict[str, Any]:
        temporary_dir: Optional[tempfile.TemporaryDirectory[str]] = None
        if self.config.audio_dir is None:
            temporary_dir = tempfile.TemporaryDirectory(prefix="pod2article-aliyun-asr-")
            audio_dir = Path(temporary_dir.name)
        else:
            audio_dir = self.config.audio_dir

        audio_path: Optional[Path] = None
        try:
            audio_path = self.download_youtube_audio(video_id, audio_dir)
            audio_url = self.upload_file_and_get_url(audio_path)
        finally:
            if audio_path is not None and not self.config.keep_audio:
                try:
                    audio_path.unlink(missing_ok=True)
                except OSError:
                    pass
            if temporary_dir is not None:
                temporary_dir.cleanup()

        task_id = self.submit_task(audio_url)
        task_payload = self.wait_for_task(task_id)
        result_payload = self.fetch_result(task_payload)
        normalized = normalize_aliyun_asr_payload(result_payload, video_id=video_id)
        normalized["task_id"] = task_id
        normalized["task_payload"] = task_payload
        normalized["uploaded_audio_url"] = audio_url
        return normalized
