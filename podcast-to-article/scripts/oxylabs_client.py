from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OxylabsError(RuntimeError):
    """Raised when Oxylabs returns an unrecoverable error."""


@dataclass
class CandidateProbe:
    metadata: dict[str, Any]
    content_payload: dict[str, Any]
    source_kind: str
    origin: str


class OxylabsClient:
    base_url = "https://realtime.oxylabs.io/v1/queries"

    def __init__(self, username: str, password: str, timeout: int = 60) -> None:
        self.username = username
        self.password = password
        self.timeout = timeout

    def _request(self, body: dict[str, Any]) -> dict[str, Any]:
        token = base64.b64encode(f"{self.username}:{self.password}".encode("utf-8")).decode("ascii")
        request = Request(
            self.base_url,
            method="POST",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Basic {token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise OxylabsError(f"Oxylabs HTTP error {exc.code}: {detail}") from exc
        except URLError as exc:
            raise OxylabsError(f"Failed to reach Oxylabs: {exc}") from exc
        return payload

    def search(self, query: str, source: str = "youtube_search", subtitles: bool = True) -> dict[str, Any]:
        body: dict[str, Any] = {"source": source, "query": query}
        if subtitles:
            body["subtitles"] = True
            body["type"] = "video"
        return self._request(body)

    def metadata(self, video_id: str) -> dict[str, Any]:
        return self._request({"source": "youtube_metadata", "query": video_id, "parse": True})

    def transcript(self, video_id: str, language_code: str = "en", origin: str = "auto_generated") -> dict[str, Any]:
        return self._request(
            {
                "source": "youtube_transcript",
                "query": video_id,
                "context": [
                    {"key": "language_code", "value": language_code},
                    {"key": "transcript_origin", "value": origin},
                ],
            }
        )

    def subtitles(self, video_id: str, language_code: str = "en", origin: str = "auto_generated") -> dict[str, Any]:
        return self._request(
            {
                "source": "youtube_subtitles",
                "query": video_id,
                "context": [
                    {"key": "language_code", "value": language_code},
                    {"key": "subtitle_origin", "value": origin},
                ],
            }
        )

    @staticmethod
    def _payload_is_usable(payload: dict[str, Any]) -> bool:
        results = payload.get("results", [])
        if not results:
            return False
        first = results[0]
        status_code = first.get("status_code")
        content = first.get("content")
        if status_code != 200:
            return False
        if content in (None, "", []):
            return False
        return True

    def fetch_best_timed_content(self, video_id: str, language_code: str = "en") -> CandidateProbe:
        metadata = self.metadata(video_id)
        attempts = [
            ("transcript", "uploader_provided", self.transcript),
            ("transcript", "auto_generated", self.transcript),
            ("subtitles", "uploader_provided", self.subtitles),
            ("subtitles", "auto_generated", self.subtitles),
        ]
        last_error: Optional[Exception] = None
        for source_kind, origin, method in attempts:
            try:
                payload = method(video_id, language_code=language_code, origin=origin)
                if not self._payload_is_usable(payload):
                    raise OxylabsError(
                        f"Oxylabs returned an unusable {source_kind} payload for origin={origin}."
                    )
                return CandidateProbe(
                    metadata=metadata,
                    content_payload=payload,
                    source_kind=source_kind,
                    origin=origin,
                )
            except Exception as exc:
                last_error = exc
        raise OxylabsError(
            f"Unable to retrieve transcript or subtitles for video {video_id}."
        ) from last_error
