from __future__ import annotations

import json
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

if __package__ in {None, ""}:
    from oxylabs_client import CandidateProbe, OxylabsError
else:
    from .oxylabs_client import CandidateProbe, OxylabsError


class SerpApiError(OxylabsError):
    """Raised when SerpApi returns an unrecoverable error."""


class SerpApiClient:
    base_url = "https://serpapi.com/search.json"

    def __init__(self, api_key: str, timeout: int = 60, *, gl: str = "us", hl: str = "en") -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.gl = gl
        self.hl = hl

    def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        query = {
            "api_key": self.api_key,
            "gl": self.gl,
            "hl": self.hl,
            **params,
        }
        request = Request(f"{self.base_url}?{urlencode(query)}", method="GET")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise SerpApiError(f"SerpApi HTTP error {exc.code}: {detail}") from exc
        except URLError as exc:
            raise SerpApiError(f"Failed to reach SerpApi: {exc}") from exc
        if payload.get("error"):
            raise SerpApiError(f"SerpApi error: {payload['error']}")
        return payload

    def search(self, query: str, source: str = "youtube_search", subtitles: bool = True) -> dict[str, Any]:
        del source, subtitles
        return self._request({"engine": "youtube", "search_query": query})

    def metadata(self, video_id: str) -> dict[str, Any]:
        return self._request({"engine": "youtube_video", "v": video_id})

    def transcript(self, video_id: str, language_code: str = "en", origin: str = "auto_generated") -> dict[str, Any]:
        params: dict[str, Any] = {
            "engine": "youtube_video_transcript",
            "v": video_id,
            "language_code": language_code,
        }
        if origin == "auto_generated":
            params["type"] = "asr"
        return self._request(params)

    def subtitles(self, video_id: str, language_code: str = "en", origin: str = "auto_generated") -> dict[str, Any]:
        return self.transcript(video_id, language_code=language_code, origin=origin)

    @staticmethod
    def _payload_is_usable(payload: dict[str, Any]) -> bool:
        transcript = payload.get("transcript")
        return isinstance(transcript, list) and bool(transcript)

    def fetch_best_timed_content(self, video_id: str, language_code: str = "en") -> CandidateProbe:
        metadata = self.metadata(video_id)
        attempts = [
            ("transcript", "uploader_provided", self.transcript),
            ("transcript", "auto_generated", self.transcript),
        ]
        last_error: Optional[Exception] = None
        for source_kind, origin, method in attempts:
            try:
                payload = method(video_id, language_code=language_code, origin=origin)
                if not self._payload_is_usable(payload):
                    raise SerpApiError(f"SerpApi returned an unusable {source_kind} payload.")
                return CandidateProbe(
                    metadata=metadata,
                    content_payload=payload,
                    source_kind=source_kind,
                    origin=origin,
                )
            except Exception as exc:
                last_error = exc
        raise SerpApiError(f"Unable to retrieve transcript for video {video_id}.") from last_error
