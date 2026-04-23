from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "podcast-to-article" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_article import parse_metadata, search_candidates  # noqa: E402
from normalize import normalize_timed_content  # noqa: E402
from oxylabs_client import OxylabsClient  # noqa: E402
from serpapi_client import SerpApiClient  # noqa: E402
from utils import load_local_env, parse_credentials, parse_serpapi_key  # noqa: E402


@dataclass
class ChannelVideo:
    video_id: str
    title: str
    url: str


@dataclass
class SearchResult:
    ok: bool
    latency_sec: float
    candidates_count: int
    expected_rank: Optional[int]
    top_video_id: Optional[str]
    error: Optional[str] = None


@dataclass
class TranscriptQuality:
    segments_count: int = 0
    words_count: int = 0
    chars_count: int = 0
    span_sec: int = 0
    monotonic_timestamps: bool = False
    avg_words_per_segment: float = 0.0
    duplicate_segment_ratio: float = 0.0
    bracket_noise_ratio: float = 0.0
    quality_score: int = 0


@dataclass
class TranscriptResult:
    ok: bool
    latency_sec: float
    source_kind: Optional[str]
    origin: Optional[str]
    quality: TranscriptQuality
    error: Optional[str] = None


@dataclass
class ProviderVideoResult:
    provider: str
    video: ChannelVideo
    search: SearchResult
    transcript: TranscriptResult


def timed_call(func):
    start = time.perf_counter()
    try:
        value = func()
    except Exception as exc:
        return None, time.perf_counter() - start, exc
    return value, time.perf_counter() - start, None


def fetch_text(url: str, timeout: int = 30) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="ignore")
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc


def extract_channel_videos(channel_url: str, limit: int) -> list[ChannelVideo]:
    url = channel_url.rstrip("/") + "/videos"
    html = fetch_text(url)
    match = re.search(r"var ytInitialData = (\{.*?\});</script>", html)
    if not match:
        match = re.search(r"window\[\"ytInitialData\"\]\s*=\s*(\{.*?\});", html)
    if not match:
        raise RuntimeError("Unable to find ytInitialData in the YouTube channel page.")

    data = json.loads(match.group(1))
    videos: list[ChannelVideo] = []
    seen: set[str] = set()

    def walk(value: Any) -> None:
        if len(videos) >= limit:
            return
        if isinstance(value, dict):
            renderer = value.get("videoRenderer")
            if isinstance(renderer, dict):
                video_id = renderer.get("videoId")
                title = extract_renderer_title(renderer)
                if isinstance(video_id, str) and title and video_id not in seen:
                    seen.add(video_id)
                    videos.append(
                        ChannelVideo(
                            video_id=video_id,
                            title=title,
                            url=f"https://www.youtube.com/watch?v={video_id}",
                        )
                    )
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(data)
    if len(videos) < limit:
        raise RuntimeError(f"Only found {len(videos)} videos, expected {limit}.")
    return videos


def extract_renderer_title(renderer: dict[str, Any]) -> str:
    title = renderer.get("title")
    if isinstance(title, dict):
        runs = title.get("runs")
        if isinstance(runs, list):
            return "".join(str(run.get("text", "")) for run in runs).strip()
        simple = title.get("simpleText")
        if isinstance(simple, str):
            return simple.strip()
    return ""


def build_client(provider: str):
    if provider == "serpapi":
        return SerpApiClient(parse_serpapi_key(REPO_ROOT))
    if provider == "oxylabs":
        username, password = parse_credentials(REPO_ROOT)
        return OxylabsClient(username, password)
    raise ValueError(f"Unknown provider: {provider}")


def benchmark_search(client: Any, video: ChannelVideo) -> SearchResult:
    payload, latency, error = timed_call(lambda: client.search(video.title, subtitles=True))
    if error is not None:
        return SearchResult(
            ok=False,
            latency_sec=latency,
            candidates_count=0,
            expected_rank=None,
            top_video_id=None,
            error=compact_error(error),
        )
    candidates = search_candidates(payload, video.title)
    expected_rank = next(
        (index for index, candidate in enumerate(candidates, start=1) if candidate.video_id == video.video_id),
        None,
    )
    return SearchResult(
        ok=True,
        latency_sec=latency,
        candidates_count=len(candidates),
        expected_rank=expected_rank,
        top_video_id=candidates[0].video_id if candidates else None,
    )


def benchmark_transcript(client: Any, video: ChannelVideo, language_code: str) -> TranscriptResult:
    probe, latency, error = timed_call(lambda: client.fetch_best_timed_content(video.video_id, language_code=language_code))
    if error is not None:
        return TranscriptResult(
            ok=False,
            latency_sec=latency,
            source_kind=None,
            origin=None,
            quality=TranscriptQuality(),
            error=compact_error(error),
        )
    metadata = parse_metadata(probe.metadata, video.video_id)
    segments = normalize_timed_content(
        probe.content_payload,
        video_id=video.video_id,
        source_kind=probe.source_kind,
        language=metadata["language"],
    )
    return TranscriptResult(
        ok=True,
        latency_sec=latency,
        source_kind=probe.source_kind,
        origin=probe.origin,
        quality=score_transcript(segments),
    )


def score_transcript(segments: list[Any]) -> TranscriptQuality:
    texts = [str(segment.text).strip() for segment in segments if str(segment.text).strip()]
    words = [word for text in texts for word in re.findall(r"[A-Za-z0-9']+", text)]
    starts = [int(segment.start_sec) for segment in segments]
    ends = [int(segment.end_sec or segment.start_sec) for segment in segments]
    span_sec = max(ends) - min(starts) if starts and ends else 0
    monotonic = all(left <= right for left, right in zip(starts, starts[1:]))
    avg_words = len(words) / max(len(texts), 1)
    duplicate_ratio = 0.0
    if texts:
        duplicate_ratio = 1.0 - (len(set(texts)) / len(texts))
    bracket_noise = sum(1 for text in texts if re.search(r"\[[^\]]+\]", text)) / max(len(texts), 1)

    score = 0
    score += min(35, int(len(words) / 200))
    score += min(25, int(len(texts) / 20))
    score += min(20, int(span_sec / 300))
    score += 10 if monotonic else 0
    if 3 <= avg_words <= 45:
        score += 5
    if duplicate_ratio <= 0.02:
        score += 3
    if bracket_noise <= 0.02:
        score += 2

    return TranscriptQuality(
        segments_count=len(texts),
        words_count=len(words),
        chars_count=sum(len(text) for text in texts),
        span_sec=span_sec,
        monotonic_timestamps=monotonic,
        avg_words_per_segment=round(avg_words, 2),
        duplicate_segment_ratio=round(duplicate_ratio, 4),
        bracket_noise_ratio=round(bracket_noise, 4),
        quality_score=min(100, score),
    )


def compact_error(error: Exception) -> str:
    message = str(error).replace("\n", " ").strip()
    return message[:500]


def summarize(results: list[ProviderVideoResult]) -> dict[str, Any]:
    providers = sorted({result.provider for result in results})
    summary: dict[str, Any] = {}
    for provider in providers:
        rows = [result for result in results if result.provider == provider]
        search_ok = [row for row in rows if row.search.ok]
        transcript_ok = [row for row in rows if row.transcript.ok]
        exact_top = [row for row in rows if row.search.expected_rank == 1]
        found_any = [row for row in rows if row.search.expected_rank is not None]
        summary[provider] = {
            "videos": len(rows),
            "search_success_rate": ratio(len(search_ok), len(rows)),
            "search_expected_top1_rate": ratio(len(exact_top), len(rows)),
            "search_expected_found_rate": ratio(len(found_any), len(rows)),
            "search_latency_avg_sec": round(mean([row.search.latency_sec for row in search_ok]), 3),
            "search_latency_p95_sec": round(p95([row.search.latency_sec for row in search_ok]), 3),
            "transcript_success_rate": ratio(len(transcript_ok), len(rows)),
            "transcript_latency_avg_sec": round(mean([row.transcript.latency_sec for row in transcript_ok]), 3),
            "transcript_latency_p95_sec": round(p95([row.transcript.latency_sec for row in transcript_ok]), 3),
            "quality_score_avg": round(mean([row.transcript.quality.quality_score for row in transcript_ok]), 1),
            "segments_avg": round(mean([row.transcript.quality.segments_count for row in transcript_ok]), 1),
            "words_avg": round(mean([row.transcript.quality.words_count for row in transcript_ok]), 1),
        }
    return summary


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 3) if denominator else 0.0


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * 0.95)))
    return ordered[index]


def render_markdown(
    *,
    channel_url: str,
    videos: list[ChannelVideo],
    results: list[ProviderVideoResult],
    summary: dict[str, Any],
    started_at: str,
    finished_at: str,
) -> str:
    lines = [
        "# YouTube Provider Benchmark",
        "",
        f"- Channel: {channel_url}",
        f"- Started: {started_at}",
        f"- Finished: {finished_at}",
        f"- Videos: {len(videos)}",
        "",
        "## Summary",
        "",
        "| Provider | Search OK | Top-1 | Found | Search avg/p95 | Transcript OK | Transcript avg/p95 | Quality avg | Segments avg | Words avg |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for provider, item in summary.items():
        lines.append(
            "| {provider} | {search_ok:.1%} | {top1:.1%} | {found:.1%} | {savg:.3f}s/{sp95:.3f}s | "
            "{tok:.1%} | {tavg:.3f}s/{tp95:.3f}s | {quality:.1f} | {segments:.1f} | {words:.1f} |".format(
                provider=provider,
                search_ok=item["search_success_rate"],
                top1=item["search_expected_top1_rate"],
                found=item["search_expected_found_rate"],
                savg=item["search_latency_avg_sec"],
                sp95=item["search_latency_p95_sec"],
                tok=item["transcript_success_rate"],
                tavg=item["transcript_latency_avg_sec"],
                tp95=item["transcript_latency_p95_sec"],
                quality=item["quality_score_avg"],
                segments=item["segments_avg"],
                words=item["words_avg"],
            )
        )

    lines.extend(
        [
            "",
            "## Videos",
            "",
            "| # | Video ID | Title |",
            "| ---: | --- | --- |",
        ]
    )
    for index, video in enumerate(videos, start=1):
        lines.append(f"| {index} | `{video.video_id}` | {escape_table(video.title)} |")

    lines.extend(
        [
            "",
            "## Per-Video Results",
            "",
            "| Provider | Video ID | Search rank | Search latency | Transcript | Transcript latency | Segments | Words | Span | Quality | Error |",
            "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in results:
        rank = result.search.expected_rank if result.search.expected_rank is not None else "-"
        transcript_state = "ok" if result.transcript.ok else "fail"
        error = result.search.error or result.transcript.error or ""
        lines.append(
            "| {provider} | `{video_id}` | {rank} | {search_latency:.3f}s | {transcript_state} | "
            "{transcript_latency:.3f}s | {segments} | {words} | {span}s | {quality} | {error} |".format(
                provider=result.provider,
                video_id=result.video.video_id,
                rank=rank,
                search_latency=result.search.latency_sec,
                transcript_state=transcript_state,
                transcript_latency=result.transcript.latency_sec,
                segments=result.transcript.quality.segments_count,
                words=result.transcript.quality.words_count,
                span=result.transcript.quality.span_sec,
                quality=result.transcript.quality.quality_score,
                error=escape_table(error),
            )
        )
    return "\n".join(lines) + "\n"


def escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def write_outputs(
    output_dir: Path,
    *,
    channel_url: str,
    videos: list[ChannelVideo],
    results: list[ProviderVideoResult],
    started_at: str,
    finished_at: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "channel_url": channel_url,
        "started_at": started_at,
        "finished_at": finished_at,
        "videos": [asdict(video) for video in videos],
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    json_path = output_dir / f"youtube-provider-benchmark-{stamp}.json"
    md_path = output_dir / f"youtube-provider-benchmark-{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(
        render_markdown(
            channel_url=channel_url,
            videos=videos,
            results=results,
            summary=summary,
            started_at=started_at,
            finished_at=finished_at,
        ),
        encoding="utf-8",
    )
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YouTube search and transcript providers.")
    parser.add_argument("--channel-url", default="https://www.youtube.com/@lexfridman")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--language-code", default="en")
    parser.add_argument("--providers", default="serpapi,oxylabs")
    parser.add_argument("--output-dir", default="benchmark-results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_local_env(REPO_ROOT)
    providers = [item.strip() for item in args.providers.split(",") if item.strip()]
    started_at = datetime.now(timezone.utc).isoformat()
    videos = extract_channel_videos(args.channel_url, args.limit)
    results: list[ProviderVideoResult] = []

    for provider in providers:
        client = build_client(provider)
        for index, video in enumerate(videos, start=1):
            print(f"[{provider}] {index}/{len(videos)} search {video.video_id}", flush=True)
            search = benchmark_search(client, video)
            print(f"[{provider}] {index}/{len(videos)} transcript {video.video_id}", flush=True)
            transcript = benchmark_transcript(client, video, args.language_code)
            results.append(
                ProviderVideoResult(
                    provider=provider,
                    video=video,
                    search=search,
                    transcript=transcript,
                )
            )

    finished_at = datetime.now(timezone.utc).isoformat()
    json_path, md_path = write_outputs(
        Path(args.output_dir),
        channel_url=args.channel_url,
        videos=videos,
        results=results,
        started_at=started_at,
        finished_at=finished_at,
    )
    print(json.dumps({"json_path": str(json_path), "markdown_path": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
