# YouTube API Workflow Reference

The bundled Python tooling supports SerpApi and Oxylabs YouTube targets.

## Provider selection

- `--provider serpapi` is the default and uses SerpApi.
- `--provider auto` prefers SerpApi when `SERPAPI_API_KEY` is set, otherwise Oxylabs.
- `--provider oxylabs` forces Oxylabs.

## SerpApi engines used

- `youtube`
- `youtube_video`
- `youtube_video_transcript`

## Oxylabs sources used

- `youtube_search`
- `youtube_search_max`
- `youtube_metadata`
- `youtube_transcript`
- `youtube_subtitles`

## Single-video workflow

1. Resolve the input type.
2. If the input is a URL or video ID, extract `video_id` directly.
3. If the input is a search query, search YouTube and rank likely candidates.
4. Fetch metadata for the chosen candidate.
5. Attempt timed content retrieval.
   - SerpApi: `youtube_video_transcript`, then ASR transcript fallback.
   - Oxylabs: transcript with `uploader_provided`, transcript with `auto_generated`, subtitles with `uploader_provided`, subtitles with `auto_generated`.
6. Normalize segments to second-based timestamps.
7. Build the article from the normalized segments.

## Search candidate ranking

Search mode should prefer:

- high textual relevance to the query
- longer-form video duration
- candidates that successfully return transcript or subtitle content

When search results are close, prefer the first candidate that can actually provide usable timestamped text.

## Credentials

The scripts read credentials from:

- `SERPAPI_API_KEY`
- `OXYLABS_USERNAME`
- `OXYLABS_PASSWORD`

Local development may use a `.env` file in the repository root, but real credentials must not be committed.
