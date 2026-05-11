# Pod2Article

Fetch normalized YouTube search results and complete timed transcript context for agent-written articles. The repository now also includes a Claude Agent SDK runner and a lightweight Video Deep Research HTTP API that can turn a YouTube URL, video ID, or search query into a timestamp-grounded Markdown article.

## Video Deep Research Agent

Create a virtual environment and install runtime dependencies, including `claude-agent-sdk` and this project's existing SerpApi requirements:

```bash
python3 -m venv .venv
.venv/bin/pip install claude-agent-sdk
```

Configure credentials in `pod2article.config`, system environment variables, or `agents/.env`. The Agent is intended to run DeepSeek through its Anthropic-compatible endpoint:

```bash
SERPAPI_API_KEY=your-serpapi-api-key
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
ANTHROPIC_API_KEY=your-deepseek-api-key
CLAUDE_AGENT_MODEL=deepseek-v4-pro
```

`ANTHROPIC_API_KEY` should be a DeepSeek API key when `ANTHROPIC_BASE_URL` points to DeepSeek. `deepseek-v4-flash` can be used instead for lower-cost runs.

Run the Agent directly:

```bash
scripts/agent/run_podcast_article_agent.sh \
  --input "https://www.youtube.com/watch?v=hmtuvNfytjM" \
  --question "请写一篇关于这期访谈核心观点的深度文章"
```

The runner writes runtime data under `output/agent/<source-id>/`:

```text
transcripts/*.transcript.json
articles/article-<timestamp>-<id>/article.md
```

## Video Deep Research API

Start the service:

```bash
scripts/api/start_service.sh
```

Create a wide-search synchronous task. This searches YouTube from the research question, fetches transcripts for relevant candidates, and writes a synthesized article:

```bash
curl -X POST http://127.0.0.1:8090/video-deep-research/api/tasks/sync \
  -H 'Content-Type: application/json' \
  -d '{"question":"请总结 Sam Altman 今年以来对 AI 的核心看法，并给出视频时间戳证据"}'
```

The synchronous endpoint blocks until the Agent finishes. A successful response includes the full article:

```json
{
  "task_id": "20260511T020405Z-1a2b3c4d",
  "status": "completed",
  "research_mode": "wide",
  "created_at": "2026-05-11T02:04:05.000000+00:00",
  "updated_at": "2026-05-11T02:10:12.000000+00:00",
  "article_available": true,
  "article_path": "output/api/20260511T020405Z-1a2b3c4d/.../article.md",
  "error_message": "",
  "article_markdown": "# ..."
}
```

Create a deep-search synchronous task for one known video:

```bash
curl -X POST http://127.0.0.1:8090/video-deep-research/api/tasks/sync \
  -H 'Content-Type: application/json' \
  -d '{"input":"https://www.youtube.com/watch?v=hmtuvNfytjM","question":"请写一篇深度研究文章"}'
```

Create an async task:

```bash
curl -X POST http://127.0.0.1:8090/video-deep-research/api/tasks \
  -H 'Content-Type: application/json' \
  -d '{"question":"请总结 Sam Altman 今年以来对 AI agent、模型能力和产品方向的看法"}'
```

The async endpoint returns immediately with a task id while the Agent keeps running in the background:

```json
{
  "task_id": "20260511T021530Z-5e6f7a8b",
  "status": "queued",
  "research_mode": "wide",
  "created_at": "2026-05-11T02:15:30.000000+00:00",
  "updated_at": "2026-05-11T02:15:30.000000+00:00",
  "article_available": false,
  "article_path": "",
  "error_message": ""
}
```

Use the returned `task_id` to poll status, stream progress events, and fetch the final article:

```bash
TASK_ID=20260511T021530Z-5e6f7a8b

curl http://127.0.0.1:8090/video-deep-research/api/tasks/$TASK_ID/status
curl http://127.0.0.1:8090/video-deep-research/api/tasks/$TASK_ID/progress
curl http://127.0.0.1:8090/video-deep-research/api/tasks/$TASK_ID/article
```

Status response:

```json
{
  "task_id": "20260511T021530Z-5e6f7a8b",
  "status": "running",
  "research_mode": "wide",
  "created_at": "2026-05-11T02:15:30.000000+00:00",
  "updated_at": "2026-05-11T02:16:08.000000+00:00",
  "article_available": false,
  "article_path": "",
  "error_message": ""
}
```

Progress response:

```json
{
  "task_id": "20260511T021530Z-5e6f7a8b",
  "status": "running",
  "next_after_seq": 3,
  "has_more": false,
  "events": [
    {
      "seq": 1,
      "type": "phase_started",
      "phase": "prepare",
      "message": "准备任务",
      "data": {}
    }
  ]
}
```

Article response after completion:

```json
{
  "task_id": "20260511T021530Z-5e6f7a8b",
  "status": "completed",
  "article_markdown": "# ...",
  "article_path": "output/api/20260511T021530Z-5e6f7a8b/.../article.md"
}
```

API endpoints:

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/video-deep-research/api/tasks` | Create an async task |
| `POST` | `/video-deep-research/api/tasks/sync` | Run a task and return the article |
| `GET` | `/video-deep-research/api/tasks/{task_id}/status` | Read task status |
| `GET` | `/video-deep-research/api/tasks/{task_id}/progress` | Read progress events |
| `GET` | `/video-deep-research/api/tasks/{task_id}/article` | Read generated Markdown |
| `DELETE` | `/video-deep-research/api/tasks/{task_id}` | Delete task output |

## Usage

Put SerpApi credentials in `pod2article.config`:

```bash
SERPAPI_API_KEY=your-serpapi-api-key
```

Credential lookup order is:

1. `pod2article.config`, `.pod2article.config`, or `config.env`
2. system environment variables
3. local `.env`

Search YouTube:

```bash
python3 podcast-to-article/scripts/search_youtube.py "lex fridman vikings" --output-dir search-results
```

Fetch complete transcript context:

```bash
python3 podcast-to-article/scripts/fetch_transcript.py "https://www.youtube.com/watch?v=hmtuvNfytjM" --output-dir transcripts
```

The transcript fetcher writes `transcripts/<slug>.transcript.json` with metadata, coverage, chapters, and full timestamped transcript segments. The codebase does not generate article drafts; agents write articles directly from this source context.

## Tests

```bash
python3 -m unittest discover -s tests
```
