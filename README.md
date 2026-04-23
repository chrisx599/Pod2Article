# Pod2Article

Fetch normalized YouTube search results and complete timed transcript context for agent-written articles.

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

## Benchmarks

Run the YouTube benchmark:

```bash
python3 benchmarks/benchmark_youtube_providers.py --limit 10 --output-dir benchmark-results
```

Cache raw and normalized transcript payloads:

```bash
python3 benchmarks/cache_transcript_payloads.py iKx3gAODybU
```

## Tests

```bash
python3 -m unittest discover -s tests
```
