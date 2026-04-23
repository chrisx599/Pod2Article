# Pod2Article

Turn a YouTube podcast URL, video ID, or search query into a Markdown article with clickable timestamp links.

## Usage

SerpApi is the default provider. Put credentials in `pod2article.config`:

```bash
SERPAPI_API_KEY=your-serpapi-api-key
```

Credential lookup order is:

1. `pod2article.config`, `.pod2article.config`, or `config.env`
2. system environment variables
3. local `.env`

Fetch complete transcript context for an agent-written article:

```bash
python3 podcast-to-article/scripts/fetch_transcript.py "https://www.youtube.com/watch?v=hmtuvNfytjM" --output-dir transcripts
```

Use Oxylabs instead:

```bash
python3 podcast-to-article/scripts/fetch_transcript.py "search query" --provider oxylabs --output-dir transcripts
```

The transcript fetcher writes `transcripts/<slug>.transcript.json` with metadata, coverage, chapters, and full timestamped transcript segments. Agents should use that context plus `podcast-to-article/templates/article-template.md` to write the final article.

The legacy automatic draft generator is still available for compatibility:

```bash
python3 podcast-to-article/scripts/build_article.py "https://www.youtube.com/watch?v=hmtuvNfytjM" --output-dir articles
```

## Benchmarks

Run the YouTube provider benchmark:

```bash
python3 benchmarks/benchmark_youtube_providers.py --limit 10 --providers serpapi,oxylabs --output-dir benchmark-results
```

Cache raw and normalized transcript payloads:

```bash
python3 benchmarks/cache_transcript_payloads.py iKx3gAODybU --providers serpapi,oxylabs
```

## Tests

```bash
python3 -m unittest discover -s tests
```
