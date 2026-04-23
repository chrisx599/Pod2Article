# Pod2Article

Turn a YouTube podcast URL, video ID, or search query into a Markdown article with clickable timestamp links.

## Usage

SerpApi is the default provider. Put credentials in `.env`:

```bash
SERPAPI_API_KEY=your-serpapi-api-key
```

Generate an article:

```bash
python3 podcast-to-article/scripts/build_article.py "https://www.youtube.com/watch?v=hmtuvNfytjM" --output-dir articles
```

Use Oxylabs instead:

```bash
python3 podcast-to-article/scripts/build_article.py "search query" --provider oxylabs --output-dir articles
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
