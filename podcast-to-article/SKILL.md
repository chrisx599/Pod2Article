---
name: podcast-to-article
description: Turn a YouTube podcast episode into a structured Markdown article with clickable timestamp links. Use this when the user wants a YouTube URL, video ID, or search query transformed into an article that mixes summary text with links back to the original video.
argument-hint: "<youtube-url|video-id|search-query>"
---

# Podcast to Article

Use this skill to turn YouTube podcast material into a Markdown article that interleaves readable narrative with source-linked video moments.

## What this skill does

This skill supports three entry modes:

- A YouTube URL
- A YouTube video ID
- A search query that should be resolved to the best matching podcast episode

The workflow is:

1. Resolve the target video.
2. Fetch metadata plus transcript or subtitles through SerpApi or Oxylabs.
3. Normalize timestamped segments.
4. Build an outline-first article draft.
5. Save the Markdown output to `articles/<slug>.md`.
6. Return the saved path and summarize what was generated.

## Runtime requirements

The bundled Python tooling uses SerpApi by default. Oxylabs remains available with `--provider oxylabs`.

SerpApi credentials:

- `SERPAPI_API_KEY`

Oxylabs credentials:

- `OXYLABS_USERNAME`
- `OXYLABS_PASSWORD`

Credential lookup order is:

1. `pod2article.config`, `.pod2article.config`, or `config.env`
2. system environment variables
3. local `.env`

Do not commit real credentials into the repository.

## How to use it

Run the bundled builder from the project root:

```bash
python3 podcast-to-article/scripts/build_article.py "$ARGUMENTS" --output-dir articles
```

Useful optional flags:

```bash
python3 podcast-to-article/scripts/build_article.py "$ARGUMENTS" --output-dir articles --mode single
python3 podcast-to-article/scripts/build_article.py "$ARGUMENTS" --output-dir articles --language-code en
python3 podcast-to-article/scripts/build_article.py "$ARGUMENTS" --output-dir articles --provider serpapi
python3 podcast-to-article/scripts/build_article.py "$ARGUMENTS" --output-dir articles --provider oxylabs
python3 podcast-to-article/scripts/build_article.py "$ARGUMENTS" --output-dir articles --search-source youtube_search_max
```

## Output expectations

The generated article should:

- Stay faithful to the source material.
- Use English structural labels in the Markdown layout.
- Preserve source-language content in summaries and excerpts unless the user explicitly asks for translation.
- Include at least one clickable timestamp link per main section.
- Use YouTube second-based links such as `https://www.youtube.com/watch?v=<id>&t=<seconds>s`.

## Mode selection

Default behavior is single-video deep conversion.

Only use aggregation mode when the user explicitly asks for:

- a roundup
- a comparison
- a multi-source article
- a cross-episode synthesis

If the request does not clearly require multiple videos, stay in single-video mode.

## Supporting files

- For implementation details, see [references/oxylabs-workflow.md](references/oxylabs-workflow.md).
- For Markdown layout requirements, see [references/article-format.md](references/article-format.md).
- For the article skeleton, see [templates/article-template.md](templates/article-template.md).

## Failure handling

Stop and explain the issue if:

- no usable video can be resolved from the input
- transcript and subtitles are both unavailable
- credentials are missing
- the selected provider returns an unrecoverable API error

Do not fabricate article content when source text could not be retrieved.
