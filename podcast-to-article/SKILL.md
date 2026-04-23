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
3. Normalize the complete timed transcript into an agent-readable context file.
4. Read the context file, including metadata, chapters, coverage, and full transcript segments.
5. Write the article yourself using the transcript context and the Markdown template.
6. Save the final Markdown output to `articles/<slug>.md`.
7. Return the article path, transcript context path, coverage status, and any important caveats.

## Agent behavior

When using this skill, do not merely describe the script or return the script path. Run the bundled transcript fetcher unless the user explicitly asks only for implementation details.

After the transcript fetcher runs:

- Open the generated `.transcript.json` file before writing.
- Check `coverage.last_end_sec`, `coverage.span_timestamp`, `segments`, and `chapters`.
- Use the full transcript context as source material; do not ask the fetcher to write the article for you.
- If YouTube chapters are available, cover all usable chapters unless the user asks for a shorter piece.
- Write a coherent article in Markdown using `templates/article-template.md` as the structure.
- Preserve source fidelity and include clickable timestamp links from the context.
- In the final response, report the article path, transcript context path, and a short coverage note. Do not lead with an implementation script path.

The bundled fetcher is deterministic and intentionally conservative. It is useful for resolving videos, fetching transcript data, preserving timestamps, and producing source context. The agent is responsible for all article writing when the user asks for an article, essay, polished draft, or publishable output.

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

Run the bundled transcript fetcher from the project root:

```bash
python3 podcast-to-article/scripts/fetch_transcript.py "$ARGUMENTS" --output-dir transcripts
```

Useful optional flags:

```bash
python3 podcast-to-article/scripts/fetch_transcript.py "$ARGUMENTS" --output-dir transcripts --language-code en
python3 podcast-to-article/scripts/fetch_transcript.py "$ARGUMENTS" --output-dir transcripts --provider serpapi
python3 podcast-to-article/scripts/fetch_transcript.py "$ARGUMENTS" --output-dir transcripts --provider oxylabs
python3 podcast-to-article/scripts/fetch_transcript.py "$ARGUMENTS" --output-dir transcripts --search-source youtube_search_max
```

The legacy `scripts/build_article.py` still exists for compatibility and can create an automatic draft, but it is not the default workflow for this skill.

## Output expectations

The delivered article should:

- Stay faithful to the source material.
- Use English structural labels in the Markdown layout.
- Preserve source-language content in summaries and excerpts unless the user explicitly asks for translation.
- Include at least one clickable timestamp link per main section.
- Use YouTube second-based links such as `https://www.youtube.com/watch?v=<id>&t=<seconds>s`.
- Read as a coherent article, not just a list of transcript snippets, when the user asks for a finished article.

The transcript context file contains:

- `video`: title, channel, URL, duration, language, and YouTube chapters.
- `coverage`: segment count, word count, first timestamp, last timestamp, and coverage ratio when duration is known.
- `chapters`: full transcript text grouped by chapter.
- `segments`: full timestamped transcript segments with timestamp URLs.
- `agent_instructions`: reminders that this is source context, not an article.

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
