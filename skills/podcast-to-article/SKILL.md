---
name: podcast-to-article
description: Turn a YouTube podcast episode, video ID, or search query into a structured Markdown article with clickable timestamp links. Use this when the user wants grounded video deep research from YouTube transcript context.
argument-hint: "<youtube-url|video-id|search-query>"
---

# Podcast to Article

Use this skill to turn YouTube podcast material into a grounded Markdown article. The agent runner may provide exact output directories and an exact article path; those runner-supplied paths take priority over the default paths below.

## Workflow

1. Resolve the target video from a YouTube URL, video ID, or search query.
2. Fetch metadata plus transcript/subtitles through the bundled SerpApi tooling.
3. Normalize the complete timed transcript into an agent-readable context file.
4. Read the generated `.transcript.json` file before writing.
5. Write the final article yourself using transcript context as evidence.
6. Include clickable YouTube timestamp links for source-backed claims.

Run the transcript fetcher from the repository root:

```bash
python3 podcast-to-article/scripts/fetch_transcript.py "$ARGUMENTS" --output-dir transcripts
```

When a runner supplies a transcript output directory, use that directory instead of `transcripts`. When a runner supplies an exact article path, write the final Markdown only to that path.

## Runtime Requirements

The bundled Python tooling uses SerpApi only.

Credential lookup order:

1. `pod2article.config`, `.pod2article.config`, or `config.env`
2. system environment variables
3. local `.env`

Required credential:

- `SERPAPI_API_KEY`

Do not commit real credentials.

## Writing Rules

- Stay faithful to the transcript context.
- Use the same language as the transcript unless the user asks for translation.
- Prefer synthesis over recap: explain the central thesis, how ideas connect, and why the episode matters.
- Use metadata, chapters, coverage, and full timestamped segments from the transcript context.
- Include at least one timestamp link per main section when evidence is available.
- Use YouTube second-based links such as `https://www.youtube.com/watch?v=<id>&t=<seconds>s`.
- Use direct quotes sparingly; prefer paraphrase plus timestamp links.
- Do not invent claims, names, dates, or examples unsupported by the transcript.
- If transcript coverage is incomplete, state that caveat and avoid writing beyond available material.

Recommended Markdown structure:

```markdown
# <article title>

## Source

## TL;DR

## Why This Conversation Matters

## Main Ideas

## Key Takeaways

## Source Timeline
```

## Failure Handling

Stop and explain the issue if:

- no usable video can be resolved from the input
- transcript and subtitles are unavailable
- credentials are missing
- SerpApi returns an unrecoverable API error

Do not fabricate article content when source text could not be retrieved.

