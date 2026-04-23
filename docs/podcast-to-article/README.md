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
2. Fetch metadata plus transcript or subtitles through SerpApi.
3. Normalize the complete timed transcript into an agent-readable context file.
4. Read the context file, including metadata, chapters, coverage, and full transcript segments.
5. Write the article yourself using the transcript context.
6. Save the final Markdown output to `articles/<slug>.md`.
7. Return the article path, transcript context path, coverage status, and any important caveats.

## Agent behavior

When using this skill, do not merely describe the script or return the script path. Run the bundled transcript fetcher unless the user explicitly asks only for implementation details.

After the transcript fetcher runs:

- Open the generated `.transcript.json` file before writing.
- Check `coverage.last_end_sec`, `coverage.span_timestamp`, `segments`, and `chapters`.
- Use the full transcript context as source material; do not ask the fetcher to write the article for you.
- If YouTube chapters are available, cover all usable chapters unless the user asks for a shorter piece.
- Write a coherent article in Markdown; the codebase intentionally does not generate article drafts.
- Preserve source fidelity and include clickable timestamp links from the context.
- In the final response, report the article path, transcript context path, and a short coverage note. Do not lead with an implementation script path.

The bundled Python tooling exposes only two capabilities: YouTube search and transcript fetching. The agent is responsible for all article writing when the user asks for an article, essay, polished draft, or publishable output.

## Runtime requirements

The bundled Python tooling uses SerpApi only.

SerpApi credentials:

- `SERPAPI_API_KEY`

Credential lookup order is:

1. `pod2article.config`, `.pod2article.config`, or `config.env`
2. system environment variables
3. local `.env`

Do not commit real credentials into the repository.

## How to use it

Search YouTube from the project root:

```bash
python3 podcast-to-article/scripts/search_youtube.py "$ARGUMENTS" --output-dir search-results
```

Fetch transcript context from the project root:

```bash
python3 podcast-to-article/scripts/fetch_transcript.py "$ARGUMENTS" --output-dir transcripts
```

Useful optional flags:

```bash
python3 podcast-to-article/scripts/fetch_transcript.py "$ARGUMENTS" --output-dir transcripts --language-code en
```

## Output expectations

The delivered article should:

- Stay faithful to the source material.
- Use English structural labels in the Markdown layout.
- Preserve source-language content in summaries and excerpts unless the user explicitly asks for translation.
- Include at least one clickable timestamp link per main section.
- Use YouTube second-based links such as `https://www.youtube.com/watch?v=<id>&t=<seconds>s`.
- Read as a coherent article, not just a list of transcript snippets, when the user asks for a finished article.

## Article Writing Workflow

Use the transcript context as evidence, not as prose to copy. Write the article in the same language as the transcript unless the user asks for translation.

Before drafting:

- Identify the central thesis of the episode from the title, chapters, and recurring arguments.
- Use `chapters` as the default coverage map; if chapters are missing, group `segments` into 4-8 chronological themes.
- Select the strongest moments for each section and keep their timestamp URLs available.
- Prefer synthesis over recap: explain why the ideas matter, how they connect, and where the speaker's reasoning changes.

Recommended Markdown structure:

- `# <article title>`
- `## Source`
- `## TL;DR`
- `## Why This Conversation Matters`
- `## Main Ideas`
- `## Key Takeaways`
- `## Source Timeline`

Writing rules:

- Every main section should include at least one timestamp link from the transcript context.
- Use direct quotes sparingly; prefer paraphrase plus timestamp links.
- Do not invent claims, names, dates, or examples that are not supported by the transcript context.
- Preserve nuance when speakers disagree, hedge, or change their mind.
- Avoid transcript-cleanup artifacts such as repeated filler, partial sentences, and ASR mistakes unless they are necessary for a quote.
- If the transcript coverage appears incomplete, state that caveat in the final response and avoid writing beyond the available material.

Quality check before returning:

- The article covers the full transcript or all user-requested scope.
- The opening states the episode's core idea, not just the guest/topic.
- Sections read as article prose, not bullet-only notes.
- Timestamp links point to relevant source moments.
- The final Markdown is saved under `articles/<slug>.md`.

The transcript context file contains:

- `video`: title, channel, URL, duration, language, and YouTube chapters.
- `coverage`: segment count, word count, first timestamp, last timestamp, and coverage ratio when duration is known.
- `chapters`: full transcript text grouped by chapter.
- `segments`: full timestamped transcript segments with timestamp URLs.

## Failure handling

Stop and explain the issue if:

- no usable video can be resolved from the input
- transcript and subtitles are both unavailable
- credentials are missing
- SerpApi returns an unrecoverable API error

Do not fabricate article content when source text could not be retrieved.
