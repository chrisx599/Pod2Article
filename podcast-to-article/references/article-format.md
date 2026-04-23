# Article Format Reference

Every generated Markdown article should follow the same high-level structure:

1. `# Title`
2. `## Source Metadata`
3. `## TL;DR`
4. `## Outline`
5. `## Main Sections`
6. `## Key Takeaways`
7. `## Source Timeline`

## Required properties

- Each main section needs at least one timestamp link.
- Timestamp links must point back to the original YouTube video using second-based anchors.
- The article should read like a longform write-up, not a raw transcript dump.
- Source excerpts should stay close to the original wording.
- Structural labels remain in English.

## Timestamp block format

Use a compact block such as:

```md
Source moment: [00:12:34](https://www.youtube.com/watch?v=VIDEO_ID&t=754s)  
Excerpt: "A short source-grounded quote or extract."
```

## Timeline section

The final timeline should include notable moments in chronological order with:

- displayed timestamp
- clickable source link
- one-line explanation of why that moment matters

