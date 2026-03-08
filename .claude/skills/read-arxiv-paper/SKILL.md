---
name: read-arxiv-paper
description: Use this skill when asked to read an arxiv paper given an arxiv URL
---

All file paths below are relative to the **project root** (the directory containing the `.claude/` folder, not the skill directory itself).

You will be given a URL of an arxiv paper, for example:

https://www.arxiv.org/abs/2601.07372

---

## Part 1: Normalize the URL

The goal is to fetch the TeX Source of the paper (not the PDF!), the URL always looks like this:

https://www.arxiv.org/src/2601.07372

Notice the `/src/` in the URL. Extract the `arxiv_id` (e.g. `2601.07372`).

## Part 2: Download the paper source

Fetch the URL to a local `.tar.gz` file.

## Part 3: Unpack

Unpack the contents into `./knowledge/{arxiv_id}/` and delete the `.tar.gz`.

## Part 4: Locate the entrypoint

Every LaTeX source usually has an entrypoint, such as `main.tex` or similar. Find it.

## Part 5: Read the paper

Read the entrypoint, then recurse through all other relevant source files to read the paper.

## Part 6: Report

Once you've read the paper, produce a summary of the paper into a markdown file at `./knowledge/summary_{tag}.md`. Extract:

- **Title**
- **Authors**
- **Core contribution / TLDR** (1-2 sentences)
- **Motivation & problem setting**
- **Method details** (architecture, algorithm, key equations)
- **Main results** (tables, figures, interesting ablations)
- **Limitations and open questions**

Remember that you're processing this paper within the context of a looped transformer research project. Therefore, you should feel free to "remind yourself" of the related project code by reading the relevant parts, and then explicitly make the connection of how this paper might relate or what are things we might be inspired about or try.

## Part 7: Write the summary to Notion

Read the project page id from `./notion.txt`. If this file does not exist, use the Notion MCP tool `notion-search` to find the page by searching for "Looped Nanochat" under the "Loop Transformers → Experimentation" hierarchy, then extract the page ID from the search results.

Use the Notion MCP tool `notion-create-pages` to create an entry in the appropriate Paper Summaries database.