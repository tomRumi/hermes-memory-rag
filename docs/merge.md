# Merging staged learnings into the wiki

The `memory` layer is a staging area. Notes collect there during sessions and are periodically
folded into the project's wiki pages, which is where written knowledge is meant to live.

## Why it exists

A store that only ever grows becomes useless: everything matches everything a little, and searching
returns noise. Merging keeps the wiki the place where knowledge accumulates in readable form, and
keeps the searchable staging area small.

There is also a hard limit. Staging is capped at 50 notes and the **oldest are dropped when it
overflows**. Merging is due at 10, so there is room, but a project that never merges will start
losing its earliest notes.

## Running it

```
python scripts/consolidate.py <project>            # dry run: prints the plan, changes nothing
python scripts/consolidate.py <project> --apply    # performs the merge
```

The dry run prints which notes would go to which wiki file, so you can see the routing before
anything is written. Routing is by the `kind` you gave a note: a note deposited as `rag:gotcha` goes
to the page for the `rag` module. Notes whose kind matches no page are kept in staging and reported,
rather than being forced somewhere.

## What it does to each page

1. Commits the wiki directory as it stands, so the merge can be undone with one command.
2. Asks the local model (`granite4:3b` by default) to rewrite the page with the notes worked in as
   prose, removing anything they contradict.
3. Checks the result before writing it: every heading from the original page is still present, the
   length is between 60% and 160% of the original, each note's content is actually in the text, and
   code fences are still balanced.
4. If those checks fail, the notes are appended under a `## Consolidated learnings` heading instead
   of being merged. Nothing is lost, but the page is not tidied.
5. Commits again, deletes the merged notes from staging, and re-indexes the wiki layer so searching
   sees the new text.

The check step matters more than it sounds: a 3B model asked to rewrite a page will sometimes drop a
section or invent a summary of it. The checks are what turn that from data loss into a fallback.

## Rolling back

```
git -C ~/.hermes/wikis/<project> log --oneline
git -C ~/.hermes/wikis/<project> reset --hard <the commit before the merge>
```

`--apply` always commits before writing, so the state before a merge is one commit back.

## Known rough edges

- When a merge falls back, the note text is appended **verbatim**, including whatever phrasing it was
  deposited with. Write notes as finished sentences.
- The local model's rewrite loses paragraph line-wrapping, arriving as long single lines.
- Nothing checks whether an older note is still true. A note written before a rename will carry the
  old name into the page. Merging stale notes is how a wiki acquires wrong details.