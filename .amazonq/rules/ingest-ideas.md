# Ingest Ideas Rule

When the user says "ingest ideas" or "ingest":

1. Read `ideas.txt` in the project root
2. Read `TODO.md` in the project root
3. For each new idea in `ideas.txt` (skip any already present in TODO.md):
   - Categorize by effort (Trivial, Easy, Easy–Medium, Medium, Medium–Hard, Hard)
   - Add as an unchecked `- [ ]` item under the appropriate section in `TODO.md` with a bold title and one-line description
4. After ingesting, clear all idea lines from `ideas.txt` (keep only the timestamp header)
5. Update the timestamp in `ideas.txt` to the current date/time: `# Last ingested to TODO.md on YYYY-MM-DD HH:MM:SS`
