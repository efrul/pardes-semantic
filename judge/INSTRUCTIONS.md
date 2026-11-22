# Judging search results of a Torah library

You grade the lines a search engine returned for real searches that users of a Jewish-texts library typed (Tanakh,
Mishnah, Talmud, halacha, midrash, later works). The grades become training data for the search's model: the good lines
are what it should bring first, the others are what it wrongly brought.

Everything stays on this machine. Do not use the web, do not send any text anywhere, do not edit or create files other
than the two output files named in your task, and do not read any file other than the ones named in your task.

## The batch file

Each search is a block:

```
## <search id>. <the query as the user typed it>
1. [<line id>] <reference> (<words>w): <the line's text, or its beginning and the part nearest the query's words>
2. ...
```

Queries are short, in today's Hebrew, sometimes with typing errors, sometimes a quote from memory, sometimes Aramaic or
English. Read the query for what the person was looking for. A long line is shown in part (`[…]`); to read a line whole:

```
python judge/lookup.py --line <line id>
```

Run commands from the root of this repository.

## The grades

Grade every line shown, each on its own, 0 to 3:

- **3, the source.** The primary place the search is about: the verse, the mishnah, the Gemara line, the halacha in
  the Rambam or the Shulchan Arukh that states it. For a quote, the earliest book that holds the quoted words. For a
  practical question, a line that states the ruling asked about plainly.
- **2, answers.** The line answers the search or says what was looked for, though it is not the primary source: a
  commentary that explains the matter, a later book that brings the halacha, a parallel, a line that quotes the source
  and discusses it. Give 2 to a line that truly answers even when it is not the source you had in mind.
- **1, related.** The same subject, but it does not answer: a neighbouring law, a general statement, a heading, a bare
  pointer ("see siman 272") with nothing said, a line that mentions the matter in passing.
- **0, not relevant.** Shares words or nothing at all with what was asked.

Rules that past judging showed are needed:
- Read exactly what is asked. "Milk spilled into the pot" is not "milk spilled out of the pot". A line on the same
  sugya that speaks of another case is 1, not 3.
- A heading or a title alone is at most 1, however well its words fit.
- Shared words are not an answer: a midrash that happens to use the query's words gets 0 for a halachic question.
- When several lines hold the same quote, the earliest level (Tanakh, then Mishnah and Talmud, then Rishonim, then
  later) is 3 and the ones quoting it are 2.
- The difference between 2 and 3 matters less than the difference between 1 and 2: decide first whether the line is
  good (2 or 3) or not (0 or 1).
- Users of the library are more generous than past judges: a line that gives the reader what was asked, even in part
  or through a close case, they call good. When you hesitate between 1 and 2 for such a line, give 2. Keep 1 for a
  line that is on the subject and gives no answer.

## The status of each search

- `ok`: you understood the need and graded the lines.
- `noise`: not a question about the library's texts (a question about the software, a test string, everyday web
  search). Grade nothing.
- `unclear`: you cannot tell what the person was looking for. Still grade 0 the lines that are plainly irrelevant to
  any reading of the query, and leave the others out.
- `none`: the need is clear, no line shown is good (2 or 3), and you found no source to add.

## A source from your own knowledge

Only when your task says to add sources. If you know where the matter is stated (for example "that is Bava Batra 12b",
"Shulchan Arukh Orach Chaim 188:8") and that place is not among the lines shown, find it in the library and add it:

```
python judge/lookup.py --book "בבא בתרא" --words "שתי לבבות"
python judge/lookup.py --ref "שולחן ערוך, אורח חיים קפח" --words שלישית
python judge/lookup.py --book "משנה תורה, הלכות מלכים" --words "שש עריות"
```

`--words` ignores niqqud and punctuation; each word or quoted phrase must be in the line, so give few and distinctive
ones, spelled as the book spells them (try a second spelling when nothing is found). `--ref` is the beginning of a
reference as the batch file writes references.

- Add a line only after you read it in the tool's output and it says what the search asks. Never write a line id you
  did not see in the tool's output.
- At most two added lines a search, the primary source first. Grade them as any line (usually 3).
- Add the primary source even when a shown line is already good, if you know it and it is not shown.
- Give up after three lookups for one search: do not guess.

## Output

Two tab-separated files, with these header lines.

Grades, one row per graded line:

```
qid	lineId	grade	from
```

`from` is `page` for a line shown in the batch, `knowledge` for a line you added.

Notes, one row per search of the batch, every search:

```
qid	status	note
```

`note`: a few words, in English: what the person was looking for, and for an added source, what it is. No tabs or
line breaks in it.

Work through the batch in order. Write the files as you go (after every few searches), so that nothing is lost.
When done, reply with counts only: searches by status, lines graded by grade, lines added from knowledge.
