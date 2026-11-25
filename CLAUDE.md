# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

The model of Pardes's semantic search (חיפוש לפי משמעות): Dicta's open neodictabert-bilingual-embed, fine-tuned and
distilled on the library's links, exported to the int8 ONNX model the app loads. It also holds the pipeline around it:
the benchmark and evaluations, the Kaggle and Colab runs, the vectors and index of the whole library, and the pack.
The trained models are this repository's releases (`models.yml`, `release_model.py`). Until 2026-11-05 this was the
`semantic-model/` folder of efrul/Pardes, whose history came with it.

The app's side stays in Pardes:
- the search code in `SeforimApp/src/jvmMain/kotlin/io/github/kdroidfilter/seforimapp/features/search/semantic/`;
- the documents `SEMANTIC_SEARCH.md` and `docs/SEMANTIC_PLAN.md`;
- the pack's releases (`semantic-release.yml`, `semantic-*` tags);
- the skill `.claude/skills/pardes-semantic-search/`, which covers both sides. Read it first. It is in the `pardes`
  submodule, or at https://github.com/efrul/Pardes/blob/main/.claude/skills/pardes-semantic-search/SKILL.md.

Read `README.md` (pipeline), `BENCHMARK.md` (numbers) and `TRAINING_PLAN.md` / `RUN3_PLAN.md` (training) before
changing the model.

## Commands

```bash
pip install -r requirements.txt          # CPU torch; the CUDA wheel only for embedding the library
python eval-passages/test_passages.py    # the tests tests.yml runs on every push
python test_release_model.py
python feedback/test_collect.py
git submodule update --init pardes       # Pardes, only for search-order/build.sh, eval-100/harness, the fixtures
git -C pardes submodule update --init SeforimLibrary   # what build.sh compiles against
```

Not `--recursive`: of Pardes's submodules only SeforimLibrary is needed.

## Rules

- **The app's name:** Pardes (פרדס). It is based on Zayit by Elie (Elyahou) Gambache, who did not make it and is not
  responsible for it. Never call it Zayit/זית.
  - Keep the LICENSE as it is: AGPL v3 with Zayit's §7b term and his copyright line.
  - Keep the line "Powered by the technologies that drive Zayit — https://zayitapp.com/" word for word.
- **Nothing from Zayit's semantic search:** Zayit's semantic search is the private package
  `io.github.kdroidfilter:seforim-semantic` (the private SeforimEmbedding repository).
  - Nothing here comes from it or from its model.
  - Never use its code, its model or its tokens (`semantic.token`, `SEFORIM_SEMANTIC_TOKEN`, `-P*.local`), nor those
    of the siddur or the quote links.
- **The Kaggle account** that holds the kernels, datasets and packs is never named in this repository, its workflows,
  their logs or a release's files.
  - Name kernels and datasets without their owner.
  - The workflows take the owner from `KAGGLE_API_TOKEN` and mask it.
  - `release_model.py` rewrites a path that names the account and refuses the release when the name is anywhere else.
- **Data that stays out (the repository is public):**
  - the text of the library's lines: ids and references instead;
  - evaluation sets without a licence (crfh1672): ids, our line ids and results only;
  - קרית חנה's questions and answers: not passed on;
  - users' search histories and feedback packages: the inbox (`feedback/fetch.py`) is kept outside the repository.
- **The models' licences:**
  - Dicta's CC BY 4.0: `NOTICE.md` travels with every model. Pardes keeps a copy as `semantic-model/NOTICE.md`,
    which Pardes for Android shows in its licences: a change here is made there too.
  - The training data's licences and permissions are in `TRAINING_PLAN.md`, רישוי. A new source gets its line there,
    and its credit in `NOTICE.md`, before a model learns from it.
- **Save the working data after every step:** a file no repository may hold (pairs, kept-out tests, what a miner
  found, a reranker's scores, ranks) goes to the account's private Kaggle dataset `pardes-semantic-work` with
  `save_work.py`, before the next step: the session's container is reclaimed. Never a user's data. The skill's
  "Saving the working data" has the details.
- **The harness runs the `pardes` submodule's code** (`search-order/build.sh`, `eval-100/harness/`). Before a run that
  is to show what users see, move the submodule to Pardes's `main`: on 2026-11-22 it was 17 days behind, and the pages
  judged then were an older fusion's (`eval-pool/results-why.md`, the correction). `PARDES=<path>` builds from another
  checkout.
- **Git:**
  - Commit as `efrul <342689225+efrul@users.noreply.github.com>`, author and committer, with `-s`. Never commit as
    Claude or as a local identity.
  - The default branch is `main`.
  - Open a pull request only when asked.
