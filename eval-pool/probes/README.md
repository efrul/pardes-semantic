<div dir="rtl">

# בדיקות חד־פעמיות (2026-11-23, 2026-11-24)

הסקריפטים של החקירה ב־[`../results-why.md`](../results-why.md): למה הזוגות השפוטים לא שיפרו בהרבה, ומה כן משפר.
`st_prompts.py`, `fuse_ranks.py` ו־`judged_pairs.py` הם של ההשוואה למודלים של גוגל ([`../results-google.md`](../results-google.md)).
הם נשמרים כדי שאפשר יהיה לחזור על המספרים. אף אחד מהם אינו חלק מהצנרת.

- **מריצים מהשורש של המאגר,** במעבד.
- **הקלט מחוץ למאגר,** תחת `build/`: המפתח של החיפושים השפוטים (`build/judge-pick/key.tsv`), הדף כמו שהשופטים ראו
  אותו (`build/judge/run.jsonl`, `build/demo/run.jsonl`), הספרייה (`build/library/`), והפלט של הקרנלים
  (`build/kaggle-output/`). החיפושים של המשתמשים והטקסט של השורות לא נכנסים למאגר.

| קובץ | מה הוא בודק |
|---|---|
| [`page_rerank.py`](page_rerank.py) | שש השורות של הדף בסדר של מדרג (קובצי דירוג של `pool_rerank.py`), מול הדף כמו שהוא |
| [`page_rerank_onnx.py`](page_rerank_onnx.py) | קובץ ONNX של מדרג במעבד, על אותם דפים: הזמן לדף, וקובץ דירוג ל־`page_rerank.py` |
| [`OrtScore.java`](OrtScore.java) | אותו קובץ ב־ONNX Runtime של האפליקציה (`java -cp <onnxruntime.jar> OrtScore.java model.onnx ids.tsv scores.tsv 4`) |
| [`cut_vocabulary.py`](cut_vocabulary.py) | קובץ ה־ONNX של המדרג רק עם האסימונים שהספרייה משתמשת בהם, ובדיקה שהציונים לא השתנו |
| [`rerank_tune.py`](rerank_tune.py) | מדרג קטן שמכוונן על הציונים של שני שלישים מהחיפושים השפוטים, על השליש שלא ראה |
| [`query_vectors.py`](query_vectors.py) | וקטורי השאילתות כמו שהאפליקציה מחשבת אותם (`build/why/qvec-int8.npz`) |
| [`memory.py`](memory.py) | זיכרון של החיפושים השפוטים: כל חיפוש מול כל האחרים, וכמה הוא מדבר במקומות אחרים |
| [`vectors.py`](vectors.py) | הווקטורים של מודל לשאילתות ולמועמדים של מאגר, קלט לשתי הבדיקות הבאות |
| [`projection.py`](projection.py) | הטלה ליניארית על וקטורים קפואים, מהציונים של השופטים |
| [`distill_map.py`](distill_map.py) | מפה ליניארית על וקטורים קפואים, מהציונים של מדרג גדול על חיפושים שלא נשפטו |
| [`st_prompts.py`](st_prompts.py) | מודל ממשפחה אחרת עם הוראות אחרות לפני השאילתה והשורה ([`../results-google.md`](../results-google.md)) |
| [`fuse_ranks.py`](fuse_ranks.py) | קובצי דירוג של כמה מודלים כאחד: הציונים מתוקננים ומחוברים |
| [`sefaria/`](sefaria/README.md) | הנתונים של ספריא: הנושאים, המקורות שלהם, והחיבור לשורות של הספרייה ([`../results-sefaria.md`](../results-sefaria.md)) |
| [`torah-pairs/`](torah-pairs/README.md) | Torah-pairs: החיבור לשורות של הספרייה, ומה `tune3` יודע ממנו ([`../results-torah-pairs.md`](../results-torah-pairs.md)) |
| [`marks_reach.py`](marks_reach.py) | כמה מהחיפושים חוזרים אצל אותו אדם, וכמה מהם עשה גם אדם אחר: לאן זיכרון של סימונים היה מגיע |
| [`marks_transfer.py`](marks_transfer.py) | כשאותו צורך חוזר, האם הסימונים של הפעם הקודמת נכונים גם עכשיו |
| [`votes.py`](votes.py) | סימונים על התוצאות של דף, והדף הבא בסדר של השאילתה המוזזת לפיהם ([`../results-votes.md`](../results-votes.md)) |
| [`votes_app.py`](votes_app.py) | אותו כלל במנועים של התוכנה, על כל הספרייה: כותב את החיפושים ל־[`Votes.kt`](../../search-order/harness/Votes.kt) ומודד את מה שהוא מחזיר |
| [`marks_memory.py`](marks_memory.py) | זיכרון של הסימונים של כל אדם, לפי סדר הזמנים: באיזה חיפושים הוא היה פועל, ומה הכלל שלו עושה שם |
| [`line_lengths.py`](line_lengths.py) | אורכי השורות של הספרייה מול מה שהמודל קורא, ואורכי השורות שקיבלו ציון ([`../results-sefaria.md`](../results-sefaria.md), Patot) |
| [`judged_pairs.py`](judged_pairs.py) | מודלים זוג מול זוג על החיפושים השפוטים, עם המקורות הידועים: טובה ראשונה, מקור ראשון, ומאיזה חלק של הספרייה |

</div>
