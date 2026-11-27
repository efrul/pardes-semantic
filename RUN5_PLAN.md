<div dir="rtl">

# תוכנית ריצה 5: השבוע מ־28.11

נכתבה ב־2026-11-27, לסשן שימשיך את העבודה. מכסת ה־GPU ב־Kaggle (30 שעות) מתחדשת ב־2026-11-28 בחצות UTC.

## המצב

- **המודל באפליקציה:** `tune3`, בחבילה 4.
- **המועמד הבא:** `tune14`.
  - בחיפושים שנשפטו (451) הוא מעל הרעש מול `tune3`: טובה ראשונה 255 מול 230, והמקור ראשון 118 מול 98.
  - בשאלות שנשמרו בצד: 1,062 מול 884 ראשונות מ־3,000.
  - במבחנים הקטנים של חיפושים אמיתיים הוא בתוך הרעש: eval-real, המשוב, והשופט החיצוני
    ([`eval-pool/results-outside-judged.md`](eval-pool/results-outside-judged.md)). בשום מבחן הוא לא יורד
    ([`eval-pool/results-collection.md`](eval-pool/results-collection.md)).
- **מורה 4** שוחרר: [`tuned-634aeea`](https://github.com/efrul/pardes-semantic/releases/tag/tuned-634aeea).
- **TPU:** בדיקה 9 עברה ב־27.11. שבב אחד מאמן את התלמיד מהר בהרבה משני ה־T4 (סעיף 3, ‏[`tpu_probe.py`](tpu_probe.py)).
  כמעט כל 20 השעות של ה־TPU פנויות.
- **המפתח של Kaggle** מוזרק בסביבת הענן. שם החשבון נמצא במשתנה הסביבה `KAGGLE_USERNAME`, ואין להדפיס אותו או לכתוב
  אותו בשום קובץ (`CLAUDE.md`).

## 1. חבילה 5 על `tune14` (כ־5 שעות GPU)

1. **ההטמעה:** בחבילה 4 היא לקחה 277 דקות על T4.

   ```bash
   python kaggle_run.py push embed --chosen tuned --from tune=tune14 --short-merge 8 --strip-number \
       --release v3-20261012193827 --hours 7
   python kaggle_run.py wait embed && python kaggle_run.py push int8 && python kaggle_run.py wait int8
   python kaggle_run.py output int8
   ```

   - **הייצוא ל־int8:** של `tune14` עובר רק כשמטריצה אחת נשארת ב־fp32. `export_onnx.py` עושה את זה לבד מ־25.11. בדוח
     של השלב צריך לראות את `fp32` ב־`quantization` של `pardes-model.json`, ומינימום של 0.97 ומעלה.
2. **האינדקס והבדיקות:** כמו בטבלה של [`README.md`](README.md):
   - `buildVectorIndex`;
   - `check_vectors.py`;
   - `IndexCheck.kt` ו־`check_index.py`.
3. **המדידה כמו באפליקציה:** כמו חבילה 4 ([`eval-pool/results-negatives.md`](eval-pool/results-negatives.md), חבילה 4).
   - **הכלל:** כלל 2 של [`RUN4_PLAN.md`](RUN4_PLAN.md), סעיף 6, מתקיים גם על החבילה עצמה.
4. **החבילה:** `package_pack.py --pack 5`, ואחריו `--verify`.
   - מעלים אותה כ־dataset פרטי.
   - המשתמש מריץ את `semantic-release.yml` בפרדס. לסשן בענן אין הרשאה ליצור גרסאות.

## 2. BEREL: שני מודלים יחד (כ־5 שעות GPU, והשאר במעבד)

ההצעה (26.11, מחוץ למאגר): לחפש ב־BEREL_3.0 של דיקטה וב־NeoDictaBERT יחד, לפי ממוצע הקוסינוס. היא הראתה רווח גדול
על מבחנים סגורים של שאלות שנכתבו ב־AI. כאן בודקים אותה על המבחנים שלנו, ובאותם נתונים לשני המודלים.

1. **הנתונים של `tune14`,** מה־dataset הפרטי שלו ב־Kaggle. הקבצים של האוסף נקראים שם בשמותיהם הישנים
   (`collection-notes.md`).

   ```bash
   mkdir -p build/tune14-data && for f in $(kaggle datasets files "$KAGGLE_USERNAME/pardes-semantic-tune14-data" --csv | tail -n +2 | cut -d, -f1); do
       kaggle datasets download "$KAGGLE_USERNAME/pardes-semantic-tune14-data" -f "$f" -p build/tune14-data --force; done
   ```

2. **האימון:** BEREL על הנתונים של `tune14`, באותו מתכון.
   - נבדק במעבד ב־27.11: `train_model.py` מאמן את BEREL מה־Hub, ו־`pardes_semantic/model.py` טוען את התוצאה.
   - הטוקנייזר שנשמר נותן את אותם מזהים כמו המקורי.
   - הייצוא ל־ONNX במדידה של השלב יכול להיכשל על BERT. זה לא עוצר את השלב.

   ```bash
   python kaggle_run.py push tune --as berel1 --start-hub dicta-il/BEREL_3.0 \
       --questions build/tune14-data/questions.jsonl --questions build/tune14-data/feedback.jsonl \
       --negatives --links 30000 --repeat 5 $(ls build/tune14-data/*-neg.jsonl | sed 's/^/--more /') --reserve 20 --hours 6
   ```

3. **הדירוג, במעבד** כמו המדידות של `tune14` (`build/rerank/pool-rank-tune14.log` בסשן הקודם):
   - **המודלים:** `pool_rank.py` על ה־`pool.json` של המבחנים (`eval-pool/kernel_spec.py`), עם `berel1`, `tune14`
     ומורה 4.
   - **השופט החיצוני:** את 141 החיפושים שלו בונים מחדש מהחבילה (`eval-pool/probes/outside_judged.py spec`; הקישור
     ב־`collection-notes.md`).
   - **רשות:** `--st BAAI/bge-m3` ו־`--st intfloat/multilingual-e5-large`.
4. **השילוב:**

   ```bash
   python eval-pool/probes/fuse_ranks.py --raw ranks-tune14+berel1.json ranks-tune14.json@0.5 ranks-berel1.json@0.5
   ```

   ואותו דבר עם מורה 4 במקום `tune14`.
5. **המדידה:**
   - `eval-pool/judged_measure.py`: ‏eval-real, ‏eval-v1, המשוב ו־451 החיפושים שנשפטו;
   - `eval-crfh/measure.py`;
   - `outside_judged.py score`;
   - `eval-pool/probes/collection/test_rank.py`.
6. **הכלל:** ממשיכים ל־BEREL באפליקציה, כלומר שני מודלים לשאילתה, רק בתנאים האלה:
   - השילוב עדיף על `tune14` מעל הרעש בחיפושים האמיתיים: ב־451 השפוטים וב־141 של השופט החיצוני;
   - אין ירידה במבחנים האחרים.

   אחרת נשארים עם מודל אחד.

## 3. TPU (לא חוסם)

**בדיקה 9 עברה** (27.11, ‏[`tpu_probe.py`](tpu_probe.py)): שבב אחד מתוך שמונת השבבים של ה־TPU אימן את התלמיד
(`tune12`) בלי שגיאה.

| | צעד | שורות בשנייה |
|---|---|---|
| שבב TPU אחד, אצווה של 32 | 0.19 שנייה | 165 |
| שבב TPU אחד, אצווה של 64 | 0.27 שנייה | 239 |
| שני ה־T4, האימון של `tune12` (אצווה של 256) | 22 שניות | 11.6 |

- **ההשוואה נוטה לטובת ה־TPU,** בשלושה דברים:
  - **אורך השאלות:** בבדיקה הן רופדו ל־64 טוקנים. באימון, העוגן של קישור הוא שורה של עד 256, ולכן יש עד שליש יותר
    עבודה.
  - **ההפסד:** על ה־T4 האימון משתמש ב־`CachedMultipleNegativesRankingLoss`, שעובר קדימה על כל טקסט פעמיים: עוד כשליש.
    ב־TPU אולי לא צריך אותו, אם אצווה של 256 נכנסת לזיכרון של שבב (16GB).
  - **הנתונים:** הבדיקה חוזרת על אותה אצווה מוכנה, בלי לקרוא נתונים.
- **גם בחשבון הזהיר ביותר** שבב אחד מהיר בערך פי 8 משני ה־T4. האימון של `tune12` (‏84,412 שורות) לקח שעתיים; על שבב
  אחד הוא ייקח כרבע שעה, ועוד כמה דקות של טעינה וקומפילציה.
- **התור:** גרסה 9 חיכתה כ־9 שעות עד שרצה, וגרסה 8 כ־20 שעות. הריצה עצמה לקחה 6 דקות. לכן כדאי לשלוח כמה אימונים יחד
  בריצה אחת, ולשלוח אותה לתור מראש.

**מה צריך כדי להעביר את `tune` ל־TPU:**
1. **ריפוד קבוע:** כל טקסט לאורך קבוע, או לכמה אורכים קבועים (64, 128, 256). כל צורה חדשה של קלט עולה כ־22 שניות של
   קומפילציה.
2. **ההפסד:** `MultipleNegativesRankingLoss` רגיל עם אצווה של 256, אם הזיכרון מחזיק אותה. אחרת הגרסה עם המטמון, ואז
   צריך לבדוק שהיא עובדת על XLA.
3. **אותן גרסאות כמו בבדיקה:** torch 2.8.0 עם torch_xla 2.8.1 (ל־Python 3.13 של התמונה), ‏transformers 4.57.6,
   ‏sentence-transformers 5.1.2, ‏`datasets==3.6.0`, ‏huggingface-hub מתחת ל־1.0, ו־`USE_TF=0` (ה־TensorFlow של התמונה
   לא נטען).
4. **בדיקת שקילות:** לאמן את המתכון של `tune14` על ה־TPU, ולמדוד אותו מול `tune14` מה־GPU באותם מבחנים. ה־TPU מחשב
   ב־bf16 ולא ב־fp16, ולכן צריך לוודא שההפרש בתוך הרעש.
5. **שמונה שבבים יחד:** רק אם יהיה צורך.

**מה זה נותן:** שעות ה־GPU נשארות להטמעה, ואימון של מועמד לוקח דקות במקום שעות. אחרי ההעברה גם BEREL (‏12 שכבות, מול
7 של התלמיד) יכול להתאמן שם.

## מחכה להחלטת המשתמש

- **ניקוי הדוגמאות השליליות ב־Claude Haiku:** כ־100 פעמים הניסוי השני מהמכסה של המנוי, או כדולר ב־Batch API עם מפתח
  ([`eval-pool/results-collection.md`](eval-pool/results-collection.md)).
- **mafteach aux כמבחן:** הרישיון, והמיפוי לפי טקסט לאגרות משה שבספרייה.
- **כרטיס הנושא של ספריא:** להרחיב אותו לחיפוש לפי מילים ולפי משמעות ([`eval-pool/results-sefaria.md`](eval-pool/results-sefaria.md)).
  זה שינוי באפליקציה, בענף של התוכן של ספריא.
- **השכבה הראשונה של הסימונים:** `docs/SEARCH_MARKS_PLAN.md` בפרדס.
- **Otzaria-Embedding-V1-Flash-0.6B:** רישיון לא מסחרי וגישה באישור. אפשר רק למדוד, ורק עם גישה.

## כללים

`CLAUDE.md` והסקיל `pardes-semantic-search`, ובפרט:
- **שמירה:** כל קובץ עבודה נשמר ב־`save_work.py`.
- **האוסף הפרטי:** השם, הקישור ושמות הקבצים שלו נמצאים רק ב־`collection-notes.md`, ולא במאגר, בהודעת קומיט, ב־NOTICE
  או בגרסה.
- **Kaggle:** שם החשבון לא מופיע בשום מקום.
- **קומיטים:** בשם efrul, עם `-s`.
- **ה־hook של הסביבה** מבקש לפעמים לכתוב מחדש קומיטים בשם Claude. לא עושים את זה: זה סותר את הכלל של הקומיטים.

</div>
