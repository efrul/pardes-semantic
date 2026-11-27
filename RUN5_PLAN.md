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
- **TPU:** ‏19.9 מ־20 השעות פנויות.
  - בגרסה 8 של הבדיקה (`pardes-semantic-tpuprobe`, ‏`build/kaggle/tpuprobe/probe.py` בסשן הקודם) torch_xla ראה את ה־TPU
    עם 8 התקנים.
  - היא נכשלה ב־import של sentence-transformers: ה־datasets של התמונה דורש huggingface-hub 1.31 ומעלה.
  - גרסה 9 מצמידה `datasets==3.6.0`, ונשלחה ב־27.11.
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

אם גרסה 9 מראה ששבב אחד מאמן לפחות כמו שני ה־T4 (‏11.6 שורות בשנייה), מעבירים את `tune` ל־TPU. כך שעות ה־GPU
נשארות להטמעה.

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
