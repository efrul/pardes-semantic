<div dir="rtl">

# הבדיקות של הנתונים של ספריא (2026-11-24)

הסקריפטים שמאחורי [`../../results-sefaria.md`](../../results-sefaria.md). חד־פעמיים, ואינם חלק מהצנרת.

- **מריצים מהשורש של המאגר,** במעבד, עם `python -I` לאלה שקוראים רק את הגיבוי.
- **הגיבוי של ספריא** אינו במאגר. ספריא מבקשת לקחת נתונים בכמות ממנו ולא מה־API:

  ```
  mkdir -p build/sefaria/dump && cd build/sefaria/dump
  curl -L -O https://storage.googleapis.com/sefaria-mongo-backup/dump_small.tar.gz        # 2.7GB
  tar -xzf dump_small.tar.gz ./dump/sefaria/{topics,topic_links,topic_link_types,topic_data_sources,index,passage,lexicon,lexicon_entry}.bson
  pip install pymongo                                                                     # בשביל bson
  ```

- **הקלט הפרטי** תחת `build/`, כמו בשאר הבדיקות: הספרייה, ההיסטוריה שהמשתמשים שלחו, והמפתח של החיפושים שנשפטו.

| קובץ | מה הוא בודק |
|---|---|
| [`links.py`](links.py) | הקישורים מנושא למקור: כמה, מאיזה סוג ומאיזה מקור. כותב `build/sefaria/ref_topic_rows.pkl` |
| [`about.py`](about.py) | הקישורים של "עוסק ב": הנושאים שלהם, אורך השמות, הספרים |
| [`maprefs.py`](maprefs.py) | מראי המקום של ספריא מתורגמים לשלנו, לפי הכללים של המחולל של הספרייה. כותב `build/sefaria/about_pairs.pkl` |
| [`overlap.py`](overlap.py) | בכמה מהזוגות אין לשורה אף מילה משם הנושא |
| [`match.py`](match.py) | כמה מהחיפושים של המשתמשים זהים לשם של נושא |
| [`hard.py`](hard.py) | באיזה מקום `tune3` שם את המקור של נושא לפי שמו, וכמה החיפושים של המשתמשים קרובים לנושאים |
| [`agree.py`](agree.py) | כשחיפוש שנשפט קרוב לנושא: האם המקורות של הנושא הם השורות שקיבלו ציון טוב |
| [`embed_pairs.py`](embed_pairs.py) | הווקטורים של `tune3` למדגם של זוגות, עד שמונה לנושא, בלי שורות שמבחן מדרג |
| [`topic_map.py`](topic_map.py) | מטריצה אחת מעל הווקטורים הקפואים שנלמדת מהזוגות, וקובצי דירוג של המבחנים שלנו איתה |

```
python -I eval-pool/probes/sefaria/links.py
python -I eval-pool/probes/sefaria/about.py
python -I eval-pool/probes/sefaria/maprefs.py
python eval-pool/probes/sefaria/overlap.py
python -I eval-pool/probes/sefaria/match.py
python eval-pool/probes/sefaria/hard.py
python eval-pool/probes/sefaria/agree.py
python eval-pool/probes/sefaria/embed_pairs.py 8
python eval-pool/probes/sefaria/topic_map.py [--origins sefaria,learning-team,sefer-haagada --out build/sefaria/ranks-curated]
python eval-crfh/measure.py build/sefaria/ranks/ranks-tune3-base.json build/sefaria/ranks/ranks-topics-base.json
```

</div>
