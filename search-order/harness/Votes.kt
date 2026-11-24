import io.github.kdroidfilter.seforimapp.features.search.semantic.EmbeddingProvider
import io.github.kdroidfilter.seforimapp.features.search.semantic.HybridSearchEngine
import io.github.kdroidfilter.seforimapp.features.search.semantic.IndependentSemanticSearch
import io.github.kdroidfilter.seforimapp.features.search.semantic.LibraryLines
import io.github.kdroidfilter.seforimapp.features.search.semantic.OnnxEmbeddingProvider
import io.github.kdroidfilter.seforimlibrary.search.LineHit
import io.github.kdroidfilter.seforimlibrary.search.LuceneSearchEngine
import kotlinx.coroutines.runBlocking
import org.apache.lucene.document.IntPoint
import org.apache.lucene.index.DirectoryReader
import org.apache.lucene.index.ReaderUtil
import org.apache.lucene.search.IndexSearcher
import org.apache.lucene.store.FSDirectory
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.file.Path
import java.sql.DriverManager
import kotlin.math.sqrt

// What marks on a search's first lines do to the lines after them, in the app's own engines over the whole library
// (eval-pool/results-votes.md, eval-pool/probes/votes_app.py). Each search runs as the app runs it; its first lines are
// marked by their grades (2 or 3 relevant, 0 or 1 not), the query's vector is moved toward the relevant lines and away
// from the others (Rocchio, on the index's own int8 vectors), and the search runs again with the moved vector on its
// dense side and the same words: q' = q + POS * mean(relevant) - NEG * mean(not relevant).
//
// args: seforim.db, searches.jsonl ({"key", "query", "grades": {"<lineId>": 0..3}, "book": id or absent, "unknown": the
//       grade of a first line that has none, or absent}), out.jsonl, the first lines marked (6), POS (0.75), NEG (0.15) [one]
// one: a single mark, the first relevant line of the first lines, or the first line as not relevant when none is.
// A search with "book" runs in that book alone; any other in the base books, or in all books when the words find none
// there. A search none of whose first lines is graded is written without "moved".
private class Fixed(
    private val model: OnnxEmbeddingProvider,
    private val vector: ByteArray,
) : EmbeddingProvider {
    override val dimension: Int get() = model.dimension

    override fun embed(text: String): ByteArray = vector

    override fun embedQuery(text: String): FloatArray = model.embedQuery(text)

    // The model is the run's, closed at its end
    override fun close() = Unit
}

private const val LIST = 400
private const val DENSE = 150

fun main(args: Array<String>): Unit =
    runBlocking {
        val db = Path.of(args[0])
        val indexPath = Path.of("$db.lucene")
        val first = args[3].toInt()
        val pos = args[4].toFloat()
        val neg = args[5].toFloat()
        val one = args.getOrNull(6) == "one"
        val lexical = LuceneSearchEngine(indexPath, null, dictionaryPath = indexPath.resolveSibling("lexical.db"))
        val spec = IndependentSemanticSearch.findOpenModel(indexPath)!!
        val model = OnnxEmbeddingProvider.open(spec)
        val vectorDir = IndependentSemanticSearch.openVectorIndexDir(indexPath)
        val library = LibraryLines.beside(indexPath)
        val con = DriverManager.getConnection("jdbc:sqlite:$db")
        val st = con.prepareStatement("SELECT l.bookId, l.lineIndex, l.content, b.title FROM line l JOIN book b ON b.id = l.bookId WHERE l.id = ?")

        fun resolve(lineId: Long): LineHit? =
            synchronized(st) {
                st.setLong(1, lineId)
                st.executeQuery().use { rs ->
                    if (!rs.next()) return null
                    LineHit(rs.getLong(1), rs.getString(4), lineId, rs.getInt(2), "", 0f, rs.getString(3))
                }
            }

        // The dense sides opened, to close each after its search: an engine's own close would close the words' too
        val opened = ArrayList<IndependentSemanticSearch>()

        fun engine(provider: EmbeddingProvider) =
            HybridSearchEngine(lexical, indexPath, { id, _ -> resolve(id) }, true, library) {
                IndependentSemanticSearch(vectorDir, provider, spec.hybridDenseWeight).also { opened += it }
            }

        suspend fun list(
            engine: HybridSearchEngine,
            query: String,
            book: Long?,
            baseOnly: Boolean,
        ): List<Long> =
            engine
                .openSession(query, 5, bookIds = book?.let { listOf(it) }, baseBookOnly = baseOnly)
                ?.use { it.nextPage(LIST, snippets = false)?.hits }
                .orEmpty()
                .map { it.lineId }

        // A line's vector as the index holds it (VectorIndexBuilder: a base book's line in vec_base, any other in vec),
        // at length 1: the index's int8 vectors each have a scale of their own
        val reader = DirectoryReader.open(FSDirectory.open(vectorDir))
        val searcher = IndexSearcher(reader)

        fun unit(lineId: Long): FloatArray? {
            for (sd in searcher.search(IntPoint.newExactQuery("line_id", lineId.toInt()), 2).scoreDocs) {
                val leaf = reader.leaves()[ReaderUtil.subIndex(sd.doc, reader.leaves())]
                val local = sd.doc - leaf.docBase
                for (field in listOf("vec", "vec_base")) {
                    val values = leaf.reader().getByteVectorValues(field) ?: continue
                    val docs = values.iterator()
                    if (docs.advance(local) != local) continue
                    val bytes = values.vectorValue(docs.index())
                    val norm = sqrt(bytes.sumOf { it.toDouble() * it }).toFloat()
                    if (norm > 0f) return FloatArray(bytes.size) { bytes[it] / norm }
                }
            }
            return null
        }

        fun mean(vectors: List<FloatArray>): FloatArray = FloatArray(vectors[0].size) { i -> vectors.sumOf { it[i].toDouble() }.toFloat() / vectors.size }

        val plain = engine(model)
        val dense = IndependentSemanticSearch(vectorDir, model, spec.hybridDenseWeight)
        // Its dense side is the first opened, the one that stays open
        check(plain.denseReady()) { "No model or vector index beside $indexPath" }
        var done = 0
        File(args[2]).bufferedWriter().use { out ->
            File(args[1]).readLines().filter { it.isNotBlank() }.forEach { line ->
                val search = JSONObject(line)
                val query = search.getString("query")
                val book = if (search.has("book")) search.getLong("book") else null
                val grades = search.getJSONObject("grades").let { g -> g.keySet().associate { it.toLong() to g.getInt(it) } }
                // What a first line no one graded counts as, when the search says ("unknown")
                val unknown = if (search.has("unknown")) search.getInt("unknown") else null
                val errors = JSONObject()
                val baseOnly =
                    book == null &&
                        runCatching { (lexical.computeFacets(query, 5, baseBookOnly = true)?.totalHits ?: 0L) > 0 }
                            .getOrElse {
                                errors.put("facets", it.javaClass.simpleName)
                                false
                            }
                val row = JSONObject().put("key", search.getString("key")).put("scope", if (book != null) "book" else if (baseOnly) "base" else "all")
                val before = runCatching { list(plain, query, book, baseOnly) }.getOrElse {
                    errors.put("list", it.javaClass.simpleName)
                    emptyList()
                }
                row.put("list", JSONArray(before))
                val page = before.take(first)
                var relevant = page.filter { ((grades[it] ?: unknown) ?: -1) >= 2 }
                var others = page.filter { id -> (grades[id] ?: unknown)?.let { it <= 1 } == true }
                if (one) {
                    if (relevant.isNotEmpty()) {
                        relevant = relevant.take(1)
                        others = emptyList()
                    } else {
                        others = others.take(1)
                    }
                }
                val towards = relevant.mapNotNull(::unit)
                val away = others.mapNotNull(::unit)
                if ((towards.isNotEmpty() && pos != 0f) || (away.isNotEmpty() && neg != 0f)) {
                    val started = System.nanoTime()
                    val q = model.embedFloat(query)
                    val v = q.copyOf()
                    if (towards.isNotEmpty()) mean(towards).forEachIndexed { i, x -> v[i] += pos * x }
                    if (away.isNotEmpty()) mean(away).forEachIndexed { i, x -> v[i] -= neg * x }
                    val fixed = Fixed(model, OnnxEmbeddingProvider.quantizeInt8(v))
                    val after =
                        runCatching { list(engine(fixed), query, book, baseOnly) }
                            .getOrElse {
                                errors.put("moved", it.javaClass.simpleName)
                                emptyList()
                            }
                    // The dense side alone, as it is and with the marks: what the fusion with the words adds or hides
                    val scope = book?.let { listOf(it) }
                    row.put("dense", JSONArray(dense.search(query, DENSE, baseOnly, scope).map { it.lineId }))
                    val denseMoved = IndependentSemanticSearch(vectorDir, fixed, spec.hybridDenseWeight).also { opened += it }
                    row.put("denseMoved", JSONArray(denseMoved.search(query, DENSE, baseOnly, scope).map { it.lineId }))
                    opened.drop(1).forEach { runCatching { it.close() } }
                    while (opened.size > 1) opened.removeAt(opened.size - 1)
                    row
                        .put("relevant", JSONArray(relevant))
                        .put("others", JSONArray(others))
                        .put("moved", JSONArray(after))
                        .put("movedMs", (System.nanoTime() - started) / 1_000_000)
                }
                out.write(row.put("errors", errors).toString())
                out.newLine()
                out.flush()
                if (++done % 50 == 0) System.err.println("$done searches")
            }
        }
        reader.close()
        plain.close()
    }
