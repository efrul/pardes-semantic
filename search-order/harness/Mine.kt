import io.github.kdroidfilter.seforimapp.features.search.semantic.EmbeddingProvider
import io.github.kdroidfilter.seforimapp.features.search.semantic.IndependentSemanticSearch
import io.github.kdroidfilter.seforimapp.features.search.semantic.OnnxEmbeddingProvider
import org.apache.lucene.document.IntPoint
import org.apache.lucene.index.DirectoryReader
import org.apache.lucene.index.ReaderUtil
import org.apache.lucene.index.VectorSimilarityFunction
import org.apache.lucene.search.IndexSearcher
import org.apache.lucene.store.FSDirectory
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.file.Path
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

// The lines the pack's model puts first for each query over the whole library, the dense side alone as the app searches
// it, with the score of the line known to answer the query: what hard negatives are mined from
// (../../mine_hard_negatives.py). No line is embedded: the library's vectors are the pack's index.
//
// args: seforim.db, queries.jsonl ({"id", "query", "line"}), out.jsonl, lines a query (100), threads (4)
// A row of out.jsonl: {"id", "pos": the score of "line" (absent when the index holds no vector of it, a short line
// merged into the next), "hits": [[lineId, score, bookId], ...]}, the scores Lucene's ((1 + cosine) / 2). The rows come
// in the order they end, not the queries' order.
private class Keeping(
    private val model: OnnxEmbeddingProvider,
) : EmbeddingProvider {
    // The last query's vector, as the search took it
    var last: ByteArray = ByteArray(0)

    override val dimension: Int get() = model.dimension

    override fun embed(text: String): ByteArray = model.embed(text).also { last = it }

    override fun close() = model.close()
}

fun main(args: Array<String>) {
    val db = Path.of(args[0])
    val indexPath = Path.of("$db.lucene")
    val k = args[3].toInt()
    val threads = args[4].toInt()
    val spec = IndependentSemanticSearch.findOpenModel(indexPath)!!
    val vectorDir = IndependentSemanticSearch.openVectorIndexDir(indexPath)
    val reader = DirectoryReader.open(FSDirectory.open(vectorDir))
    val searcher = IndexSearcher(reader)

    // A line's vector as the index holds it (a base book's line in vec_base, any other in vec)
    fun vector(lineId: Long): ByteArray? {
        for (sd in searcher.search(IntPoint.newExactQuery("line_id", lineId.toInt()), 2).scoreDocs) {
            val leaf = reader.leaves()[ReaderUtil.subIndex(sd.doc, reader.leaves())]
            val local = sd.doc - leaf.docBase
            for (field in listOf("vec", "vec_base")) {
                val values = leaf.reader().getByteVectorValues(field) ?: continue
                val docs = values.iterator()
                if (docs.advance(local) == local) return values.vectorValue(docs.index()).copyOf()
            }
        }
        return null
    }

    val queries = File(args[1]).readLines().filter { it.isNotBlank() }.map { JSONObject(it) }
    val done = AtomicInteger()
    val pool = Executors.newFixedThreadPool(threads)
    File(args[2]).bufferedWriter().use { out ->
        repeat(threads) { t ->
            pool.execute {
                // A model and a search of its own for each thread
                val provider = Keeping(OnnxEmbeddingProvider.open(spec))
                val dense = IndependentSemanticSearch(vectorDir, provider, spec.hybridDenseWeight)
                for (i in t until queries.size step threads) {
                    val q = queries[i]
                    val row = JSONObject().put("id", q.get("id"))
                    runCatching {
                        val hits = dense.search(q.getString("query"), k)
                        vector(q.getLong("line"))?.let { row.put("pos", VectorSimilarityFunction.COSINE.compare(provider.last, it).toDouble()) }
                        row.put("hits", JSONArray(hits.map { JSONArray().put(it.lineId).put(it.score.toDouble()).put(it.bookId) }))
                    }.onFailure { row.put("error", it.javaClass.simpleName) }
                    synchronized(out) {
                        out.write(row.toString())
                        out.newLine()
                    }
                    val n = done.incrementAndGet()
                    if (n % 2000 == 0) System.err.println("$n of ${queries.size}")
                }
                dense.close()
            }
        }
        pool.shutdown()
        pool.awaitTermination(24, TimeUnit.HOURS)
    }
    reader.close()
}
