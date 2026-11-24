import io.github.kdroidfilter.seforimapp.features.search.semantic.OnnxEmbeddingProvider
import io.github.kdroidfilter.seforimapp.features.search.semantic.OpenModelSpec
import java.io.File
import java.nio.file.Path

// The queries of a file, a line each, as the app's model embeds a query (the float vector, before the int8 of the
// search): a line of numbers for each, in the file's order. For checking a threshold found on the PyTorch model's
// vectors against the app's own (eval-pool/probes/marks_memory.py --queries / --vectors).
//
// args: pardes-model dir, queries.txt, out.txt
fun main(args: Array<String>) {
    val spec = OpenModelSpec.read(Path.of(args[0])) ?: error("no pardes-model.json in ${args[0]}")
    OnnxEmbeddingProvider.open(spec).use { model ->
        File(args[2]).bufferedWriter().use { out ->
            File(args[1]).readLines().forEach { query ->
                out.write(model.embedQuery(query).joinToString(" "))
                out.newLine()
            }
        }
    }
}
