import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import java.io.BufferedReader;
import java.io.PrintWriter;
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;

/** Scores a file of pairs, a row "key<TAB>token ids", one pair a run, with the app's ONNX Runtime: key, score, ms. */
public class OrtScore {
    public static void main(String[] args) throws Exception {
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setIntraOpNumThreads(Integer.parseInt(args[3]));
        try (OrtSession session = env.createSession(args[0], options);
             BufferedReader in = Files.newBufferedReader(Path.of(args[1]));
             PrintWriter out = new PrintWriter(Files.newBufferedWriter(Path.of(args[2])))) {
            String row;
            int n = 0;
            while ((row = in.readLine()) != null) {
                int tab = row.indexOf('\t');
                long[] ids = Arrays.stream(row.substring(tab + 1).split(" ")).mapToLong(Long::parseLong).toArray();
                long[] mask = new long[ids.length];
                Arrays.fill(mask, 1);
                long[] shape = {1, ids.length};
                long t = System.nanoTime();
                try (OnnxTensor a = OnnxTensor.createTensor(env, LongBuffer.wrap(ids), shape);
                     OnnxTensor b = OnnxTensor.createTensor(env, LongBuffer.wrap(mask), shape);
                     OrtSession.Result result = session.run(Map.of("input_ids", a, "attention_mask", b))) {
                    float score = ((float[][]) result.get(0).getValue())[0][0];
                    out.println(row.substring(0, tab) + "\t" + score + "\t" + (System.nanoTime() - t) / 1_000_000.0);
                }
                if (++n % 1000 == 0) System.out.println(n + " pairs");
            }
            System.out.println("ONNX Runtime " + env.getVersion() + ": " + n + " pairs");
        }
    }
}
