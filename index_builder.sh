#!/usr/bin/env bash
# Builds Pardes's VectorIndexBuilder (SeforimApp/.../features/search/semantic/VectorIndexBuilder.kt) with a standalone
# Kotlin compiler, outside Gradle, which cannot resolve the Kotlin plugin in the cloud container: the builder needs only
# Lucene, org.json, sqlite-jdbc and Kotlin's library. The versions are those of search-order/build.sh.
#
#   index_builder.sh PARDES_DIR OUT_DIR
#   "$JAVA" -Xmx6g --add-modules=jdk.incubator.vector -cp "$(cat OUT_DIR/classpath.txt)" \
#       io.github.kdroidfilter.seforimapp.features.search.semantic.VectorIndexBuilder \
#       --vectors build/vectors-merged --db seforim.db --out pardes.vec.lucene
#
# Needs a JDK 21 (JAVA, or java on PATH) and curl. The jars come from the Gradle cache when they are there, from Maven
# Central otherwise.
set -euo pipefail

repo="$(cd "$1" && pwd)"
out="$2"
JAVA="${JAVA:-java}"
src="$repo/SeforimApp/src/jvmMain/kotlin/io/github/kdroidfilter/seforimapp/features/search/semantic/VectorIndexBuilder.kt"
[ -f "$src" ] || { echo "No VectorIndexBuilder.kt in $repo" >&2; exit 1; }
mkdir -p "$out"
out="$(cd "$out" && pwd)"
rm -rf "$out/classes"
mkdir -p "$out/classes" "$out/jars" "$out/kotlinc"

# group artifact version dir: the jar, from the Gradle cache or Maven Central
fetch() {
    local g="$1" a="$2" v="$3" dir="$4" f="$2-$3.jar" cached
    [ -f "$dir/$f" ] && return
    cached="$(find "$HOME/.gradle/caches/modules-2/files-2.1/$g/$a/$v" -name "$f" 2> /dev/null | head -1 || true)"
    if [ -n "$cached" ]; then
        cp "$cached" "$dir/"
    else
        # Maven Central answers 429 when asked too fast: a few tries, a few seconds apart
        for try in 1 2 3 4 5; do
            curl -fsSL -o "$dir/$f" "https://repo1.maven.org/maven2/${g//.//}/$a/$v/$f" && return
            sleep $((try * 5))
        done
        return 1
    fi
}

fetch org.apache.lucene lucene-core 10.5.2 "$out/jars"
fetch org.xerial sqlite-jdbc 3.50.3.0 "$out/jars"
fetch org.json json 20240303 "$out/jars"
fetch org.jetbrains.kotlin kotlin-stdlib 2.4.20 "$out/jars"
fetch org.jetbrains annotations 23.0.0 "$out/jars"

fetch org.jetbrains.kotlin kotlin-compiler-embeddable 2.4.20 "$out/kotlinc"
fetch org.jetbrains.kotlin kotlin-stdlib 2.4.20 "$out/kotlinc"
fetch org.jetbrains.kotlin kotlin-script-runtime 2.4.20 "$out/kotlinc"
fetch org.jetbrains.kotlin kotlin-reflect 2.4.10 "$out/kotlinc"
fetch org.jetbrains.kotlin kotlin-daemon-embeddable 2.4.20 "$out/kotlinc"
fetch org.jetbrains.intellij.deps trove4j 1.0.20200330 "$out/kotlinc"
fetch org.jetbrains annotations 13.0 "$out/kotlinc"
fetch org.jetbrains.kotlinx kotlinx-coroutines-core-jvm 1.8.0 "$out/kotlinc"

# The builder quantizes float vectors with OnnxEmbeddingProvider.quantizeInt8: that function alone, taken from the
# app's file as it is, and not the provider with ONNX Runtime
rm -rf "$out/stub"
mkdir -p "$out/stub"
{
    echo "package io.github.kdroidfilter.seforimapp.features.search.semantic"
    echo "import kotlin.math.abs"
    echo "import kotlin.math.roundToInt"
    echo "// OnnxEmbeddingProvider.quantizeInt8 only, copied from the app's OnnxEmbeddingProvider.kt by index_builder.sh"
    echo "class OnnxEmbeddingProvider private constructor() {"
    echo "    companion object {"
    awk '/fun quantizeInt8\(/ {p = 1} p {print} p && /^        }$/ {exit}' "$(dirname "$src")/OnnxEmbeddingProvider.kt"
    echo "    }"
    echo "}"
} > "$out/stub/OnnxEmbeddingProvider.kt"
grep -q "return out" "$out/stub/OnnxEmbeddingProvider.kt" || { echo "quantizeInt8 not found in OnnxEmbeddingProvider.kt" >&2; exit 1; }

cp_of() { local list=""; for j in "$1"/*.jar; do list="$list$j:"; done; printf '%s' "${list%:}"; }
"$JAVA" -Xmx2g -cp "$(cp_of "$out/kotlinc")" org.jetbrains.kotlin.cli.jvm.K2JVMCompiler -no-stdlib -no-reflect -nowarn \
    -jvm-target 21 -cp "$(cp_of "$out/jars")" -d "$out/classes" "$src" "$out/stub/OnnxEmbeddingProvider.kt"
printf '%s:%s' "$out/classes" "$(cp_of "$out/jars")" > "$out/classpath.txt"
echo "built: $out/classes; classpath in $out/classpath.txt"
