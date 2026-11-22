package io.github.kdroidfilter.seforimlibrary.core.models

// SeforimLibrary's Category, as far as the app's semantic package names it (MeaningFacets.kt: its id and its order):
// the harness compiles that package without SeforimLibrary's core module, as LoggerStub.kt stands for the app's logger.
data class Category(
    val id: Long,
    val order: Int,
)
