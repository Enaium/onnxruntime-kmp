plugins {
    alias(libs.plugins.kotlin.multiplatform) apply false
    alias(libs.plugins.android.kmp.library) apply false
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.maven.publish) apply false
}

allprojects {
    group = "cn.enaium.onnxruntime"
    // Version scheme: <onnxruntime-version>.<revision>; bump the revision for
    // each build against the same ONNX Runtime version. The bundled ONNX
    // Runtime version is also set in gradle/libs.versions.toml and the native
    // build scripts.
    version = "1.31.1.1"
}
