plugins {
    alias(libs.plugins.kotlin.multiplatform) apply false
    alias(libs.plugins.android.kmp.library) apply false
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.maven.publish) apply false
}

allprojects {
    group = "cn.enaium.onnxruntime"
    // Project version (major.minor.revision); bump the revision number for
    // each release. The bundled ONNX Runtime version is set separately in
    // gradle/libs.versions.toml and the native build scripts.
    version = "1.0.2"
}
