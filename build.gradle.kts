plugins {
    alias(libs.plugins.kotlin.multiplatform) apply false
    alias(libs.plugins.android.kmp.library) apply false
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.maven.publish) apply false
}

allprojects {
    group = "cn.enaium.onnxruntime"
    // Version scheme: <onnxruntime-version>.<revision> — bump the revision
    // for each new build against the same ONNX Runtime version.
    version = "1.28.0.1"
}
