pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.PREFER_SETTINGS)
    repositories {
        google()
        mavenCentral()
        mavenLocal()
    }
}

rootProject.name = "onnxruntime-kmp"

include(":onnxruntime-kmp")
include(":example")
include(":example-android")

// Per-platform artifacts that ship the prebuilt libonnxruntime shared
// library (the klib cannot embed dynamic libraries).
listOf(
    "macosarm64",
    "macosx64",
    "linuxx64",
    "linuxarm64",
    "mingwx64",
).forEach { classifier ->
    val name = ":onnxruntime-lib-$classifier"
    include(name)
    project(name).projectDir = file("lib/onnxruntime-lib-$classifier")
}
