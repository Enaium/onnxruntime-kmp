/*
 * Per-platform artifact that ships the official prebuilt libonnxruntime
 * shared library. The klib cannot embed dynamic libraries, so consumers
 * resolve this artifact to obtain the .so/.dylib/.dll (either by depending
 * on it directly or by extracting the file next to their binary, where the
 * klib's embedded @loader_path / $ORIGIN rpath finds it).
 *
 * The files come from the :onnxruntime-kmp extraction (native/<key>/lib).
 */

import org.gradle.internal.os.OperatingSystem

plugins {
    `java-library`
    alias(libs.plugins.maven.publish)
}

group = rootProject.group
version = rootProject.version

val onnxRuntimeVersion = "1.28.0"
val targetName = "linuxArm64"
val platformKey = "linux-aarch64"
val classifier = targetName.lowercase()
val resourceDir = "cn/enaium/onnxruntime/lib/$platformKey"

// (name in the native/<key>/lib dir) to (name inside the JAR). Both the
// versioned file and the SONAME/install-name alias are shipped so the
// artifact is usable as a drop-in runtime library.
val libFiles = listOf(
        "libonnxruntime.so.1.28.0" to "libonnxruntime.so.1.28.0",
        "libonnxruntime.so.1" to "libonnxruntime.so.1"
)

val nativeLibDir = rootProject.projectDir.resolve("native/$platformKey/lib")

tasks.jar {
    dependsOn(":onnxruntime-kmp:extractNativeLib_$targetName")
    libFiles.forEach { (source, name) ->
        from(nativeLibDir.resolve(source)) {
            into(resourceDir)
            rename { name }
        }
    }
    archiveBaseName.set("onnxruntime-lib-$classifier")
}

mavenPublishing {
    publishToMavenCentral(automaticRelease = true)
    signAllPublications()
    coordinates(
        groupId = group.toString(),
        artifactId = "onnxruntime-lib-$classifier",
        version = version.toString(),
    )
    pom {
        name.set("onnxruntime-lib-$classifier")
        description.set(
            "Prebuilt libonnxruntime shared library for onnxruntime-kmp on $platformKey " +
                "(onnxruntime $onnxRuntimeVersion, CPU only). Loaded at link/run time by the " +
                "onnxruntime-kmp klib; not intended to be depended on directly.",
        )
        url.set("https://github.com/Enaium/onnxruntime-kmp")
        inceptionYear.set("2026")
        licenses {
            license {
                name.set("MIT")
                url.set("https://opensource.org/licenses/MIT")
                distribution.set("repo")
            }
        }
        developers {
            developer {
                id.set("Enaium")
            }
        }
        scm {
            url.set("https://github.com/Enaium/onnxruntime-kmp")
            connection.set("scm:git:git@github.com:Enaium/onnxruntime-kmp.git")
            developerConnection.set("scm:git:git@github.com:Enaium/onnxruntime-kmp.git")
        }
        issueManagement {
            system.set("GitHub")
            url.set("https://github.com/Enaium/onnxruntime-kmp/issues")
        }
    }
}
