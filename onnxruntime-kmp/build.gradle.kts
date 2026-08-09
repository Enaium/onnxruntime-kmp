import java.io.File
import java.net.HttpURLConnection
import java.net.URI
import java.nio.file.Files
import org.gradle.internal.os.OperatingSystem
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget
import org.jetbrains.kotlin.gradle.tasks.KotlinNativeLink

plugins {
    alias(libs.plugins.kotlin.multiplatform)
    alias(libs.plugins.android.kmp.library)
    alias(libs.plugins.maven.publish)
}

group = rootProject.group
version = rootProject.version

// =========================================================================
// Prebuilt ONNX Runtime shared libraries
//
// Instead of building onnxruntime from source, we download the official
// CPU-only prebuilt packages from the GitHub releases and link against the
// bundled shared library (libonnxruntime). The C API headers ship inside the
// same package, so cinterop consumes them directly.
//
// Note: Microsoft stopped shipping Intel-mac binaries after 1.23.2 (only
// osx-arm64 is published since 1.24), so the macosX64 target links the last
// available x86_64 build while all other targets use the latest release.
// =========================================================================

val onnxRuntimeVersion = "1.26.0"
val onnxRuntimeMacosX64Version = "1.23.2"

data class NativeLibSpec(
    val targetName: String,
    val key: String,
    val artifact: String,
    val version: String,
    val url: String,
    val zip: Boolean,
    val symlinks: List<Pair<String, String>>,
    val runtimeFile: String,
)

val nativeRoot = rootProject.projectDir.resolve("native")
val downloadsDir = nativeRoot.resolve("downloads")

fun nativeLibs(): Map<String, NativeLibSpec> {
    val release = "https://github.com/microsoft/onnxruntime/releases/download"
    return listOf(
        NativeLibSpec(
            targetName = "macosArm64",
            key = "osx-arm64",
            artifact = "onnxruntime-osx-arm64-$onnxRuntimeVersion.tgz",
            version = onnxRuntimeVersion,
            url = "$release/v$onnxRuntimeVersion/onnxruntime-osx-arm64-$onnxRuntimeVersion.tgz",
            zip = false,
            symlinks = listOf(
                "libonnxruntime.1.dylib" to "libonnxruntime.$onnxRuntimeVersion.dylib",
                "libonnxruntime.dylib" to "libonnxruntime.$onnxRuntimeVersion.dylib",
            ),
            runtimeFile = "libonnxruntime.1.dylib",
        ),
        NativeLibSpec(
            targetName = "macosX64",
            key = "osx-x86_64",
            artifact = "onnxruntime-osx-x86_64-$onnxRuntimeMacosX64Version.tgz",
            version = onnxRuntimeMacosX64Version,
            url = "$release/v$onnxRuntimeMacosX64Version/onnxruntime-osx-x86_64-$onnxRuntimeMacosX64Version.tgz",
            zip = false,
            symlinks = listOf(
                "libonnxruntime.1.dylib" to "libonnxruntime.$onnxRuntimeMacosX64Version.dylib",
                "libonnxruntime.dylib" to "libonnxruntime.$onnxRuntimeMacosX64Version.dylib",
            ),
            runtimeFile = "libonnxruntime.$onnxRuntimeMacosX64Version.dylib",
        ),
        NativeLibSpec(
            targetName = "linuxX64",
            key = "linux-x64",
            artifact = "onnxruntime-linux-x64-$onnxRuntimeVersion.tgz",
            version = onnxRuntimeVersion,
            url = "$release/v$onnxRuntimeVersion/onnxruntime-linux-x64-$onnxRuntimeVersion.tgz",
            zip = false,
            symlinks = listOf(
                "libonnxruntime.so.1" to "libonnxruntime.so.$onnxRuntimeVersion",
                "libonnxruntime.so" to "libonnxruntime.so.1",
            ),
            runtimeFile = "libonnxruntime.so.1",
        ),
        NativeLibSpec(
            targetName = "linuxArm64",
            key = "linux-aarch64",
            artifact = "onnxruntime-linux-aarch64-$onnxRuntimeVersion.tgz",
            version = onnxRuntimeVersion,
            url = "$release/v$onnxRuntimeVersion/onnxruntime-linux-aarch64-$onnxRuntimeVersion.tgz",
            zip = false,
            symlinks = listOf(
                "libonnxruntime.so.1" to "libonnxruntime.so.$onnxRuntimeVersion",
                "libonnxruntime.so" to "libonnxruntime.so.1",
            ),
            runtimeFile = "libonnxruntime.so.1",
        ),
        NativeLibSpec(
            targetName = "mingwX64",
            key = "win-x64",
            artifact = "onnxruntime-win-x64-$onnxRuntimeVersion.zip",
            version = onnxRuntimeVersion,
            url = "$release/v$onnxRuntimeVersion/onnxruntime-win-x64-$onnxRuntimeVersion.zip",
            zip = true,
            symlinks = emptyList(),
            runtimeFile = "onnxruntime.dll",
        ),
    ).associateBy { it.targetName }
}

val nativeLibSpecs = nativeLibs()

// =========================================================================
// Download + extract tasks (one pair per native target)
// =========================================================================

fun registerNativeLibTasks(spec: NativeLibSpec) {
    val targetName = spec.targetName
    val libDir = nativeRoot.resolve(spec.key)
    val marker = File(libDir, ".onnxruntime-version")
    val archive = downloadsDir.resolve(spec.artifact)

    val downloadTask = tasks.register("downloadNativeLib_$targetName") {
        group = "onnxruntime"
        description = "Downloads the onnxruntime $targetName prebuilt package."
        onlyIf {
            val upToDate = marker.isFile && marker.readText().trim() == spec.version &&
                File(libDir, "include").isDirectory && File(libDir, "lib").isDirectory
            !upToDate
        }
        doLast {
            downloadsDir.mkdirs()
            val connection = URI(spec.url).toURL().openConnection()
            connection.setRequestProperty("User-Agent", "onnxruntime-kmp-gradle")
            connection.connect()
            if ((connection as HttpURLConnection).responseCode != 200) {
                error("Failed to download ${spec.url}: HTTP ${connection.responseCode}")
            }
            val tmp = File.createTempFile("onnxruntime", ".part", downloadsDir)
            connection.inputStream.use { input ->
                tmp.outputStream().use { output -> input.copyTo(output) }
            }
            tmp.renameTo(archive)
        }
    }

    val extractTask = tasks.register<Exec>("extractNativeLib_$targetName") {
        group = "onnxruntime"
        description = "Extracts the onnxruntime $targetName package into native/${spec.key}."
        dependsOn(downloadTask)
        onlyIf {
            !(marker.isFile && marker.readText().trim() == spec.version &&
                File(libDir, "include").isDirectory && File(libDir, "lib").isDirectory)
        }
        doFirst { libDir.mkdirs() }
        // Use the system tar (bsdtar on macOS/Windows, GNU tar on Linux):
        // it preserves the versioned symlinks Gradle's archive trees mangle.
        // The macOS packages carry a leading "./" component, the Linux and
        // Windows ones do not, so probe the first entry for the strip count.
        doFirst {
            val stripComponents = if (spec.zip) 1 else {
                val first = ProcessBuilder("tar", "-tzf", archive.absolutePath)
                    .redirectErrorStream(true)
                    .start()
                    .inputStream.bufferedReader()
                    .use { it.readLine() }
                if (first.orEmpty().startsWith("./")) 2 else 1
            }
            commandLine = when {
                spec.zip -> listOf(
                    "tar", "-xf", archive.absolutePath,
                    "--strip-components=$stripComponents", "-C", libDir.absolutePath,
                    "--exclude=*.pdb",
                )
                else -> listOf(
                    "tar", "-xzf", archive.absolutePath,
                    "--strip-components=$stripComponents", "-C", libDir.absolutePath,
                    "--exclude=*.pdb", "--exclude=*.dSYM",
                )
            }
        }
        doLast {
            marker.writeText(spec.version)
        }
    }
}

// cinterop must run after the matching prebuilt package is downloaded,
// extracted and (on Unix) its version symlinks recreated.
tasks.configureEach {
    if (name.startsWith("cinteropOnnxruntime")) {
        val targetName = name
            .removePrefix("cinteropOnnxruntime")
            .replaceFirstChar { it.lowercase() }
        if (nativeLibSpecs.containsKey(targetName)) {
            dependsOn("downloadNativeLib_$targetName", "extractNativeLib_$targetName")
        }
    }
}

nativeLibSpecs.values.forEach { registerNativeLibTasks(it) }

// =========================================================================
// cinterop def file: bindings are generated from the bundled C API header,
// and the linker options embed the shared library location + rpath into the
// produced klib, so consumers don't need to configure anything.
// =========================================================================

fun onnxruntimeDefFile(targetName: String): File {
    val spec = nativeLibSpecs.getValue(targetName)
    val libDir = nativeRoot.resolve(spec.key).resolve("lib")
    val includeDir = nativeRoot.resolve(spec.key).resolve("include")

    val linkerOpts = mutableListOf(
        "-L${libDir.absolutePath}",
        "-lonnxruntime",
    )
    when (spec.targetName) {
        "macosArm64", "macosX64" -> linkerOpts += "-rpath @loader_path"
        "linuxX64", "linuxArm64" -> linkerOpts += "-rpath ${'$'}ORIGIN"
    }

    val dir = layout.buildDirectory.dir("def").get().asFile
    dir.mkdirs()
    val file = File(dir, "onnxruntime-$targetName.def")
    file.writeText(
        buildString {
            appendLine("headers = onnxruntime_c_api.h")
            appendLine("compilerOpts = -std=c11 -I${includeDir.absolutePath}")
            appendLine("linkerOpts = ${linkerOpts.joinToString(" ")}")
        },
    )
    return file
}

kotlin {
    // ==================== JVM ====================
    jvm {
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_21)
        }
        testRuns["test"].executionTask.configure {
            useJUnitPlatform()
        }
    }

    // ==================== Android ====================
    android {
        namespace = "cn.enaium.onnxruntime"
        compileSdk = 37
        minSdk = 24

        compilations.configureEach {
            compileTaskProvider.configure {
                compilerOptions {
                    jvmTarget.set(JvmTarget.JVM_21)
                }
            }
        }
    }

    // ==================== Native (desktop only) ====================
    macosArm64()
    macosX64()

    linuxX64()
    linuxArm64()

    mingwX64()

// ==================== cinterop for all native targets ====================
targets.withType<KotlinNativeTarget> {
    val targetName = this.name
    compilations.getByName("main") {
        cinterops {
            create("onnxruntime") {
                defFile(onnxruntimeDefFile(targetName))
                packageName("onnxruntime")
                includeDirs(
                    project.file("src/nativeInterop/cinterop"),
                    nativeRoot.resolve(nativeLibSpecs.getValue(targetName).key).resolve("include"),
                )
            }
        }
        defaultSourceSet.kotlin.srcDir("src/nativeMain/kotlin")
    }
}

// Native tests run against the prebuilt shared library: copy it next to the
// test binary, where the embedded @loader_path / $ORIGIN rpath finds it.
kotlin.targets.withType<KotlinNativeTarget>().configureEach {
    val targetName = this.name
    val spec = nativeLibSpecs.getValue(targetName)
    val capitalized = targetName.replaceFirstChar { it.uppercase() }
    val linkTask = tasks.named<KotlinNativeLink>("linkDebugTest$capitalized")
    tasks.register<Copy>("copyRuntimeLib_$targetName") {
        group = "onnxruntime"
        description = "Copies the $targetName libonnxruntime next to the test binary."
        dependsOn(linkTask)
        // The source may be a versioned symlink; refresh unconditionally so
        // stale copies from previous extractions never linger.
        outputs.upToDateWhen { false }
        doFirst {
            // The source may be a versioned symlink; always refresh the
            // destination so stale copies from previous extractions don't
            // linger (Gradle's up-to-date check compares source content, not
            // link structure).
            File(linkTask.get().outputFile.get().parentFile, spec.runtimeFile).delete()
        }
        from(nativeRoot.resolve(spec.key).resolve("lib").resolve(spec.runtimeFile)) {
            rename { spec.runtimeFile }
        }
        into(linkTask.get().outputFile.get().parentFile)
    }
    tasks.configureEach {
        if (name == "${targetName}Test") {
            dependsOn("copyRuntimeLib_$targetName")
        }
    }
}

    // ==================== Source sets ====================
    sourceSets {
        getByName("commonMain") {
            dependencies {
                implementation(kotlin("stdlib-common"))
            }
        }

        getByName("commonTest") {
            dependencies {
                implementation(libs.kotlin.test)
            }
        }

        getByName("jvmMain") {
            dependencies {
                // Official ONNX Runtime Java package: bundles the per-platform
                // native library and loads it automatically.
                api(libs.onnxruntime)
            }
        }

        getByName("androidMain") {
            dependencies {
                // Official ONNX Runtime Android package: the .so files live in
                // the AAR's jniLibs, loaded via System.loadLibrary.
                api(libs.onnxruntime.android)
            }
            // Same ai.onnxruntime Java API as the desktop artifact, so the
            // implementation source is shared with the JVM target.
            kotlin.srcDir("src/jvmMain/kotlin")
        }

        getByName("jvmTest") {
            dependencies {
                implementation(libs.junit.jupiter)
                runtimeOnly(libs.junit.platform.launcher)
            }
        }
    }
}

// ==================== Publishing ====================
mavenPublishing {
    publishToMavenCentral(automaticRelease = true)
    signAllPublications()

    coordinates(
        groupId = group.toString(),
        artifactId = "onnxruntime-kmp",
        version = null,
    )

    pom {
        name.set("onnxruntime-kmp")
        description.set(
            "Kotlin Multiplatform bindings for ONNX Runtime (CPU). JVM and Android use the official " +
                "com.microsoft.onnxruntime packages; Kotlin/Native targets (macOS x64/arm64, Linux x64/arm64, " +
                "Windows x64) link the official prebuilt libonnxruntime shared library.",
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
