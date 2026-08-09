import java.io.File
import org.gradle.internal.os.OperatingSystem
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.plugin.mpp.Executable
import org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget
import org.jetbrains.kotlin.gradle.tasks.KotlinNativeCompile
import org.jetbrains.kotlin.gradle.tasks.KotlinNativeLink
import org.jetbrains.kotlin.konan.target.HostManager

// =========================================================================
// Desktop example: JVM + Kotlin/Native executables for every host buildable
// target. Runs a real MNIST CNN inference with the bundled mnist-8.onnx
// model, and hosts the per-platform tests (commonTest).
//
// Consumes the artifact published to the local Maven repository; run
//   ./gradlew publishToMavenLocal
// from the root first (the CI workflows do exactly that).
// =========================================================================

plugins {
    alias(libs.plugins.kotlin.multiplatform)
}

val hostOs = OperatingSystem.current()
val hostArch = System.getProperty("os.arch").lowercase()

val nativeTargets = mutableListOf<String>()

kotlin {
    jvm {
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_21)
        }
    }

    if (hostOs.isMacOsX) {
        macosArm64 { binaries.executable() }
        macosX64 { binaries.executable() }
        nativeTargets += listOf("macosArm64", "macosX64")
    } else if (hostOs.isLinux) {
        linuxX64 { binaries.executable() }
        nativeTargets += "linuxX64"
        if (hostArch == "aarch64") {
            linuxArm64 { binaries.executable() }
            nativeTargets += "linuxArm64"
        }
    } else if (hostOs.isWindows) {
        mingwX64 { binaries.executable() }
        nativeTargets += "mingwX64"
    }

    // With kotlin.mpp.applyDefaultHierarchyTemplate=false (set project-wide)
    // there is no intermediate nativeMain source set; share the native entry
    // point across all native targets explicitly.
    targets.withType<KotlinNativeTarget>().configureEach {
        compilations.getByName("main").defaultSourceSet.kotlin.srcDir("src/nativeMain/kotlin")
        binaries.withType<Executable>().configureEach {
            entryPoint = "cn.enaium.onnxruntime.example.main"
        }
    }

    sourceSets {
        getByName("commonMain") {
            dependencies {
                // Consume the artifact published to the local Maven repository.
                implementation("cn.enaium.onnxruntime:onnxruntime-kmp:${rootProject.version}")
            }
        }

        getByName("commonTest") {
            dependencies {
                implementation(kotlin("test"))
            }
        }
    }
}

// =========================================================================
// ONNX Runtime shared library at runtime
//
// The klib embeds the link-time library path and an @loader_path / $ORIGIN
// rpath, so the only thing the executables need is the shared library next
// to the binary. Per target:
//   macosArm64   -> libonnxruntime.1.dylib      (install name @rpath/libonnxruntime.1.dylib)
//   macosX64     -> libonnxruntime.1.23.2.dylib (install name @rpath/libonnxruntime.1.23.2.dylib)
//   linuxX64/Arm64 -> libonnxruntime.so.1       (SONAME libonnxruntime.so.1)
//   mingwX64     -> onnxruntime.dll
// =========================================================================

data class RuntimeLib(val targetName: String, val key: String, val version: String, val fileName: String)

val onnxVersion = "1.26.0"
val onnxMacosX64Version = "1.23.2"

val runtimeLibs = listOf(
    RuntimeLib("macosArm64", "osx-arm64", onnxVersion, "libonnxruntime.1.dylib"),
    RuntimeLib("macosX64", "osx-x86_64", onnxMacosX64Version, "libonnxruntime.$onnxMacosX64Version.dylib"),
    RuntimeLib("linuxX64", "linux-x64", onnxVersion, "libonnxruntime.so.1"),
    RuntimeLib("linuxArm64", "linux-aarch64", onnxVersion, "libonnxruntime.so.1"),
    RuntimeLib("mingwX64", "win-x64", onnxVersion, "onnxruntime.dll"),
).associateBy { it.targetName }

val modelPath = rootProject.projectDir.resolve("example/model/mnist-8.onnx")
val nativeRoot = rootProject.projectDir.resolve("native")

// Copies the matching libonnxruntime shared library next to the binary so the
// embedded @loader_path / $ORIGIN rpath can find it at runtime.
fun runtimeCopyTask(linkTaskProvider: TaskProvider<KotlinNativeLink>, taskName: String, targetName: String) {
    val spec = runtimeLibs.getValue(targetName)
    val source = nativeRoot.resolve(spec.key).resolve("lib").resolve(spec.fileName)
    tasks.register<Copy>(taskName) {
        group = "example"
        description = "Copies $source next to the $targetName binary."
        dependsOn(linkTaskProvider)
        // The source may be a versioned symlink; refresh unconditionally so
        // stale copies from previous extractions never linger.
        outputs.upToDateWhen { false }
        doFirst {
            File(linkTaskProvider.get().outputFile.get().parentFile, spec.fileName).delete()
        }
        from(source) {
            // Symlink sources are followed, producing a regular file.
            rename { spec.fileName }
        }
        into(linkTaskProvider.get().outputFile.get().parentFile)
    }
}

// Native tests: the model is resolved by the common test code through the
// ONNX_MODEL_PATH env var, the onnx.model.path system property, or relative
// paths (the test binaries run with the example module as working dir).
nativeTargets.forEach { targetName ->
    val capitalized = targetName.replaceFirstChar { it.uppercase() }
    val linkTask = tasks.named<KotlinNativeLink>("linkDebugTest$capitalized")
    runtimeCopyTask(linkTask, "copyRuntimeLib_$targetName", targetName)
    tasks.named("${targetName}Test") {
        dependsOn("copyRuntimeLib_$targetName")
    }
}

// JVM test: pass the model path as a system property.
tasks.withType<Test>().configureEach {
    if (name.contains("Test", ignoreCase = true)) {
        systemProperty("onnx.model.path", modelPath.absolutePath)
    }
}

// =========================================================================
// Run tasks
// =========================================================================

// JVM: run the desktop example.
val jvmMainCompilation = kotlin.targets.getByName("jvm").compilations.getByName("main")

tasks.register<JavaExec>("run") {
    group = "application"
    description = "Runs the JVM example."
    mainClass.set("cn.enaium.onnxruntime.example.MainKt")
    classpath = (jvmMainCompilation.runtimeDependencyFiles ?: files()) +
        jvmMainCompilation.output.allOutputs
    environment("ONNX_MODEL_PATH", modelPath.absolutePath)
}

// Native: link the debug executable, copy the runtime library next to it and
// run it. mingwX64 executables must be copied to a Windows machine, so no run
// task is registered for them.
nativeTargets.forEach { targetName ->
    if (targetName.startsWith("mingw")) return@forEach
    val capitalized = targetName.replaceFirstChar { it.uppercase() }
    val linkTask = tasks.named<KotlinNativeLink>("linkDebugExecutable$capitalized")
    runtimeCopyTask(linkTask, "copyRuntimeLibExec_$targetName", targetName)
    tasks.register<Exec>("run$capitalized") {
        group = "application"
        description = "Runs the $targetName example."
        dependsOn("copyRuntimeLibExec_$targetName")
        environment("ONNX_MODEL_PATH", modelPath.absolutePath)
        val libDir = linkTask.get().outputFile.get().parentFile
        when (runtimeLibs.getValue(targetName).key) {
            "osx-arm64", "osx-x86_64" -> environment("DYLD_LIBRARY_PATH", libDir.absolutePath)
            "linux-x64", "linux-aarch64" -> environment("LD_LIBRARY_PATH", libDir.absolutePath)
        }
        doFirst {
            commandLine(linkTask.get().outputFile.get().absolutePath)
        }
    }
}
