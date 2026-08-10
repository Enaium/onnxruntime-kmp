# onnxruntime-kmp

Kotlin Multiplatform bindings for [ONNX Runtime](https://onnxruntime.ai/) (CPU only).

A single common Kotlin API in front of the two official ONNX Runtime distributions:

| Platform | Backend |
| --- | --- |
| JVM (desktop) | [`com.microsoft.onnxruntime:onnxruntime`](https://central.sonatype.com/artifact/com.microsoft.onnxruntime/onnxruntime) |
| Android | [`com.microsoft.onnxruntime:onnxruntime-android`](https://central.sonatype.com/artifact/com.microsoft.onnxruntime/onnxruntime-android) |
| macOS arm64 | official prebuilt `libonnxruntime` 1.26.0 (osx-arm64), linked via cinterop |
| macOS x64 | official prebuilt `libonnxruntime` 1.23.2 (osx-x86_64, the last Intel build Microsoft shipped) |
| Linux x64 / arm64 | official prebuilt `libonnxruntime` 1.26.0, linked via cinterop |
| Windows x64 | official prebuilt `onnxruntime.dll` 1.26.0, linked via cinterop |

The Kotlin/Native klibs embed the link configuration (`@loader_path` / `$ORIGIN`
rpath). Since a dynamic library cannot be embedded into a klib, the shared
libraries are published as standalone per-platform artifacts
(`onnxruntime-lib-<platform>`); see [Installation](#installation).

## Installation

Published to Maven Central since `1.0.0`.

### Multiplatform (Kotlin/Native) project

```kotlin
// settings.gradle.kts
dependencyResolutionManagement {
    repositories {
        mavenCentral()
    }
}

// module build.gradle.kts
kotlin {
    sourceSets {
        commonMain.dependencies {
            implementation("cn.enaium.onnxruntime:onnxruntime-kmp:1.0.0")
        }
    }
}
```

Add the matching native runtime artifact, and pass its extracted directory to the
Kotlin/Native linker:

```kotlin
// e.g. for linuxX64
val libDir = project.layout.buildDirectory.dir("onnxruntime-lib") // extracted location

kotlin {
    targets.withType<org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget>().configureEach {
        binaries.all {
            linkerOpts("-L${libDir.get().asFile.absolutePath}", "-lonnxruntime")
        }
    }
}
```

At run time, ship the shared library next to the final binary — the klib embeds an
`@loader_path` (macOS) / `$ORIGIN` (Linux) rpath that finds it there.

| Platform | Artifact to resolve | Runtime file to ship |
| --- | --- | --- |
| macOS arm64 | `cn.enaium.onnxruntime:onnxruntime-lib-macosarm64:1.0.0` | `libonnxruntime.1.dylib` |
| macOS x64 | `cn.enaium.onnxruntime:onnxruntime-lib-macosx64:1.0.0` | `libonnxruntime.1.23.2.dylib` |
| Linux x64 | `cn.enaium.onnxruntime:onnxruntime-lib-linuxx64:1.0.0` | `libonnxruntime.so.1` |
| Linux arm64 | `cn.enaium.onnxruntime:onnxruntime-lib-linuxarm64:1.0.0` | `libonnxruntime.so.1` |
| Windows x64 | `cn.enaium.onnxruntime:onnxruntime-lib-mingwx64:1.0.0` | `onnxruntime.dll` |

> **Note:** the klib cannot embed a dynamic library, and the build machine's library
> path baked into the published klib does not exist on consumer machines — it is only
> a linker warning, but the `-lonnxruntime` search still needs the `-L` flag above.

### JVM only

```kotlin
// module build.gradle.kts
dependencies {
    implementation("cn.enaium.onnxruntime:onnxruntime-kmp-jvm:1.0.0")
}
```

or keep `onnxruntime-kmp` in a KMP module and consume the `jvm()` target variant.
The official `com.microsoft.onnxruntime:onnxruntime` package (with its bundled
per-platform natives) is pulled in automatically.

### Android

```kotlin
// module build.gradle.kts
dependencies {
    implementation("cn.enaium.onnxruntime:onnxruntime-kmp-android:1.0.0")
}
```

The `.so` files are bundled in the official `onnxruntime-android` AAR and loaded via
`System.loadLibrary`.

## Usage

```kotlin
import cn.enaium.onnxruntime.*

fun classify(modelPath: String, pixels: FloatArray): Int {
    val env = createEnv()
    val session = createSession(env, modelPath)
    val input = pixels.toTensor(longArrayOf(1, 1, 28, 28))
    val result = session.run("Input3" to input)
    val logits = result.getValue("Plus214_Output_0") as FloatTensor
    session.close()
    env.close()
    return logits.argMax()
}
```

### API overview

- `createEnv(logLevel, logId): Env` — environment (owns sessions; on JVM it wraps
  the process-global `OrtEnvironment`).
- `createSessionOptions(): SessionOptions` — threads, graph optimization level,
  log level, session config entries.
- `createRunOptions(): RunOptions` — per-run termination control.
- `createSession(env, modelPath | modelBytes, options?): Session` — loads an
  `.onnx`/`.ort` model from a file or from memory.
- `Session.inputNames` / `outputNames`, `Session.run(inputs, outputs?, runOptions?)`
  returning `Map<String, Tensor>`.
- Tensors: `FloatTensor`, `DoubleTensor`, `LongTensor`, `IntTensor`, `StringTensor`
  with `shape`, `size`, `data`; convert with `FloatArray.toTensor(shape)` etc.
- Errors: `OnnxRuntimeException(code, message)`.

### Kotlin features

- **Operator overloading** on tensors: `+`, `-`, `*`, `/` (element-wise, tensor or
  scalar), `unaryMinus`, `get`/`set` with multi-dimensional indices, `iterator()`.
- **Member extensions**: `FloatTensor.dot(other)` (2-D matrix multiply),
  `normalizeInPlace()`, `argMax()`.
- **Extension factories**: `FloatArray.toTensor(shape)`, `LongArray.toTensor(shape)`, ...
- `Tensor.run(vararg inputs: Pair<String, Tensor>)` and `Session.run(inputName, tensor)`
  conveniences.

### Module layout

- `onnxruntime-kmp` – the multiplatform library (common API + JVM/Android/`cinterop`
  implementations). The prebuilt shared libraries are downloaded from the official
  GitHub releases by the build into `native/` (git-ignored).
- `onnxruntime-lib-<platform>` – per-platform artifacts that package the prebuilt
  shared library for publication (see the table above).
- `example` – desktop example + tests: JVM and Kotlin/Native executables running real
  MNIST inference with the bundled `model/mnist-8.onnx` (committed).
- `example-android` – Android example app + instrumented test.
- `.github/workflows/test.yml` / `publish.yml` – CI and Maven Central publishing.

### Building

```bash
./gradlew publishToMavenLocal          # publishes :onnxruntime-kmp to ~/.m2
./gradlew :example:run                 # JVM MNIST example
./gradlew :example:runMacosArm64       # native MNIST example (per host target)
./gradlew :example:macosArm64Test      # per-platform tests (jvmTest, linuxX64Test, ...)
./gradlew :example-android:assembleDebug
```

### Notes

- Only the CPU execution provider is wired in; GPU/CUDA packages are deliberately
  not included.
- The JVM backend wraps the process-global `OrtEnvironment` (closing it is a no-op);
  the native backend creates and releases a real `OrtEnv*`.
- Native consumers must provide the matching shared library: macOS targets look for
  `libonnxruntime.1.dylib` (arm64) / `libonnxruntime.1.23.2.dylib` (x64) via the
  embedded `@loader_path` rpath, Linux for `libonnxruntime.so.1` via `$ORIGIN`, and
  Windows for `onnxruntime.dll` next to the executable.

## License

MIT, see [LICENSE](LICENSE).
