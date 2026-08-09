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

The Kotlin/Native klibs embed the link configuration (library path + `@loader_path` /
`$ORIGIN` rpath). Since a dynamic library cannot be embedded into a klib, the
shared libraries are also published as standalone per-platform artifacts:

| Artifact | Contents |
| --- | --- |
| `onnxruntime-lib-macosarm64` | `libonnxruntime.1.26.0.dylib` + `libonnxruntime.1.dylib` |
| `onnxruntime-lib-macosx64` | `libonnxruntime.1.23.2.dylib` |
| `onnxruntime-lib-linuxx64` / `onnxruntime-lib-linuxarm64` | `libonnxruntime.so.1.26.0` + `libonnxruntime.so.1` |
| `onnxruntime-lib-mingwx64` | `onnxruntime.dll` + `onnxruntime.lib` |

Consumers resolve the matching `onnxruntime-lib-<platform>` artifact to obtain the
runtime library and pass its location to the linker (e.g. via
`binaries.all { linkerOpts("-L<extracted-dir>") }`), or simply ship the file next
to the final binary where the embedded rpath finds it.

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
