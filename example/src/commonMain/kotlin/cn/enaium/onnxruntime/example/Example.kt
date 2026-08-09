/*
 * Copyright (c) 2026 Enaium
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package cn.enaium.onnxruntime.example

import cn.enaium.onnxruntime.*

/** Reads an environment variable (JVM: System.getenv; native: getenv). */
internal expect fun envVar(name: String): String?

/** Checks whether a file exists on disk. */
internal expect fun fileExists(path: String): Boolean

/**
 * Locates the mnist-8.onnx model. Resolution order:
 *  1. the `ONNX_MODEL_PATH` environment variable
 *  2. the `onnx.model.path` system property (JVM)
 *  3. a few well-known relative paths (test/run tasks use the module dir)
 */
internal fun resolveModelPath(): String {
    envVar("ONNX_MODEL_PATH")?.takeIf { it.isNotBlank() }?.let { return it }
    val candidates = listOf(
        "model/mnist-8.onnx",
        "example/model/mnist-8.onnx",
        "build/model/mnist-8.onnx",
        "../model/mnist-8.onnx",
    )
    candidates.firstOrNull { fileExists(it) }?.let { return it }
    error(
        "mnist-8.onnx not found. Set the ONNX_MODEL_PATH environment variable or run from the " +
            "example module directory. Looked for: ${candidates.joinToString(", ")}",
    )
}

/**
 * Runs the MNIST CNN on the given 28x28 grayscale pixels and returns the
 * digit with the highest probability.
 *
 * The model is loaded from [modelPath]; the input name of mnist-8.onnx is
 * `Input3` and the output `Plus214_Output_0` (shape [1, 10]).
 */
fun runMnistInference(modelPath: String, pixels: FloatArray): Int {
    check(pixels.size == 28 * 28) { "Expected 784 pixels, got ${pixels.size}" }

    val env = createEnv()
    try {
        val session = createSession(env, modelPath)
        try {
            check(session.inputNames.contains("Input3")) {
                "Unexpected model inputs: ${session.inputNames}"
            }

            val input = pixels.toTensor(longArrayOf(1, 1, 28, 28))
            val result = session.run("Input3" to input)
            val logits = result.getValue("Plus214_Output_0") as FloatTensor
            return logits.argMax()
        } finally {
            session.close()
        }
    } finally {
        env.close()
    }
}

/** Renders a downscaled ASCII representation of a 28x28 digit for fun. */
fun renderDigit(pixels: FloatArray): String {
    val sb = StringBuilder()
    for (y in 0 until 28) {
        for (x in 0 until 28) {
            sb.append(if (pixels[y * 28 + x] > 0.5f) '#' else '.')
        }
        sb.append('\n')
    }
    return sb.toString()
}

/** Reads a file's bytes (JVM: java.io; native: POSIX). */
internal expect fun readFile(path: String): ByteArray
