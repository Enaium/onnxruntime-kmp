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

package cn.enaium.onnxruntime

/**
 * An ONNX Runtime environment: the top-level scope that owns sessions.
 *
 * Create instances with [createEnv]. On the JVM this wraps the process-global
 * `ai.onnxruntime.OrtEnvironment` (which cannot be closed); on Kotlin/Native
 * it wraps an `OrtEnv*` that is released on [close].
 */
interface Env : AutoCloseable {

    /** The version string of the underlying ONNX Runtime library. */
    val version: String

    override fun close() = Unit
}

/**
 * Creates an [Env].
 *
 * @param logLevel The logging severity threshold.
 * @param logId An identifier used in log output.
 */
expect fun createEnv(
    logLevel: LogLevel = LogLevel.WARNING,
    logId: String = "onnxruntime-kmp",
): Env

/**
 * Configuration for creating a [Session].
 *
 * Create instances with [createSessionOptions]; configure it, then pass it
 * to [createSession].
 */
interface SessionOptions : AutoCloseable {

    /** Sets the number of threads used to parallelize the execution of nodes. */
    fun setIntraOpNumThreads(threads: Int)

    /** Sets the number of threads used to parallelize the execution of graphs. */
    fun setInterOpNumThreads(threads: Int)

    /** Sets the graph optimization level. */
    fun setGraphOptimizationLevel(level: GraphOptimizationLevel)

    /** Sets the session logging severity threshold. */
    fun setSessionLogLevel(level: LogLevel)

    /** Sets an arbitrary session configuration entry (e.g. "session.intra_op.allow_spinning"). */
    fun addConfigEntry(key: String, value: String)

    override fun close() = Unit
}

/** Creates a [SessionOptions]. */
expect fun createSessionOptions(): SessionOptions

/**
 * Per-inference-run options.
 *
 * Create instances with [createRunOptions] and pass to
 * [Session.run].
 */
interface RunOptions : AutoCloseable {

    /** Requests that the current run terminates as soon as possible. */
    fun setTerminate()

    /** Clears the termination request set by [setTerminate]. */
    fun unsetTerminate()

    override fun close() = Unit
}

/** Creates a [RunOptions]. */
expect fun createRunOptions(): RunOptions

/**
 * An ONNX Runtime inference session for a single model.
 *
 * Create instances with [createSession]; run inference with [run]; release
 * the native session with [close].
 */
interface Session : AutoCloseable {

    /** The names of the model inputs, in declaration order. */
    val inputNames: List<String>

    /** The names of the model outputs, in declaration order. */
    val outputNames: List<String>

    /**
     * Runs the model.
     *
     * @param inputs The input tensors keyed by input name.
     * @param outputs The output names to compute; defaults to all outputs.
     * @param runOptions Optional per-run options.
     * @return A map of output name to result tensor. The returned tensors are
     *   backed by Kotlin arrays.
     */
    fun run(
        inputs: Map<String, Tensor>,
        outputs: Set<String> = outputNames.toSet(),
        runOptions: RunOptions? = null,
    ): Map<String, Tensor>

    /** Runs the model with all outputs computed. */
    fun run(vararg inputs: Pair<String, Tensor>): Map<String, Tensor> =
        run(inputs.toMap())

    /** Runs the model with a single input. */
    fun run(inputName: String, input: Tensor): Map<String, Tensor> =
        run(mapOf(inputName to input))

    override fun close() = Unit
}

/**
 * Creates a [Session] from a model file on disk.
 *
 * @param env The environment owning the session.
 * @param modelPath Path to an `.onnx` (or `.ort`) model file.
 * @param options Optional session options.
 */
expect fun createSession(env: Env, modelPath: String, options: SessionOptions? = null): Session

/**
 * Creates a [Session] from model bytes in memory.
 *
 * @param env The environment owning the session.
 * @param model The serialized model.
 * @param options Optional session options.
 */
expect fun createSession(env: Env, model: ByteArray, options: SessionOptions? = null): Session
