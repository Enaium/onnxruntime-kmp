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

@file:JvmName("OnnxRuntimeJvm")

package cn.enaium.onnxruntime

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession

internal fun LogLevel.toOrt(): OrtLoggingLevel = when (this) {
    LogLevel.VERBOSE -> OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE
    LogLevel.INFO -> OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO
    LogLevel.WARNING -> OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
    LogLevel.ERROR -> OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
    LogLevel.FATAL -> OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL
}

internal fun GraphOptimizationLevel.toOrt(): OrtSession.SessionOptions.OptLevel = when (this) {
    GraphOptimizationLevel.DISABLE_ALL -> OrtSession.SessionOptions.OptLevel.NO_OPT
    GraphOptimizationLevel.ENABLE_BASIC -> OrtSession.SessionOptions.OptLevel.BASIC_OPT
    GraphOptimizationLevel.ENABLE_EXTENDED -> OrtSession.SessionOptions.OptLevel.EXTENDED_OPT
    GraphOptimizationLevel.ENABLE_ALL -> OrtSession.SessionOptions.OptLevel.ALL_OPT
}

internal fun throwOrt(e: OrtException): Nothing =
    throw OnnxRuntimeException(e.code.ordinal - 1, e.message ?: "ONNX Runtime error")

internal inline fun <T> ortCall(block: () -> T): T = try {
    block()
} catch (e: OrtException) {
    throwOrt(e)
}

/**
 * JVM implementation backed by the official `ai.onnxruntime` package.
 *
 * `OrtEnvironment.getEnvironment(...)` returns the process-global
 * environment, which is owned by the runtime and cannot be closed, so
 * [close] is a no-op.
 */
internal class JvmEnv internal constructor(internal val ortEnv: OrtEnvironment) : Env {
    override val version: String get() = ortEnv.version

    override fun close() = Unit
}

actual fun createEnv(logLevel: LogLevel, logId: String): Env =
    JvmEnv(OrtEnvironment.getEnvironment(logLevel.toOrt(), logId))

internal class JvmSessionOptions internal constructor(
    internal val ortOptions: OrtSession.SessionOptions,
) : SessionOptions {

    override fun setIntraOpNumThreads(threads: Int) = ortCall { ortOptions.setIntraOpNumThreads(threads) }

    override fun setInterOpNumThreads(threads: Int) = ortCall { ortOptions.setInterOpNumThreads(threads) }

    override fun setGraphOptimizationLevel(level: GraphOptimizationLevel) =
        ortCall { ortOptions.setOptimizationLevel(level.toOrt()) }

    override fun setSessionLogLevel(level: LogLevel) = ortCall { ortOptions.setSessionLogLevel(level.toOrt()) }

    override fun addConfigEntry(key: String, value: String) = ortCall { ortOptions.addConfigEntry(key, value) }

    override fun close() = ortOptions.close()
}

actual fun createSessionOptions(): SessionOptions = JvmSessionOptions(OrtSession.SessionOptions())

internal class JvmRunOptions internal constructor(
    internal val ortRunOptions: OrtSession.RunOptions,
) : RunOptions {

    override fun setTerminate() = ortCall { ortRunOptions.setTerminate(true) }

    override fun unsetTerminate() = ortCall { ortRunOptions.setTerminate(false) }

    override fun close() = ortRunOptions.close()
}

actual fun createRunOptions(): RunOptions = JvmRunOptions(OrtSession.RunOptions())

internal class JvmSession internal constructor(
    private val env: OrtEnvironment,
    internal val ortSession: OrtSession,
) : Session {

    override val inputNames: List<String> get() = ortSession.inputNames.toList()

    override val outputNames: List<String> get() = ortSession.outputNames.toList()

    override fun run(
        inputs: Map<String, Tensor>,
        outputs: Set<String>,
        runOptions: RunOptions?,
    ): Map<String, Tensor> = ortCall {
        val javaInputs = HashMap<String, ai.onnxruntime.OnnxTensorLike>(inputs.size)
        try {
            for ((name, tensor) in inputs) {
                javaInputs[name] = tensor.toOnnxTensor(env)
            }
            val result = if (runOptions != null) {
                ortSession.run(javaInputs, outputs, (runOptions as JvmRunOptions).ortRunOptions)
            } else {
                ortSession.run(javaInputs, outputs)
            }
            try {
                val converted = LinkedHashMap<String, Tensor>(result.size())
                for (entry in result) {
                    val value = entry.value
                    if (value !is ai.onnxruntime.OnnxTensor) {
                        throw OnnxRuntimeException(
                            -1,
                            "Unsupported output value type ${value.javaClass.name} for '${entry.key}'",
                        )
                    }
                    converted[entry.key] = value.toTensor()
                }
                converted
            } finally {
                result.close()
            }
        } finally {
            javaInputs.values.forEach { it.close() }
        }
    }

    override fun close() = ortSession.close()
}

actual fun createSession(env: Env, modelPath: String, options: SessionOptions?): Session {
    val jvmEnv = env as JvmEnv
    return JvmSession(
        jvmEnv.ortEnv,
        ortCall {
            val ortOptions = (options as? JvmSessionOptions)?.ortOptions
            if (ortOptions != null) {
                jvmEnv.ortEnv.createSession(modelPath, ortOptions)
            } else {
                jvmEnv.ortEnv.createSession(modelPath)
            }
        },
    )
}

actual fun createSession(env: Env, model: ByteArray, options: SessionOptions?): Session {
    val jvmEnv = env as JvmEnv
    return JvmSession(
        jvmEnv.ortEnv,
        ortCall {
            val ortOptions = (options as? JvmSessionOptions)?.ortOptions
            if (ortOptions != null) {
                jvmEnv.ortEnv.createSession(model, ortOptions)
            } else {
                jvmEnv.ortEnv.createSession(model)
            }
        },
    )
}
