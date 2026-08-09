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

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import java.nio.DoubleBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer

/** Converts a common [Tensor] into a Java [OnnxTensor] owned by the caller. */
internal fun Tensor.toOnnxTensor(env: ai.onnxruntime.OrtEnvironment): OnnxTensor = when (this) {
    is FloatTensor -> OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)
    is DoubleTensor -> OnnxTensor.createTensor(env, DoubleBuffer.wrap(data), shape)
    is LongTensor -> OnnxTensor.createTensor(env, LongBuffer.wrap(data), shape)
    is IntTensor -> OnnxTensor.createTensor(env, IntBuffer.wrap(data), shape)
    is StringTensor -> OnnxTensor.createTensor(env, data, shape)
}

/** Copies a Java [OnnxTensor] into a common [Tensor] backed by Kotlin arrays. */
internal fun OnnxTensor.toTensor(): Tensor {
    val info = info
    val shape = info.shape
    return when (info.type) {
        OnnxJavaType.FLOAT -> FloatTensor(shape, getFloatBuffer().toFloatArray())
        OnnxJavaType.DOUBLE -> DoubleTensor(shape, getDoubleBuffer().toDoubleArray())
        OnnxJavaType.INT64 -> LongTensor(shape, getLongBuffer().toLongArray())
        OnnxJavaType.INT32 -> IntTensor(shape, getIntBuffer().toIntArray())
        OnnxJavaType.STRING -> StringTensor(shape, flattenStrings(getValue()))
        else -> throw OnnxRuntimeException(
            -1,
            "Unsupported output element type ${info.type}",
        )
    }
}

/**
 * The Java API reshapes string tensors into nested arrays, so flatten them
 * back into the row-major element order.
 */
private fun flattenStrings(value: Any): Array<String> {
    if (value is String) return arrayOf(value)
    val result = ArrayList<String>()
    fun visit(node: Any) {
        when (node) {
            is Array<*> -> node.forEach { visit(it ?: "") }
            is String -> result.add(node)
            else -> throw OnnxRuntimeException(-1, "Unexpected string tensor element: $node")
        }
    }
    visit(value)
    return result.toTypedArray()
}

private fun FloatBuffer.toFloatArray(): FloatArray {
    val copy = FloatArray(remaining())
    get(copy)
    return copy
}

private fun DoubleBuffer.toDoubleArray(): DoubleArray {
    val copy = DoubleArray(remaining())
    get(copy)
    return copy
}

private fun LongBuffer.toLongArray(): LongArray {
    val copy = LongArray(remaining())
    get(copy)
    return copy
}

private fun IntBuffer.toIntArray(): IntArray {
    val copy = IntArray(remaining())
    get(copy)
    return copy
}
