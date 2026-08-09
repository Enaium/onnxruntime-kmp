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

import kotlin.math.max
import kotlin.math.min

/**
 * Base type for all ONNX Runtime tensor values.
 *
 * Tensors store their data in plain Kotlin arrays; the data is converted to
 * the native representation when passed to [Session.run].
 *
 * @property shape The tensor dimensions. An empty array represents a scalar.
 */
sealed class Tensor(open val shape: LongArray) : AutoCloseable {

    /** The element type of this tensor. */
    abstract val elementType: ElementType

    /** Total number of elements. */
    val size: Long
        get() = shape.fold(1L) { acc, dim -> acc * dim }

    /** Whether this tensor has a rank of 0 (an empty [shape]). */
    val isScalar: Boolean
        get() = shape.isEmpty()

    /** Returns a copy of the dimensions as a fresh [LongArray]. */
    fun shapeCopy(): LongArray = shape.copyOf()

    override fun close() {
        // Tensors are backed by Kotlin arrays; nothing to release.
    }
}

internal fun requireSameShape(a: Tensor, b: Tensor, op: String) {
    check(a.shape.contentEquals(b.shape)) {
        "Cannot $op tensors with mismatched shapes ${a.shape.contentToString()} and ${b.shape.contentToString()}"
    }
}

internal fun linearIndex(shape: LongArray, indices: LongArray): Int {
    require(indices.size == shape.size) {
        "Expected ${shape.size} indices for shape ${shape.contentToString()}, got ${indices.size}"
    }
    var index = 0L
    for (i in indices.indices) {
        val dim = shape[i]
        val idx = indices[i]
        require(idx in 0 until dim) { "Index $idx out of bounds for dimension $i (size $dim)" }
        index = index * dim + idx
    }
    return index.toInt()
}

/**
 * A single-precision float tensor.
 *
 * Supports element access via `get`/`set` and element-wise arithmetic
 * operators (`+`, `-`, `*`, `/` with tensors or scalars, `unaryMinus`).
 */
class FloatTensor(
    override val shape: LongArray,
    val data: FloatArray,
) : Tensor(shape) {

    override val elementType: ElementType get() = ElementType.FLOAT

    /** Linear index access into the underlying row-major data. */
    operator fun get(index: Int): Float = data[index]

    /** Multi-dimensional index access (row-major). */
    operator fun get(vararg index: Long): Float = data[linearIndex(shape, index)]

    operator fun set(index: Int, value: Float) {
        data[index] = value
    }

    operator fun set(vararg index: Long, value: Float) {
        data[linearIndex(shape, index)] = value
    }

    operator fun iterator(): Iterator<Float> = data.iterator()

    operator fun plus(other: FloatTensor): FloatTensor {
        requireSameShape(this, other, "add")
        return FloatTensor(shape, FloatArray(data.size) { data[it] + other.data[it] })
    }

    operator fun minus(other: FloatTensor): FloatTensor {
        requireSameShape(this, other, "subtract")
        return FloatTensor(shape, FloatArray(data.size) { data[it] - other.data[it] })
    }

    operator fun times(other: FloatTensor): FloatTensor {
        requireSameShape(this, other, "multiply")
        return FloatTensor(shape, FloatArray(data.size) { data[it] * other.data[it] })
    }

    operator fun div(other: FloatTensor): FloatTensor {
        requireSameShape(this, other, "divide")
        return FloatTensor(shape, FloatArray(data.size) { data[it] / other.data[it] })
    }

    operator fun plus(scalar: Float): FloatTensor =
        FloatTensor(shape, FloatArray(data.size) { data[it] + scalar })

    operator fun minus(scalar: Float): FloatTensor =
        FloatTensor(shape, FloatArray(data.size) { data[it] - scalar })

    operator fun times(scalar: Float): FloatTensor =
        FloatTensor(shape, FloatArray(data.size) { data[it] * scalar })

    operator fun div(scalar: Float): FloatTensor =
        FloatTensor(shape, FloatArray(data.size) { data[it] / scalar })

    operator fun unaryMinus(): FloatTensor =
        FloatTensor(shape, FloatArray(data.size) { -data[it] })

    /** Returns a copy of the underlying data. */
    fun toArray(): FloatArray = data.copyOf()

    /** 2-D matrix multiplication (the second dimension of `this` must equal the first of `other`). */
    infix fun dot(other: FloatTensor): FloatTensor {
        require(shape.size == 2 && other.shape.size == 2) {
            "dot expects 2-D tensors, got ${shape.contentToString()} and ${other.shape.contentToString()}"
        }
        val m = shape[0].toInt()
        val k = shape[1].toInt()
        val n = other.shape[1].toInt()
        require(k == other.shape[0].toInt()) {
            "Cannot multiply ${shape.contentToString()} by ${other.shape.contentToString()}"
        }
        val result = FloatArray(m * n)
        for (i in 0 until m) {
            for (j in 0 until n) {
                var sum = 0f
                for (t in 0 until k) sum += data[i * k + t] * other.data[t * n + j]
                result[i * n + j] = sum
            }
        }
        return FloatTensor(longArrayOf(m.toLong(), n.toLong()), result)
    }

    /** Normalizes the data to [0, 1] in-place using min-max scaling. */
    fun normalizeInPlace() {
        if (data.isEmpty()) return
        var minValue = data[0]
        var maxValue = data[0]
        for (v in data) {
            minValue = min(minValue, v)
            maxValue = max(maxValue, v)
        }
        val range = maxValue - minValue
        if (range == 0f) {
            data.fill(0f)
        } else {
            for (i in data.indices) data[i] = (data[i] - minValue) / range
        }
    }

    override fun toString(): String =
        "FloatTensor(shape=${shape.contentToString()}, data=${data.contentToString()})"
}

/**
 * A double-precision tensor, with the same element-wise operators as
 * [FloatTensor].
 */
class DoubleTensor(
    override val shape: LongArray,
    val data: DoubleArray,
) : Tensor(shape) {

    override val elementType: ElementType get() = ElementType.DOUBLE

    operator fun get(index: Int): Double = data[index]

    operator fun get(vararg index: Long): Double = data[linearIndex(shape, index)]

    operator fun set(index: Int, value: Double) {
        data[index] = value
    }

    operator fun set(vararg index: Long, value: Double) {
        data[linearIndex(shape, index)] = value
    }

    operator fun iterator(): Iterator<Double> = data.iterator()

    operator fun plus(other: DoubleTensor): DoubleTensor {
        requireSameShape(this, other, "add")
        return DoubleTensor(shape, DoubleArray(data.size) { data[it] + other.data[it] })
    }

    operator fun minus(other: DoubleTensor): DoubleTensor {
        requireSameShape(this, other, "subtract")
        return DoubleTensor(shape, DoubleArray(data.size) { data[it] - other.data[it] })
    }

    operator fun times(other: DoubleTensor): DoubleTensor {
        requireSameShape(this, other, "multiply")
        return DoubleTensor(shape, DoubleArray(data.size) { data[it] * other.data[it] })
    }

    operator fun div(other: DoubleTensor): DoubleTensor {
        requireSameShape(this, other, "divide")
        return DoubleTensor(shape, DoubleArray(data.size) { data[it] / other.data[it] })
    }

    operator fun plus(scalar: Double): DoubleTensor =
        DoubleTensor(shape, DoubleArray(data.size) { data[it] + scalar })

    operator fun minus(scalar: Double): DoubleTensor =
        DoubleTensor(shape, DoubleArray(data.size) { data[it] - scalar })

    operator fun times(scalar: Double): DoubleTensor =
        DoubleTensor(shape, DoubleArray(data.size) { data[it] * scalar })

    operator fun div(scalar: Double): DoubleTensor =
        DoubleTensor(shape, DoubleArray(data.size) { data[it] / scalar })

    operator fun unaryMinus(): DoubleTensor =
        DoubleTensor(shape, DoubleArray(data.size) { -data[it] })

    /** Returns a copy of the underlying data. */
    fun toArray(): DoubleArray = data.copyOf()

    override fun toString(): String =
        "DoubleTensor(shape=${shape.contentToString()}, data=${data.contentToString()})"
}

/**
 * A 64-bit integer tensor (ONNX INT64), with element-wise operators.
 */
class LongTensor(
    override val shape: LongArray,
    val data: LongArray,
) : Tensor(shape) {

    override val elementType: ElementType get() = ElementType.INT64

    operator fun get(index: Int): Long = data[index]

    operator fun get(vararg index: Long): Long = data[linearIndex(shape, index)]

    operator fun set(index: Int, value: Long) {
        data[index] = value
    }

    operator fun set(vararg index: Long, value: Long) {
        data[linearIndex(shape, index)] = value
    }

    operator fun iterator(): Iterator<Long> = data.iterator()

    operator fun plus(other: LongTensor): LongTensor {
        requireSameShape(this, other, "add")
        return LongTensor(shape, LongArray(data.size) { data[it] + other.data[it] })
    }

    operator fun minus(other: LongTensor): LongTensor {
        requireSameShape(this, other, "subtract")
        return LongTensor(shape, LongArray(data.size) { data[it] - other.data[it] })
    }

    operator fun times(other: LongTensor): LongTensor {
        requireSameShape(this, other, "multiply")
        return LongTensor(shape, LongArray(data.size) { data[it] * other.data[it] })
    }

    operator fun plus(scalar: Long): LongTensor =
        LongTensor(shape, LongArray(data.size) { data[it] + scalar })

    operator fun minus(scalar: Long): LongTensor =
        LongTensor(shape, LongArray(data.size) { data[it] - scalar })

    operator fun times(scalar: Long): LongTensor =
        LongTensor(shape, LongArray(data.size) { data[it] * scalar })

    operator fun unaryMinus(): LongTensor =
        LongTensor(shape, LongArray(data.size) { -data[it] })

    /** Returns a copy of the underlying data. */
    fun toArray(): LongArray = data.copyOf()

    override fun toString(): String =
        "LongTensor(shape=${shape.contentToString()}, data=${data.contentToString()})"
}

/**
 * A 32-bit integer tensor (ONNX INT32), with element-wise operators.
 */
class IntTensor(
    override val shape: LongArray,
    val data: IntArray,
) : Tensor(shape) {

    override val elementType: ElementType get() = ElementType.INT32

    operator fun get(index: Int): Int = data[index]

    operator fun get(vararg index: Long): Int = data[linearIndex(shape, index)]

    operator fun set(index: Int, value: Int) {
        data[index] = value
    }

    operator fun set(vararg index: Long, value: Int) {
        data[linearIndex(shape, index)] = value
    }

    operator fun iterator(): Iterator<Int> = data.iterator()

    operator fun plus(other: IntTensor): IntTensor {
        requireSameShape(this, other, "add")
        return IntTensor(shape, IntArray(data.size) { data[it] + other.data[it] })
    }

    operator fun minus(other: IntTensor): IntTensor {
        requireSameShape(this, other, "subtract")
        return IntTensor(shape, IntArray(data.size) { data[it] - other.data[it] })
    }

    operator fun times(other: IntTensor): IntTensor {
        requireSameShape(this, other, "multiply")
        return IntTensor(shape, IntArray(data.size) { data[it] * other.data[it] })
    }

    operator fun plus(scalar: Int): IntTensor =
        IntTensor(shape, IntArray(data.size) { data[it] + scalar })

    operator fun minus(scalar: Int): IntTensor =
        IntTensor(shape, IntArray(data.size) { data[it] - scalar })

    operator fun times(scalar: Int): IntTensor =
        IntTensor(shape, IntArray(data.size) { data[it] * scalar })

    operator fun unaryMinus(): IntTensor =
        IntTensor(shape, IntArray(data.size) { -data[it] })

    /** Returns a copy of the underlying data. */
    fun toArray(): IntArray = data.copyOf()

    override fun toString(): String =
        "IntTensor(shape=${shape.contentToString()}, data=${data.contentToString()})"
}

/**
 * A string tensor (ONNX STRING).
 */
class StringTensor(
    override val shape: LongArray,
    val data: Array<String>,
) : Tensor(shape) {

    override val elementType: ElementType get() = ElementType.STRING

    operator fun get(index: Int): String = data[index]

    operator fun get(vararg index: Long): String = data[linearIndex(shape, index)]

    operator fun set(index: Int, value: String) {
        data[index] = value
    }

    operator fun set(vararg index: Long, value: String) {
        data[linearIndex(shape, index)] = value
    }

    operator fun iterator(): Iterator<String> = data.iterator()

    /** Returns a copy of the underlying data. */
    fun toArray(): Array<String> = data.copyOf()

    override fun toString(): String =
        "StringTensor(shape=${shape.contentToString()}, data=${data.contentToString()})"
}
