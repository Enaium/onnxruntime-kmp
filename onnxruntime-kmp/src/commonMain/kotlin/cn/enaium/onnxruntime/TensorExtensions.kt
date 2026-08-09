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

/** Wraps [this] float array as a [FloatTensor] with the given dimensions. */
fun FloatArray.toTensor(shape: LongArray): FloatTensor {
    require(shape.fold(1L) { a, b -> a * b } == size.toLong()) {
        "Data size $size does not match shape ${shape.contentToString()}"
    }
    return FloatTensor(shape, this)
}


/** Wraps [this] double array as a [DoubleTensor] with the given dimensions. */
fun DoubleArray.toTensor(shape: LongArray): DoubleTensor {
    require(shape.fold(1L) { a, b -> a * b } == size.toLong()) {
        "Data size $size does not match shape ${shape.contentToString()}"
    }
    return DoubleTensor(shape, this)
}


/** Wraps [this] long array as a [LongTensor] with the given dimensions. */
fun LongArray.toTensor(shape: LongArray): LongTensor {
    require(shape.fold(1L) { a, b -> a * b } == size.toLong()) {
        "Data size $size does not match shape ${shape.contentToString()}"
    }
    return LongTensor(shape, this)
}


/** Wraps [this] int array as an [IntTensor] with the given dimensions. */
fun IntArray.toTensor(shape: LongArray): IntTensor {
    require(shape.fold(1L) { a, b -> a * b } == size.toLong()) {
        "Data size $size does not match shape ${shape.contentToString()}"
    }
    return IntTensor(shape, this)
}


/** Wraps [this] string array as a [StringTensor] with the given dimensions. */
fun Array<String>.toTensor(shape: LongArray): StringTensor {
    require(shape.fold(1L) { a, b -> a * b } == size.toLong()) {
        "Data size $size does not match shape ${shape.contentToString()}"
    }
    return StringTensor(shape, this)
}

/**
 * Returns the index of the maximum element (useful for classification
 * outputs).
 */
fun FloatTensor.argMax(): Int {
    require(data.isNotEmpty()) { "argMax requires a non-empty tensor" }
    var best = 0
    for (i in 1 until data.size) {
        if (data[i] > data[best]) best = i
    }
    return best
}
