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

@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package cn.enaium.onnxruntime

import cnames.structs.*
import kotlinx.cinterop.*
import onnxruntime.*
import platform.posix.*

/** Wraps a shape array into an OrtValue backed by the caller's scoped memory. */
private fun <T : CPointed> Tensor.wrapData(
    scope: MemScope,
    dataPtr: CPointer<T>,
    elementSize: Int,
    type: ONNXTensorElementDataType,
): CPointer<OrtValue>? {
    val shapePtr = scope.allocArray<int64_tVar>(shape.size)
    for (i in shape.indices) shapePtr[i] = shape[i]
    val out = scope.alloc<CPointerVar<OrtValue>>()
    Ort.api.CreateTensorWithDataAsOrtValue!!(
        Ort.cpuMemoryInfo,
        dataPtr,
        (size * elementSize.toLong()).toULong(),
        shapePtr,
        shape.size.toULong(),
        type,
        out.ptr,
    ).check()
    return out.value
}

/**
 * Converts a common [Tensor] into an [OrtValue] whose data buffer lives in
 * [scope] (safe: `OrtRun` is synchronous, the buffer is only needed until it
 * returns). The caller (e.g. `NativeSession.run`) must keep the scope alive
 * for the whole run.
 */
internal fun Tensor.toOrtValue(scope: MemScope): CPointer<OrtValue>? = when (this@toOrtValue) {
    is FloatTensor -> {
        val dataPtr = scope.allocArray<FloatVar>(data.size)
        for (i in data.indices) dataPtr[i] = data[i]
        wrapData(scope, dataPtr, 4, ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    }

    is DoubleTensor -> {
        val dataPtr = scope.allocArray<DoubleVar>(data.size)
        for (i in data.indices) dataPtr[i] = data[i]
        wrapData(scope, dataPtr, 8, ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)
    }

    is LongTensor -> {
        val dataPtr = scope.allocArray<LongVar>(data.size)
        for (i in data.indices) dataPtr[i] = data[i]
        wrapData(scope, dataPtr, 8, ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
    }

    is IntTensor -> {
        val dataPtr = scope.allocArray<IntVar>(data.size)
        for (i in data.indices) dataPtr[i] = data[i]
        wrapData(scope, dataPtr, 4, ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
    }

    is StringTensor -> {
        val shapePtr = scope.allocArray<int64_tVar>(shape.size)
        for (i in shape.indices) shapePtr[i] = shape[i]
        val out = scope.alloc<CPointerVar<OrtValue>>()
        Ort.api.CreateTensorAsOrtValue!!(
            Ort.defaultAllocator,
            shapePtr,
            shape.size.toULong(),
            ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
            out.ptr,
        ).check()
        val value = out.value
        val strings = scope.allocArray<CPointerVar<ByteVar>>(data.size)
        for (i in data.indices) strings[i] = data[i].cstr.getPointer(scope)
        Ort.api.FillStringTensor!!(value, strings, data.size.toULong()).check()
        value
    }
}

/** Reads the shape and element type of an OrtValue tensor. */
private fun CPointer<OrtValue>.tensorInfo(): Pair<LongArray, ONNXTensorElementDataType> {
    val value = this
    return memScoped {
    val infoPtr = alloc<CPointerVar<OrtTensorTypeAndShapeInfo>>()
    Ort.api.GetTensorTypeAndShape!!(value, infoPtr.ptr).check()
    val info = infoPtr.value!!

    val elementTypePtr = alloc<ONNXTensorElementDataType.Var>()
    Ort.api.GetTensorElementType!!(info, elementTypePtr.ptr).check()

    val dimsCountPtr = alloc<ULongVar>()
    Ort.api.GetDimensionsCount!!(info, dimsCountPtr.ptr).check()
    val dimsCount = dimsCountPtr.value.toInt()

    val dimsPtr = allocArray<int64_tVar>(dimsCount.coerceAtLeast(1))
    Ort.api.GetDimensions!!(info, dimsPtr, dimsCount.toULong()).check()

    val shape = LongArray(dimsCount) { dimsPtr[it] }
    Ort.api.ReleaseTensorTypeAndShapeInfo!!(info)
    shape to elementTypePtr.value
    }
}

/** Reads the raw data pointer of a numeric OrtValue tensor. */
private fun CPointer<OrtValue>.mutableData(): COpaquePointer {
    val value = this
    return memScoped {
        val out = alloc<COpaquePointerVar>()
        Ort.api.GetTensorMutableData!!(value, out.ptr).check()
        out.value ?: throw OnnxRuntimeException(-1, "GetTensorMutableData returned null")
    }
}

/**
 * Copies an output [OrtValue] into a common [Tensor] backed by Kotlin
 * arrays. The OrtValue is owned by the caller (released after this call).
 */
internal fun CPointer<OrtValue>.toTensor(): Tensor {
    val (shape, elementType) = tensorInfo()
    return when (elementType) {
        ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT -> {
            val ptr = mutableData().reinterpret<FloatVar>()
            FloatTensor(shape, FloatArray(shapeElementCount(shape)) { ptr[it] })
        }

        ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE -> {
            val ptr = mutableData().reinterpret<DoubleVar>()
            DoubleTensor(shape, DoubleArray(shapeElementCount(shape)) { ptr[it] })
        }

        ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 -> {
            val ptr = mutableData().reinterpret<LongVar>()
            LongTensor(shape, LongArray(shapeElementCount(shape)) { ptr[it] })
        }

        ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 -> {
            val ptr = mutableData().reinterpret<IntVar>()
            IntTensor(shape, IntArray(shapeElementCount(shape)) { ptr[it] })
        }

        ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING -> {
            StringTensor(shape, Array(shapeElementCount(shape)) { i -> readStringElement(i) })
        }

        else -> throw OnnxRuntimeException(
            -1,
            "Unsupported output element type ${elementType.name}",
        )
    }
}

private fun shapeElementCount(shape: LongArray): Int =
    shape.fold(1L) { acc, dim -> acc * dim }.toInt()

private fun CPointer<OrtValue>.readStringElement(index: Int): String {
    val value = this
    return memScoped {
        val lengthPtr = alloc<ULongVar>()
        Ort.api.GetStringTensorElementLength!!(value, index.toULong(), lengthPtr.ptr).check()
        val length = lengthPtr.value.toInt()
        val buffer = allocArray<ByteVar>(length + 1)
        Ort.api.GetStringTensorElement!!(value, lengthPtr.value, index.toULong(), buffer).check()
        buffer.toKString()
    }
}
