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
 * ONNX Runtime logging severity levels, mirroring [OrtLoggingLevel].
 */
enum class LogLevel(val value: Int) {
    VERBOSE(0),
    INFO(1),
    WARNING(2),
    ERROR(3),
    FATAL(4),
}

/**
 * Graph optimization levels, mirroring [GraphOptimizationLevel] (ORT_*).
 */
enum class GraphOptimizationLevel(val value: Int) {
    DISABLE_ALL(0),
    ENABLE_BASIC(1),
    ENABLE_EXTENDED(2),
    ENABLE_ALL(99),
}

/**
 * Tensor element data types supported by this binding, mirroring
 * ONNXTensorElementDataType (ONNX_TENSOR_ELEMENT_DATA_TYPE_*).
 */
enum class ElementType(val value: Int) {
    FLOAT(1),
    UINT8(2),
    INT8(3),
    UINT16(4),
    INT16(5),
    INT32(6),
    INT64(7),
    STRING(8),
    BOOL(9),
    FLOAT16(10),
    DOUBLE(11),
    UINT32(12),
    UINT64(13),
}
