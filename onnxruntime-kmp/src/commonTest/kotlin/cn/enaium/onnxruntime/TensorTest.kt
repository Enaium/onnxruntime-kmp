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

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class TensorTest {

    @Test
    fun elementAccess() {
        val tensor = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f).toTensor(longArrayOf(2, 3))
        assertEquals(2L, tensor.shape[0])
        assertEquals(3L, tensor.shape[1])
        assertEquals(6L, tensor.size)
        assertEquals(1f, tensor[0, 0])
        assertEquals(6f, tensor[1, 2])

        // Single Int index addresses the flattened row-major data.
        assertEquals(1f, tensor[0])
        assertEquals(6f, tensor[5])

        tensor[0, 1] = 20f
        assertEquals(20f, tensor[0, 1])
        assertFailsWith<IllegalArgumentException> { tensor[2, 0] }
        // A single Long index is a multi-dimensional access: wrong rank.
        assertFailsWith<IllegalArgumentException> { tensor[0L] }
    }

    @Test
    fun elementWiseOperators() {
        val a = floatArrayOf(1f, 2f, 3f).toTensor(longArrayOf(3))
        val b = floatArrayOf(4f, 5f, 6f).toTensor(longArrayOf(3))

        assertEquals(listOf(5f, 7f, 9f), (a + b).toArray().toList())
        assertEquals(listOf(-3f, -3f, -3f), (a - b).toArray().toList())
        assertEquals(listOf(4f, 10f, 18f), (a * b).toArray().toList())
        assertEquals(listOf(0.25f, 0.4f, 0.5f), (a / b).toArray().toList())

        assertEquals(listOf(6f, 7f, 8f), (a + 5f).toArray().toList())
        assertEquals(listOf(2f, 4f, 6f), (a * 2f).toArray().toList())
        assertEquals(listOf(-1f, -2f, -3f), (-a).toArray().toList())

        assertFailsWith<IllegalStateException> { a + floatArrayOf(1f, 2f).toTensor(longArrayOf(2)) }
    }

    @Test
    fun longTensorOperators() {
        val a = longArrayOf(10L, 20L).toTensor(longArrayOf(2))
        val b = longArrayOf(1L, 2L).toTensor(longArrayOf(2))
        assertEquals(listOf(11L, 22L), (a + b).toArray().toList())
        assertEquals(listOf(9L, 18L), (a - b).toArray().toList())
        assertEquals(listOf(20L, 40L), (a * 2L).toArray().toList())
        assertEquals(listOf(-10L, -20L), (-a).toArray().toList())
    }

    @Test
    fun iteratorAndDot() {
        val a = floatArrayOf(1f, 2f, 3f, 4f).toTensor(longArrayOf(2, 2))
        assertEquals(listOf(1f, 2f, 3f, 4f), a.toArray().toList())

        val b = floatArrayOf(1f, 0f, 0f, 1f).toTensor(longArrayOf(2, 2))
        val identity = a dot b
        assertEquals(listOf(1f, 2f, 3f, 4f), identity.toArray().toList())

        val x = floatArrayOf(1f, 2f).toTensor(longArrayOf(2))
        assertFailsWith<IllegalArgumentException> { a dot x }
    }

    @Test
    fun normalizeInPlace() {
        val t = floatArrayOf(0f, 5f, 10f).toTensor(longArrayOf(3))
        t.normalizeInPlace()
        assertEquals(0f, t[0])
        assertEquals(1f, t[2])
    }

    @Test
    fun scalarTensor() {
        val scalar = FloatTensor(longArrayOf(), floatArrayOf(42f))
        assertTrue(scalar.isScalar)
        assertEquals(0, scalar.shape.size)
        assertEquals(1L, scalar.size)
    }

    @Test
    fun stringTensor() {
        val t = arrayOf("a", "b", "c").toTensor(longArrayOf(3))
        assertEquals("b", t[1])
        t[2] = "d"
        assertEquals("d", t[2])
        assertEquals(ElementType.STRING, t.elementType)
    }

    @Test
    fun shapeMismatchOnWrap() {
        assertFailsWith<IllegalArgumentException> { floatArrayOf(1f, 2f).toTensor(longArrayOf(3)) }
    }
}
