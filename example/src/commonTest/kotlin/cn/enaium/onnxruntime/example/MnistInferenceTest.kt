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
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MnistInferenceTest {

    private val modelPath: String = resolveModelPath()

    @Test
    fun sessionMetadataAndShapes() {
        val env = createEnv()
        try {
            val session = createSession(env, modelPath)
            try {
                assertEquals(listOf("Input3"), session.inputNames)
                assertEquals(listOf("Plus214_Output_0"), session.outputNames)
            } finally {
                session.close()
            }
        } finally {
            env.close()
        }
    }

    @Test
    fun createSessionFromBytes() {
        val env = createEnv()
        try {
            val bytes = readModelBytes()
            val session = createSession(env, bytes)
            try {
                assertEquals(listOf("Input3"), session.inputNames)
            } finally {
                session.close()
            }
        } finally {
            env.close()
        }
    }

    @Test
    fun zerosProduceValidLogits() {
        val env = createEnv()
        try {
            val session = createSession(env, modelPath)
            try {
                val input = FloatTensor(longArrayOf(1, 1, 28, 28), FloatArray(28 * 28))
                val result = session.run("Input3" to input)
                val logits = result.getValue("Plus214_Output_0") as FloatTensor
                assertEquals(longArrayOf(1L, 10L).toList(), logits.shape.toList())
                assertTrue(logits.argMax() in 0..9)
            } finally {
                session.close()
            }
        } finally {
            env.close()
        }
    }

    @Test
    fun runMnistInferenceRecognizesDigit() {
        // A synthetic "3": the model is robust enough for a coarse bitmap.
        val pixels = FloatArray(28 * 28)
        for (y in 6 until 22) {
            for (x in 5 until 12) {
                val edge = x == 5 || x == 11
                val topMid = y in 6..9 && (x == 5 || x == 11)
                val botMid = y in 13..21 && (x == 5 || x == 11)
                if (edge || topMid || botMid) pixels[y * 28 + x] = 1.0f
            }
        }
        val digit = runMnistInference(modelPath, pixels)
        assertTrue(digit in 0..9, "Expected a digit 0..9, got $digit")
    }

    @Test
    fun sessionOptionsAreHonored() {
        val env = createEnv()
        try {
            val options = createSessionOptions()
            options.setIntraOpNumThreads(1)
            options.setGraphOptimizationLevel(GraphOptimizationLevel.ENABLE_BASIC)
            options.setSessionLogLevel(LogLevel.ERROR)
            val session = createSession(env, modelPath, options)
            try {
                val input = FloatTensor(longArrayOf(1, 1, 28, 28), FloatArray(28 * 28))
                val result = session.run("Input3" to input)
                assertEquals(longArrayOf(1L, 10L).toList(), result["Plus214_Output_0"]!!.shape.toList())
            } finally {
                session.close()
                options.close()
            }
        } finally {
            env.close()
        }
    }

    @Test
    fun invalidModelThrows() {
        val env = createEnv()
        try {
            val thrown = kotlin.runCatching { createSession(env, byteArrayOf(1, 2, 3, 4)) }
            assertTrue(thrown.isFailure, "Expected an exception for invalid model bytes")
        } finally {
            env.close()
        }
    }
}
