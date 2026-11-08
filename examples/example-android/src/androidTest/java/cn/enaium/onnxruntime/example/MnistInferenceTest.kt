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

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import cn.enaium.onnxruntime.FloatTensor
import cn.enaium.onnxruntime.argMax
import cn.enaium.onnxruntime.createEnv
import cn.enaium.onnxruntime.createSession
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Instrumented test exercising the ONNX Runtime Android AAR through the
 * onnxruntime-kmp bindings on a real device/emulator.
 */
@RunWith(AndroidJUnit4::class)
class MnistInferenceTest {

    private fun loadModel(context: Context): ByteArray =
        context.assets.open("mnist-8.onnx").readBytes()

    @Test
    fun mnistClassifiesZeros() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val model = loadModel(context)
        val input = FloatTensor(longArrayOf(1, 1, 28, 28), FloatArray(28 * 28))

        val env = createEnv()
        try {
            val session = createSession(env, model)
            try {
                assertEquals(listOf("Input3"), session.inputNames)
                assertEquals(listOf("Plus214_Output_0"), session.outputNames)

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
}
