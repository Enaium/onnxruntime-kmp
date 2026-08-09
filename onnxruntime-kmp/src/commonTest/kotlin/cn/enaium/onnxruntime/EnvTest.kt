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
import kotlin.test.assertTrue

class EnvTest {

    @Test
    fun createEnvAndQueryVersion() {
        val env = createEnv(LogLevel.WARNING, "test-log")
        try {
            assertTrue(env.version.isNotBlank(), "env.version should not be blank, got '${env.version}'")
        } finally {
            env.close()
        }
    }

    @Test
    fun createOptions() {
        val options = createSessionOptions()
        try {
            options.setIntraOpNumThreads(1)
            options.setInterOpNumThreads(1)
            options.setGraphOptimizationLevel(GraphOptimizationLevel.ENABLE_EXTENDED)
            options.setSessionLogLevel(LogLevel.ERROR)
            options.addConfigEntry("session.intra_op.allow_spinning", "0")
        } finally {
            options.close()
        }
    }

    @Test
    fun runOptionsLifecycle() {
        val runOptions = createRunOptions()
        try {
            runOptions.setTerminate()
            runOptions.unsetTerminate()
        } finally {
            runOptions.close()
        }
    }
}
