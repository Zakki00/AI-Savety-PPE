import android.content.res.AssetManager
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.channels.FileChannel

class TFLiteModelLoader(private val assetManager: AssetManager, private val modelPath: String) {
    fun loadModel(): Interpreter {
        val assetFileDescriptor = assetManager.openFd(modelPath)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        val model = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        return Interpreter(model)
    }
}
