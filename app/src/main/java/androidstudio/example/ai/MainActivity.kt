package androidstudio.example.ai

import TFLiteModelLoader
import android.content.pm.PackageManager
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.Manifest
import android.widget.TextView


class MainActivity : AppCompatActivity() {

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var tfliteModel: Interpreter
    private lateinit var viewFinder: PreviewView
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }
        val REQUEST_CODE_CAMERA = 1001
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), REQUEST_CODE_CAMERA)
        }


        // Inisialisasi PreviewView untuk menampilkan kamera
        viewFinder = findViewById(R.id.viewFinder)

        // Memuat model TensorFlow Lite
        tfliteModel = TFLiteModelLoader(assets, "model_unquant.tflite").loadModel()

        // Inisialisasi kamera
        initializeCamera()

        // Inisialisasi Executor untuk kamera
        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun initializeCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewFinder.surfaceProvider)
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, { imageProxy ->
                        val bitmap = imageProxy.toBitmap()
                        processImage(bitmap)
                        imageProxy.close()
                    })
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
            } catch (exc: Exception) {
                exc.printStackTrace()
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImage(bitmap: Bitmap) {
        val inputArray = prepareInput(bitmap)
        val outputArray = Array(1) { FloatArray(2) } // 4 adalah jumlah kelas

        // Jalankan inferensi model
        tfliteModel.run(inputArray, outputArray)

        // Ambil hasil prediksi
        val result = outputArray[0]
        handlePrediction(result)
    }

    private fun prepareInput(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val input = Array(1) { Array(224) { Array(224) { FloatArray(3) } } }

        for (x in 0 until 224) {
            for (y in 0 until 224) {
                val pixel = scaledBitmap.getPixel(x, y)
                input[0][x][y][0] = (pixel shr 16 and 0xFF) / 255.0f
                input[0][x][y][1] = (pixel shr 8 and 0xFF) / 255.0f
                input[0][x][y][2] = (pixel and 0xFF) / 255.0f
            }
        }
        return input
    }

    private fun handlePrediction(predictions: FloatArray) {
        val classes = listOf("jari 1", "jari 2")
//        val classes = listOf("Safety Vest + Helmet", "Safety Vest Only", "Helmet Only", "Person")
        val predictedClass = predictions.indices.maxByOrNull { predictions[it] } ?: 0
        val maxPersent = classes[predictedClass]
        val confidence = predictions[predictedClass] * 100 // Confidence dalam persen

        runOnUiThread {
            // Ambil TextView dari layout
            val textViewPrediction = findViewById<TextView>(R.id.textView1)
            // Tampilkan hasil prediksi di TextView
            textViewPrediction.text = "Prediksi: $maxPersent (${confidence.toInt()}%)"
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    // Ekstensi untuk mengubah ImageProxy menjadi Bitmap
    private fun ImageProxy.toBitmap(): Bitmap {
        val buffer = planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size, null)
    }
}

// Utilitas untuk memuat model TensorFlow Lite
class TFLiteModelLoader(private val assetManager: AssetManager, private val modelPath: String) {
    fun loadModel(): Interpreter {
        val modelFile = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(modelFile.fileDescriptor)
        val model = inputStream.channel.map(
            FileChannel.MapMode.READ_ONLY,
            modelFile.startOffset,
            modelFile.declaredLength
        )
        return Interpreter(model)
    }
}