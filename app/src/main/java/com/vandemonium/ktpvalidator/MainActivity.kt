package com.vandemonium.ktpvalidator

import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    // Views (match your XML)
    private lateinit var btnCamera: Button
    private lateinit var btnGallery: Button
    private lateinit var imageView: ImageView
    private lateinit var tvResult: TextView

    private val analyzer = ImageAnalyzer()
    private var pendingCameraUri: Uri? = null

    // Pick from gallery
    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { handleImageUri(it) }
    }

    // Take photo (full-res) to a MediaStore Uri
    private val takePictureLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success: Boolean ->
        val savedUri = pendingCameraUri // copy to local to avoid smart-cast error
        pendingCameraUri = null
        if (success && savedUri != null) {
            handleImageUri(savedUri)
        } else {
            tvResult.text = "Camera cancelled or failed."
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // wire up IDs
        btnCamera = findViewById(R.id.btnCamera)
        btnGallery = findViewById(R.id.btnGallery)
        imageView = findViewById(R.id.imageView)
        tvResult = findViewById(R.id.tvResult)

        // optional warm-up to trigger ML Kit model download on first run
        warmUpOcrOnce()

        btnGallery.setOnClickListener {
            pickImageLauncher.launch("image/*")
        }

        btnCamera.setOnClickListener {
            pendingCameraUri = createImageUri()
            val uri = pendingCameraUri
            if (uri != null) takePictureLauncher.launch(uri)
            else tvResult.text = "Failed to create camera URI."
        }
    }

    private fun handleImageUri(uri: Uri) {
        // Decode bitmap and analyze off the UI thread
        lifecycleScope.launch(Dispatchers.Default) {
            val bm = loadBitmapFromUri(uri)
            if (bm == null) {
                withContext(Dispatchers.Main) {
                    tvResult.text = "Failed to load image."
                }
                return@launch
            }
            withContext(Dispatchers.Main) {
                imageView.setImageBitmap(bm)
            }
            analyzeAndShow(bm)
        }
    }

    private fun analyzeAndShow(bitmap: Bitmap) {
        lifecycleScope.launch(Dispatchers.Default) {
            val result = analyzer.analyze(bitmap) // heavy work off main thread
            withContext(Dispatchers.Main) {
                renderResult(result)
            }
        }
    }

    private fun renderResult(a: AnalysisResult) {
        val sb = StringBuilder()
        sb.appendLine("Final Score: %.1f (%s)".format(a.finalScore, a.label))
        sb.appendLine()
        sb.appendLine("Document Type: ${a.docLabel}")
        sb.appendLine("— colored_fraction: %.3f".format(a.coloredFraction))
        sb.appendLine()
        sb.appendLine("Text Legibility")
        sb.appendLine("— sharpness_vlap: %.1f".format(a.sharpnessVlap))
        sb.appendLine("— edge_density: %.4f".format(a.edgeDensity))
        sb.appendLine("— rms_contrast: %.4f".format(a.rmsContrast))
        sb.appendLine("— text_density (approx): %.4f".format(a.textDensity))
        sb.appendLine()
        sb.appendLine("Censor/Occlusion")
        sb.appendLine("— censor_area_frac: %.4f".format(a.censorAreaFrac))
        sb.appendLine("— occlusion_frac: %.4f".format(a.occlusionFrac))
        sb.appendLine()
        sb.appendLine("OCR")
        sb.appendLine("— outcome: ${a.ocrOutcome}")
        sb.appendLine("— has_keywords: ${a.ocrHasKeywords}")
        sb.appendLine("— text_chars: ${a.ocrTextChars}")
        if (a.ocrSample.isNotBlank()) {
            sb.appendLine("— sample: ${a.ocrSample.replace('\n',' ').take(200)}")
        }
        tvResult.text = sb.toString()
    }

    // ---------- helpers ----------

    private fun createImageUri(): Uri? {
        val cv = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "ktp_${System.currentTimeMillis()}.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
        }
        return contentResolver.insert(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI, cv
        )
    }

    private fun loadBitmapFromUri(uri: Uri): Bitmap? {
        return try {
            if (Build.VERSION.SDK_INT >= 28) {
                val src = ImageDecoder.createSource(contentResolver, uri)
                ImageDecoder.decodeBitmap(src) { decoder, _, _ ->
                    // IMPORTANT: force software so we can call getPixels()
                    decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                    decoder.isMutableRequired = false
                }
            } else {
                @Suppress("DEPRECATION")
                MediaStore.Images.Media.getBitmap(contentResolver, uri)
            }
        } catch (_: Throwable) {
            null
        }
    }


    private fun warmUpOcrOnce() {
        try {
            val tiny = Bitmap.createBitmap(32, 32, Bitmap.Config.ARGB_8888)
            val image = com.google.mlkit.vision.common.InputImage.fromBitmap(tiny, 0)
            val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
            recognizer.process(image)
                .addOnSuccessListener { android.util.Log.d("OCR", "Warm-up success") }
                .addOnFailureListener { e ->
                    android.util.Log.w("OCR", "Warm-up failed: ${e.message}")
                }
        } catch (_: Throwable) { }
    }
}
