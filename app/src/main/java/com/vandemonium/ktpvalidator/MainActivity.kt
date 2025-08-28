
package com.vandemonium.ktpvalidator

import android.Manifest
import android.app.Activity
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.vandemonium.ktpvalidator.databinding.ActivityMainBinding
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var photoUri: Uri? = null

    private val requestCameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) openCamera() else toast("Camera permission denied")
    }

    private val takePictureLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success ->
        if (success && photoUri != null) {
            val bmp = loadBitmapFromUri(photoUri!!)
            bmp?.let { analyzeAndShow(it) } ?: toast("Failed to load image")
        } else {
            toast("No photo captured")
        }
    }

    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) {
            val bmp = loadBitmapFromUri(uri)
            bmp?.let { analyzeAndShow(it) } ?: toast("Failed to load image")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnCamera.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
                openCamera()
            } else {
                requestCameraPermission.launch(Manifest.permission.CAMERA)
            }
        }

        binding.btnGallery.setOnClickListener {
            val mime = if (Build.VERSION.SDK_INT >= 33) "image/*" else "image/*"
            pickImageLauncher.launch(mime)
        }
    }

    private fun openCamera() {
        try {
            val imageFile = createImageFile()
            val uri = FileProvider.getUriForFile(
                this,
                "${applicationContext.packageName}.fileprovider",
                imageFile
            )
            photoUri = uri
            takePictureLauncher.launch(uri)
        } catch (e: IOException) {
            e.printStackTrace()
            toast("Failed to open camera")
        }
    }

    @Throws(IOException::class)
    private fun createImageFile(): File {
        val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val storageDir: File = cacheDir
        val imagesDir = File(storageDir, "images")
        if (!imagesDir.exists()) imagesDir.mkdirs()
        return File.createTempFile("JPEG_${timeStamp}_", ".jpg", imagesDir)
    }

    private fun loadBitmapFromUri(uri: Uri): Bitmap? {
        val maxDim = 2000 // avoid huge bitmaps; analyzer downsamples further

        return try {
            if (Build.VERSION.SDK_INT >= 28) {
                val src = ImageDecoder.createSource(contentResolver, uri)
                val bmp = ImageDecoder.decodeBitmap(src) { decoder, info, _ ->
                    decoder.isMutableRequired = true
                    decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE // <-- force software
                    // downsample large images to reduce memory
                    val w = info.size.width
                    val h = info.size.height
                    val larger = maxOf(w, h)
                    val sample = if (larger > maxDim) {
                        // integer power-of-two-ish sample
                        (larger / maxDim).coerceAtLeast(1)
                    } else 1
                    decoder.setTargetSampleSize(sample)
                }
                if (bmp.config != Bitmap.Config.ARGB_8888 || !bmp.isMutable) {
                    bmp.copy(Bitmap.Config.ARGB_8888, /*mutable=*/true)
                } else bmp
            } else {
                // API < 28
                val opts = android.graphics.BitmapFactory.Options().apply {
                    inPreferredConfig = Bitmap.Config.ARGB_8888
                    inMutable = true
                    inJustDecodeBounds = true
                }
                contentResolver.openInputStream(uri)?.use { ins ->
                    android.graphics.BitmapFactory.decodeStream(ins, null, opts)
                }
                val larger = maxOf(opts.outWidth, opts.outHeight)
                val inSample = if (larger > maxDim) (larger / maxDim).coerceAtLeast(1) else 1

                val opts2 = android.graphics.BitmapFactory.Options().apply {
                    inPreferredConfig = Bitmap.Config.ARGB_8888
                    inMutable = true
                    inSampleSize = inSample
                }
                contentResolver.openInputStream(uri)?.use { ins ->
                    android.graphics.BitmapFactory.decodeStream(ins, null, opts2)
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }


    private fun analyzeAndShow(bitmap: Bitmap) {
        binding.imageView.setImageBitmap(bitmap)

        val analyzer = ImageAnalyzer()
        val result = analyzer.analyze(bitmap)

        val sb = StringBuilder()
        sb.appendLine("Final Score: ${"%.1f".format(result.finalScore)} (${result.label})")
        sb.appendLine()
        sb.appendLine("Document Type: ${result.docLabel}")
        sb.appendLine("— colored_fraction: ${"%.3f".format(result.coloredFraction)}")
        sb.appendLine()
        sb.appendLine("Text Legibility")
        sb.appendLine("— sharpness_vlap: ${"%.1f".format(result.sharpnessVlap)}")
        sb.appendLine("— edge_density: ${"%.4f".format(result.edgeDensity)}")
        sb.appendLine("— rms_contrast: ${"%.4f".format(result.rmsContrast)}")
        sb.appendLine("— text_density (approx): ${"%.4f".format(result.textDensity)}")
        sb.appendLine()
        sb.appendLine("Censor/Occlusion")
        sb.appendLine("— censor_area_frac: ${"%.4f".format(result.censorAreaFrac)}")
        sb.appendLine("— occlusion_frac: ${"%.4f".format(result.occlusionFrac)}")

        binding.tvResult.text = sb.toString()
    }

    private fun toast(msg: String) {
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
    }
}
