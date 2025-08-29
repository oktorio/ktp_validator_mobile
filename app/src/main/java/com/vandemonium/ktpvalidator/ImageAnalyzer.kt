package com.vandemonium.ktpvalidator

import android.graphics.Bitmap
import android.graphics.Color
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.google.mlkit.common.MlKitException
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

// ---------- Result container ----------
data class AnalysisResult(
    val finalScore: Double,
    val label: String,
    val docLabel: String,
    val coloredFraction: Double,
    val sharpnessVlap: Double,
    val edgeDensity: Double,
    val rmsContrast: Double,
    val textDensity: Double,
    val censorAreaFrac: Double,
    val occlusionFrac: Double,
    // OCR diagnostics
    val ocrOutcome: String,
    val ocrHasKeywords: Boolean,
    val ocrTextChars: Int,
    val ocrSample: String,
)

// ---------- Analyzer ----------
class ImageAnalyzer {

    // -------- KTP VALIDATION --------
    private val ktpBlueHueMin = 185.0
    private val ktpBlueHueMax = 265.0
    private val ktpAspectMin = 1.2
    private val ktpAspectMax = 2.4
    private val ktpMinBlueBgFrac = 0.08
    private val ktpMinTextDensity = 0.12
    private val ktpPortraitRightBias = 0.05

    // -------- FEATURES --------
    private val satThreshold = 0.18
    private val edgeThreshold = 20.0

    // -------- WEIGHTS --------
    private val wColorDoc = 0.10
    private val wSharpness = 0.12
    private val wEdge = 0.18
    private val wContrast = 0.28
    private val wTextDensity = 0.32

    // -------- PENALTIES --------
    private val photocopyPenalty = 0.60
    private val censorPenaltyMax = 0.85
    private val occlusionPenaltyMax = 0.45
    private val censorKnee = 0.06
    private val occlusionKnee = 0.30
    private val hardCensorReject = 0.10

    // -------- NORMALISATION BANDS --------
    private val bandSharpMin = 2.0
    private val bandSharpMax = 80.0
    private val bandEdgeMin  = 0.01
    private val bandEdgeMax  = 0.25
    private val bandContrMin = 0.015
    private val bandContrMax = 0.30
    private val bandTextMin  = 0.10
    private val bandTextMax  = 0.95
    private val bandColorMin = 0.02
    private val bandColorMax = 0.35

    // -------- OCR TUNING --------
    private val ocrTimeoutMs = 12000L
    private val ocrTimeoutPenaltyScale = 0.25
    private val ocrAutoRejectOnWeakVisual = true
    private val ocrWeakTextDensityThresh = 0.16
    private val ocrWeakEdgeFracThresh = 0.06

    fun analyze(src: Bitmap): AnalysisResult {
        val bmp = src.safeDownscale(1400)

        // ---------- OCR GATE ----------
        val ocr = runOcrGate(bmp)
        if (ocr.outcome == "no_keywords") {
            return AnalysisResult(
                finalScore = 0.0,
                label = "reject_non_ktp",
                docLabel = "not_ktp",
                coloredFraction = 0.0,
                sharpnessVlap = 0.0,
                edgeDensity = 0.0,
                rmsContrast = 0.0,
                textDensity = 0.0,
                censorAreaFrac = 0.0,
                occlusionFrac = 0.0,
                ocrOutcome = ocr.outcome,
                ocrHasKeywords = false,
                ocrTextChars = ocr.textLen,
                ocrSample = ocr.sample
            )
        }

        // ---------- Visual features ----------
        val (gray, satFrac) = toGrayAndSaturationFraction(bmp)
        val vlap = varianceOfLaplacian(gray)
        val (edges, edgeFrac) = sobelEdgeFraction(gray, edgeThreshold)
        val contrast = rmsContrast(gray)
        val textDens = approxTextDensity(gray, edges)

        val blueBgFrac = blueBackgroundFraction(bmp)
        val portraitRightScore = redPortraitRightScore(bmp)
        val censorFrac = censorAreaFraction(bmp, gray)
        val occlFrac = occlusionFraction(bmp)

        val imgAspect = bmp.width.toDouble() / bmp.height.toDouble()
        val likelyKTP = isLikelyKTP(blueBgFrac, textDens, imgAspect, portraitRightScore)

        val docType = when {
            satFrac > 0.35 -> "color_document"
            satFrac > 0.08 -> "grayscale_document_on_colored_bg"
            else -> "grayscale_document"
        }

        // Visual early rejects
        if (!likelyKTP) {
            return AnalysisResult(
                finalScore = 0.0,
                label = "reject_non_ktp",
                docLabel = docType,
                coloredFraction = satFrac,
                sharpnessVlap = vlap,
                edgeDensity = edgeFrac,
                rmsContrast = contrast,
                textDensity = textDens,
                censorAreaFrac = censorFrac,
                occlusionFrac = occlFrac,
                ocrOutcome = ocr.outcome,
                ocrHasKeywords = ocr.hasKeywords,
                ocrTextChars = ocr.textLen,
                ocrSample = ocr.sample
            )
        }
        if (censorFrac >= hardCensorReject) {
            return AnalysisResult(
                finalScore = 0.0,
                label = "reject_censored",
                docLabel = docType,
                coloredFraction = satFrac,
                sharpnessVlap = vlap,
                edgeDensity = edgeFrac,
                rmsContrast = contrast,
                textDensity = textDens,
                censorAreaFrac = censorFrac,
                occlusionFrac = occlFrac,
                ocrOutcome = ocr.outcome,
                ocrHasKeywords = ocr.hasKeywords,
                ocrTextChars = ocr.textLen,
                ocrSample = ocr.sample
            )
        }

        // ---------- Scoring ----------
        val nSharp = normalizeLog(vlap, bandSharpMin, bandSharpMax)
        val nEdge  = normalize(edgeFrac, bandEdgeMin, bandEdgeMax)
        val nContr = normalize(contrast, bandContrMin, bandContrMax)
        val nText  = normalize(textDens, bandTextMin, bandTextMax)
        val nColor = normalize(satFrac, bandColorMin, bandColorMax)

        var score = 100.0 * (
                wColorDoc    * nColor +
                        wSharpness   * nSharp +
                        wEdge        * nEdge +
                        wContrast    * nContr +
                        wTextDensity * nText
                )

        if (docType != "color_document") score *= photocopyPenalty

        // Penalties
        val cf = (censorFrac / censorKnee).coerceIn(0.0, 1.0)
        val censorScale = 1.0 - censorPenaltyMax * cf
        val of = (occlFrac / occlusionKnee).coerceIn(0.0, 1.0)
        val occlScale   = 1.0 - occlusionPenaltyMax * of.pow(2.0)
        score *= censorScale
        score *= occlScale

        // OCR timeout handling / extra punishment
        if (ocr.outcome == "timeout_or_error") {
            score *= ocrTimeoutPenaltyScale
            if (ocrAutoRejectOnWeakVisual &&
                (nText < normalize(ocrWeakTextDensityThresh, bandTextMin, bandTextMax) ||
                        edgeFrac < ocrWeakEdgeFracThresh)
            ) {
                return AnalysisResult(
                    finalScore = 0.0,
                    label = "reject_non_ktp_ocr_timeout",
                    docLabel = docType,
                    coloredFraction = satFrac,
                    sharpnessVlap = vlap,
                    edgeDensity = edgeFrac,
                    rmsContrast = contrast,
                    textDensity = textDens,
                    censorAreaFrac = censorFrac,
                    occlusionFrac = occlFrac,
                    ocrOutcome = ocr.outcome,
                    ocrHasKeywords = false,
                    ocrTextChars = ocr.textLen,
                    ocrSample = ocr.sample
                )
            }
        }

        val label = when {
            score >= 80 -> "good"
            score >= 60 -> "fair"
            score >= 40 -> "poor"
            else -> "reject"
        }

        return AnalysisResult(
            finalScore = score.coerceIn(0.0, 100.0),
            label = label,
            docLabel = docType,
            coloredFraction = satFrac,
            sharpnessVlap = vlap,
            edgeDensity = edgeFrac,
            rmsContrast = contrast,
            textDensity = textDens,
            censorAreaFrac = censorFrac,
            occlusionFrac = occlFrac,
            ocrOutcome = ocr.outcome,
            ocrHasKeywords = ocr.hasKeywords,
            ocrTextChars = ocr.textLen,
            ocrSample = ocr.sample
        )
    }

    // ---------- OCR ----------
    private data class OcrDiag(
        val outcome: String,   // "found_keywords" | "no_keywords" | "timeout_or_error"
        val hasKeywords: Boolean,
        val textLen: Int,
        val sample: String,
    )

    private fun runOcrGate(bmp: Bitmap): OcrDiag {
        return try {
            val image = InputImage.fromBitmap(bmp, 0)
            val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

            // Trigger processing (this will also trigger model download on first run)
            val result = Tasks.await(recognizer.process(image), ocrTimeoutMs, TimeUnit.MILLISECONDS)

            val raw = result.text ?: ""
            android.util.Log.d("OCR", "Raw OCR length: ${raw.length}, text: $raw")

            val text = normalizeOcr(raw)
            val found = detectKeywords(text)
            val has = found.isNotEmpty()
            val sample = text.take(120)

            if (has) OcrDiag("found_keywords", true, text.length, sample)
            else     OcrDiag("no_keywords", false, text.length, sample)

        } catch (e: Exception) {
            // Distinguish first-run / model-download issues from other errors
            if (e is MlKitException) {
                // Common cases: LIBRARY_LOAD_FAILED, REMOTE_MODEL_NOT_DOWNLOADED (14)
                android.util.Log.e("OCR", "MlKitException(code=${e.errorCode}): ${e.message}")
                return OcrDiag("model_downloading", false, 0, "")
            } else {
                android.util.Log.e("OCR", "OCR error: ${e.message}", e)
                return OcrDiag("timeout_or_error", false, 0, "")
            }
        }
    }

    private fun normalizeOcr(s: String): String {
        var x = s.uppercase()
        x = x.replace('0','O')
            .replace('1','I')
            .replace('5','S')
            .replace('6','G')
            .replace('8','B')
            .replace('4','A')
            .replace('|','I')
        return x
    }

    private fun detectKeywords(text: String, threshold: Double = 0.78): Map<String, Double> {
        val canon = listOf("NIK","NAMA","ALAMAT")
        val tokens = text.split(Regex("\\s+")).filter { it.isNotBlank() }
        val found = mutableMapOf<String,Double>()

        for (kw in canon) {
            if (text.contains(kw)) {
                found[kw] = 1.0
                continue
            }
            var best = 0.0
            for (w in tokens) {
                val score = similarity(kw, w)
                if (score > best) best = score
            }
            if (best >= threshold) {
                found[kw] = best
            }
        }
        return found
    }

    private fun similarity(a: String, b: String): Double {
        if (a == b) return 1.0
        val la = a.length; val lb = b.length
        if (la == 0 || lb == 0) return 0.0
        val dp = IntArray(lb+1) { it }
        for (i in 1..la) {
            var prev = dp[0]
            dp[0] = i
            for (j in 1..lb) {
                val temp = dp[j]
                val cost = if (a[i-1]==b[j-1]) 0 else 1
                dp[j] = minOf(dp[j]+1, dp[j-1]+1, prev+cost)
                prev = temp
            }
        }
        val dist = dp[lb]
        return 1.0 - (dist.toDouble()/minOf(la,lb).toDouble())
    }

    // ---------- metrics & heuristics ----------
    private fun toGrayAndSaturationFraction(bmp: Bitmap): Pair<Array<DoubleArray>, Double> {
        val w = bmp.width; val h = bmp.height
        val gray = Array(h) { DoubleArray(w) }
        val pixels = IntArray(w * h)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)
        var colored = 0; var i = 0
        for (y in 0 until h) for (x in 0 until w) {
            val c = pixels[i++]
            val r = Color.red(c) / 255.0
            val g = Color.green(c) / 255.0
            val b = Color.blue(c) / 255.0
            val maxc = max(r, max(g, b))
            val minc = min(r, min(g, b))
            val v = maxc
            val s = if (v == 0.0) 0.0 else (v - minc) / v
            if (s > satThreshold) colored++
            gray[y][x] = 0.299 * r + 0.587 * g + 0.114 * b
        }
        return Pair(gray, colored.toDouble() / (w * h).toDouble())
    }

    private fun varianceOfLaplacian(gray: Array<DoubleArray>): Double {
        val k = arrayOf(intArrayOf(0,1,0), intArrayOf(1,-4,1), intArrayOf(0,1,0))
        val h = gray.size; val w = gray[0].size
        var sum = 0.0; var sum2 = 0.0; var n = 0
        for (y in 1 until h-1) for (x in 1 until w-1) {
            var acc = 0.0
            for (j in -1..1) for (i in -1..1) acc += k[j+1][i+1] * gray[y+j][x+i]
            val v = abs(acc); sum += v; sum2 += v*v; n++
        }
        val mean = sum / max(1, n).toDouble()
        val varr = max(0.0, (sum2 / max(1, n).toDouble()) - mean*mean)
        return varr * 1000.0
    }

    private fun sobelEdgeFraction(gray: Array<DoubleArray>, thr: Double): Pair<Array<BooleanArray>, Double> {
        val gxk = arrayOf(intArrayOf(-1,0,1), intArrayOf(-2,0,2), intArrayOf(-1,0,1))
        val gyk = arrayOf(intArrayOf(-1,-2,-1), intArrayOf(0,0,0), intArrayOf(1,2,1))
        val h = gray.size; val w = gray[0].size
        val edges = Array(h){ BooleanArray(w) }
        var cnt = 0; var tot = 0
        for (y in 1 until h-1) for (x in 1 until w-1) {
            var gx = 0.0; var gy = 0.0
            for (j in -1..1) for (i in -1..1) {
                val v = gray[y+j][x+i]
                gx += gxk[j+1][i+1] * v; gy += gyk[j+1][i+1] * v
            }
            val mag = sqrt(gx*gx + gy*gy) * 255.0
            if (mag > thr) { edges[y][x] = true; cnt++ }
            tot++
        }
        return Pair(edges, cnt.toDouble()/max(1, tot).toDouble())
    }

    private fun rmsContrast(gray: Array<DoubleArray>): Double {
        val h = gray.size; val w = gray[0].size
        var sum = 0.0; var sum2 = 0.0; val n = h*w
        for (y in 0 until h) for (x in 0 until w) {
            val v = gray[y][x]; sum += v; sum2 += v*v
        }
        val mean = sum / n.toDouble()
        val varr = (sum2 / n.toDouble()) - mean*mean
        return sqrt(max(0.0, varr))
    }

    private fun approxTextDensity(gray: Array<DoubleArray>, edges: Array<BooleanArray>): Double {
        val h = gray.size; val w = gray[0].size
        var textLike = 0; var tot = 0
        for (y in 0 until h) for (x in 0 until w) {
            if (edges[y][x]) {
                val v = gray[y][x]
                if (v > 0.15 && v < 0.85) textLike++
            }
            tot++
        }
        return textLike.toDouble()/max(1, tot).toDouble()
    }

    private fun blueBackgroundFraction(bmp: Bitmap): Double {
        val w = bmp.width; val h = bmp.height
        val pixels = IntArray(w*h); bmp.getPixels(pixels, 0, w, 0, 0, w, h)
        var blue = 0
        for (c in pixels) {
            val r = Color.red(c)/255.0; val g = Color.green(c)/255.0; val b = Color.blue(c)/255.0
            val maxc = max(r, max(g, b)); val minc = min(r, min(g, b))
            val v = maxc
            val s = if (v==0.0) 0.0 else (v-minc)/v
            val hue = hueDeg(r,g,b,maxc,minc)
            if (s > 0.15 && hue in ktpBlueHueMin..ktpBlueHueMax) blue++
        }
        return blue.toDouble() / (w*h).toDouble()
    }

    private fun redPortraitRightScore(bmp: Bitmap): Double {
        val w = bmp.width; val h = bmp.height
        val x0 = (w*2)/3
        val pixels = IntArray(w*h); bmp.getPixels(pixels, 0, w, 0, 0, w, h)
        var hits = 0; var tot = 0
        for (y in 0 until h) for (x in x0 until w) {
            val c = pixels[y*w + x]
            val r = Color.red(c)/255.0; val g = Color.green(c)/255.0; val b = Color.blue(c)/255.0
            val maxc = max(r, max(g, b)); val minc = min(r, min(g, b))
            val v = maxc; val s = if (v==0.0) 0.0 else (v-minc)/v
            val hue = hueDeg(r,g,b,maxc,minc)
            val isRed = (hue <= 25.0 || hue >= 335.0)
            if (s > 0.35 && isRed) hits++
            tot++
        }
        return if (tot==0) 0.0 else hits.toDouble()/tot.toDouble()
    }

    private fun isLikelyKTP(blueBgFrac: Double, textDens: Double, imgAspect: Double, portraitRightScore: Double): Boolean {
        val aspectOk = imgAspect in ktpAspectMin..ktpAspectMax
        val bgOk = blueBgFrac >= ktpMinBlueBgFrac
        val textOk = textDens >= ktpMinTextDensity
        val portraitHelp = portraitRightScore >= ktpPortraitRightBias
        return aspectOk && textOk && (bgOk || portraitHelp)
    }

    private fun censorAreaFraction(bmp: Bitmap, gray: Array<DoubleArray>): Double {
        val w = bmp.width; val h = bmp.height
        val block = max(8, min(w, h) / 40)
        var censored = 0; var tot = 0
        for (by in 0 until h step block) for (bx in 0 until w step block) {
            val xEnd = min(bx+block, w); val yEnd = min(by+block, h)
            var sum = 0.0; var sum2 = 0.0; var n = 0
            for (y in by until yEnd) for (x in bx until xEnd) { val v = gray[y][x]; sum += v; sum2 += v*v; n++ }
            val mean = sum / max(1, n).toDouble()
            val varr = max(0.0, (sum2 / max(1, n).toDouble()) - mean*mean)
            val isExtreme = (mean < 0.08) || (mean > 0.96)
            val isFlat = varr < 0.00035
            if (isExtreme && isFlat) censored += (xEnd - bx) * (yEnd - by)
            tot += (xEnd - bx) * (yEnd - by)
        }
        return if (tot==0) 0.0 else censored.toDouble()/tot.toDouble()
    }

    private fun occlusionFraction(bmp: Bitmap): Double {
        val w = bmp.width; val h = bmp.height
        val pixels = IntArray(w*h); bmp.getPixels(pixels, 0, w, 0, 0, w, h)
        val rightThird = (w * 2) / 3
        var occl = 0
        for (y in 0 until h) for (x in 0 until w) {
            val c = pixels[y*w + x]
            val r = Color.red(c)/255.0; val g = Color.green(c)/255.0; val b = Color.blue(c)/255.0
            val maxc = max(r, max(g, b)); val minc = min(r, min(g, b))
            val v = maxc; val s = if (v==0.0) 0.0 else (v - minc)/v
            val hue = hueDeg(r,g,b,maxc,minc)
            val isBlue = hue in ktpBlueHueMin..ktpBlueHueMax
            val isRed  = (hue <= 25.0 || hue >= 335.0)
            if (s > 0.55 && !isBlue) {
                val inPortrait = (x >= rightThird) && isRed
                if (!inPortrait) occl++
            }
        }
        return occl.toDouble() / (w*h).toDouble()
    }

    // ---------- utils ----------
    private fun normalize(value: Double, lo: Double, hi: Double): Double =
        ((value - lo) / (hi - lo)).coerceIn(0.0, 1.0)

    private fun normalizeLog(value: Double, lo: Double, hi: Double): Double {
        val v = value.coerceIn(lo, hi)
        val a = ln(1.0 + v); val l = ln(1.0 + lo); val h = ln(1.0 + hi)
        return ((a - l) / (h - l)).coerceIn(0.0, 1.0)
    }

    private fun hueDeg(r: Double, g: Double, b: Double, maxc: Double, minc: Double): Double {
        val hdeg = when (maxc) {
            r -> 60.0 * (((g - b) / (maxc - minc + 1e-6)) % 6.0)
            g -> 60.0 * (((b - r) / (maxc - minc + 1e-6)) + 2.0)
            else -> 60.0 * (((r - g) / (maxc - minc + 1e-6)) + 4.0)
        }
        return if (hdeg < 0) hdeg + 360.0 else hdeg
    }
}

// ---------- Bitmap helper ----------
fun Bitmap.safeDownscale(maxDim: Int): Bitmap {
    val larger = max(this.width, this.height)
    if (larger <= maxDim) return this
    val scale = maxDim.toDouble() / larger.toDouble()
    val w = (this.width * scale).toInt().coerceAtLeast(1)
    val h = (this.height * scale).toInt().coerceAtLeast(1)
    return Bitmap.createScaledBitmap(this, w, h, true)
}
