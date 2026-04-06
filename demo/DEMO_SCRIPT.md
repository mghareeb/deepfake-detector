# Hackathon Demo Script — Deepfake Detector

> **Total runtime:** 60 seconds
> **Format:** Screen recording with voiceover
> **Tools:** QuickTime / OBS for capture, app running on localhost

---

## PRE-RECORDING CHECKLIST

- [ ] Backend running: `cd backend && uvicorn main:app --port 8000`
- [ ] Frontend running: `cd frontend && npm run dev` (localhost:5173)
- [ ] Browser open to localhost:5173, dark mode, no bookmarks bar
- [ ] Two test images ready on Desktop:
  - `real_photo.jpg` — unedited portrait photo
  - `fake_photo.jpg` — AI-generated or face-swapped image
- [ ] Browser zoom set to 100 %, window at 1280 x 800
- [ ] Close all notifications, hide dock

---

## SCENE 1 — Problem statement (0:00 – 0:12)

**[SCREEN: Title card — dark background, white text center-screen]**

Title card text (create in Keynote / Canva / simple HTML page):

```
DEEPFAKE DETECTOR
AI-generated images fool humans 38 % of the time.
Our model catches them 93 % of the time.
```

**VOICEOVER:**

> "Last year over 500,000 deepfake videos and images were shared
> online. The human eye detects them barely a third of the time.
> We built a tool that does it in under a second."

**[0:10 — cut to the app landing page showing the headline
"Is this image real?" and the empty drop zone]**

> "Here's how it works."

---

## SCENE 2 — Real image upload (0:12 – 0:30)

**[ACTION: Drag `real_photo.jpg` from Desktop into the drop zone]**

**VOICEOVER (while image uploads and spinner shows):**

> "We drag in a real, unedited photo."

**[PAUSE ~2 s for the score meter to animate]**

> "The score meter lands in the green — 12 % fake probability.
> The model is 94 % confident this image is authentic."

**[ACTION: Slowly scroll down to show the side-by-side comparison]**

> "On the left, the original. On the right, the Grad-CAM heatmap.
> Notice how there's almost no highlighted region — the model
> found nothing suspicious."

**[ACTION: Scroll to the bar chart]**

> "The confidence breakdown shows both the pixel-level CNN score
> and the frequency-domain FFT score are low. The ensemble
> combines them 70-30."

---

## SCENE 3 — Deepfake image upload (0:30 – 0:48)

**[ACTION: Click "Clear image", then drag `fake_photo.jpg` into the
drop zone]**

**VOICEOVER:**

> "Now let's try a deepfake."

**[PAUSE ~2 s for animation]**

> "Immediately — 87 % fake. The meter turns red."

**[ACTION: Scroll to side-by-side comparison]**

> "Look at the heatmap. The model lights up the jawline, the
> hairline, and around the eyes — exactly the regions where
> face-swap artifacts hide."

**[ACTION: Scroll to bar chart]**

> "The pixel score is high, and the frequency analysis confirms it —
> deepfake generators leave spectral fingerprints that our FFT
> module picks up."

---

## SCENE 4 — Technical stack (0:48 – 0:55)

**[SCREEN: Cut to a prepared slide / HTML page]**

Slide content:

```
HOW IT WORKS

    Image  ──>  Face Detection (OpenCV)
                      │
                      ▼
              EfficientNet-B4 ──> Pixel Score (70 %)
                      │
              NumPy 2-D FFT   ──> Freq Score  (30 %)
                      │
                      ▼
               Ensemble Score + Grad-CAM Heatmap

Stack: FastAPI · PyTorch · React · Tailwind CSS
Deploy: Docker on Hugging Face Spaces (T4 GPU)
```

**VOICEOVER:**

> "Under the hood: EfficientNet-B4 handles the pixel analysis,
> a 2-D FFT catches frequency artifacts, and we ensemble both
> scores. The Grad-CAM heatmap shows exactly where the model
> is looking. The whole stack is FastAPI, PyTorch, React, and
> Tailwind — containerised for one-click deploy on Hugging Face
> Spaces."

---

## SCENE 5 — Benchmarks + close (0:55 – 1:00)

**[SCREEN: Cut back to the app, scroll to the footer metrics]**

**VOICEOVER:**

> "Our benchmarks on FaceForensics++: AUC 0.94, F1 0.91,
> accuracy 93.2 %. Deepfake Detector — because seeing
> shouldn't mean believing."

**[Hold on footer for 2 seconds as the screen fades to black]**

---

## POST-PRODUCTION NOTES

| Item               | Detail                                    |
|--------------------|-------------------------------------------|
| Resolution         | 1920 x 1080 (or 1280 x 800 scaled up)    |
| Frame rate         | 30 fps                                    |
| Audio              | Record VO separately, mix at -3 dB        |
| Background music   | Lo-fi / ambient, -18 dB under voice       |
| Transitions        | Simple crossfade (0.3 s) between scenes   |
| Title card font    | Inter Bold, 48 pt, #FFFFFF on #0a0a14     |
| Export format      | MP4 H.264, AAC audio                      |

### Timing cheat-sheet

| Timestamp | Duration | Scene                        |
|-----------|----------|------------------------------|
| 0:00      | 12 s     | Title card + problem stat    |
| 0:12      | 18 s     | Real image demo              |
| 0:30      | 18 s     | Deepfake image demo          |
| 0:48      |  7 s     | Tech stack slide             |
| 0:55      |  5 s     | Benchmarks + closing line    |
