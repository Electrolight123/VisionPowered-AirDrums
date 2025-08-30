# Air Drums — Pad Mode (MediaPipe + OpenCV)

Play a **virtual drum kit with your hands** using your webcam. Move your index finger into on‑screen pads to trigger **velocity‑sensitive** drum sounds with natural **stereo panning**. It even supports an optional **“head‑shake kick”** gesture. No gloves or special hardware needed.

---

## ✨ Features
- **5 velocity‑sensitive pads**: Snare, Hi‑Hat, Clap, Kick, Bass
- **Index‑finger speed = volume** (with a smooth curve & volume floor)
- **Equal‑power stereo panning** (left/right hand mapped to L/R)
- **Head‑shake “kick”** (optional) using FaceMesh yaw
- **Live HUD**: FPS, yaw, tip speed, last hit (L/R)
- **On‑screen pads** with hit flash, dwell‑based debouncing, cooldowns
- **Fully adjustable layout** with **save/load** to `pad_layout.json`

---

## 🗂️ Project Structure
```
.
├── air_drums.py
├── requirements.txt
├── common/
│   ├── hands.py
│   └── sound.py
├── sounds/
│   ├── snare.wav
│   ├── hihat.wav
│   ├── clap.wav
│   ├── kick.wav
│   └── bass.wav
└── pad_layout.json        # created after you press 's' (save)
```
> All WAV files are in **`sounds/`**. The hand‑tracking and audio helpers live in **`common/`**.

---

## 🚀 Quickstart
1. **Python 3.10+** recommended. Create a virtual env (optional):
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Connect a webcam** and run:
   ```bash
   python air_drums.py
   ```

---

## ⌨️ Controls (while the window is focused)
- **Esc**: Quit
- **F**: Toggle fullscreen
- **[ / ]**: Decrease / Increase **hit‑speed threshold**
- **R**: Re‑zero head **yaw** baseline
- **G**: Toggle grid overlay
- **S / L**: **Save / Load** pad layout → `pad_layout.json`
- **1–5**: Select pad (SNARE, HIHAT, CLAP, KICK, BASS)
- **← ↑ → ↓**: Nudge selected pad
- **Z / X**: Pad width − / +
- **C / V**: Pad height − / +

---

## 🎛️ How it works (high level)
- **Hand tracking** uses MediaPipe Hands to get 21 landmarks per hand. Only the **index fingertip** is used to sense hits.
- A hit triggers when the fingertip is **inside a pad** (with a small dwell time) **and** its **speed crosses a threshold**.
- **Volume** is computed from speed with a **floor and an exponential curve** for a natural feel. **Panning** follows the left/right slot of the hand with an **equal‑power pan law**.
- An optional **FaceMesh yaw** gesture triggers a **kick** when you do a quick left‑to‑right head move.

---

## 🔊 Audio tips
- The mixer starts with a Windows‑friendly driver; it will **fallback automatically** if needed.
- If you have no sound output on Linux/macOS, try setting `SDL_AUDIODRIVER` (e.g., `pulseaudio` or `alsa`) before launching:
  ```bash
  export SDL_AUDIODRIVER=pulseaudio   # or: alsa, coreaudio (macOS)
  python air_drums.py
  ```
- Multiple overlapping hits are handled with gentle **ducking** to avoid clipping; the code will **skip** notes only if the mixer is completely saturated.

---

## 🧩 Customization
- **Swap/replace samples**: Drop new WAVs into `sounds/` and update the mapping in code if you change file names.
- **Missing sample?** The app will **alias to a loaded sound** (e.g., `clap`) so you’re never silent.
- **Pad layout**: Press **`1–5`** to select a pad, use **arrow keys** to place it, **Z/X/C/V** to resize, then **`S`** to save. Use **`L`** to reload later.
- **Mirrored view** is enabled by default so your movement feels natural.

---

## ✅ Requirements
- Python 3.10+
- See `requirements.txt` for exact packages:
  - opencv-python
  - mediapipe
  - numpy
  - pygame
  - scikit-image

---

## 🛠️ Troubleshooting
- **Black window or no camera**: Ensure your webcam isn’t in use by another app. Try lowering camera resolution if needed.
- **No audio**: Set `SDL_AUDIODRIVER` as described above; confirm other apps can play audio.
- **Laggy video**: Close heavy apps; ensure you’re running on the discrete GPU if available.

---

## 📄 License
MIT — feel free to use and adapt. Please credit if you build on it.
