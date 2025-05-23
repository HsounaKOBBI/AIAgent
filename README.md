# AIAgent English Accent Detection Tool
# Video Transcription & Accent Analysis App

This project is a web application that extracts audio from videos, transcribes speech using Whisper, detects the speaker’s accent with SpeechBrain, applies noise reduction, and generates a summary of the transcription. The app is built with Streamlit for an easy-to-use interface.

## Technical Details
- Audio extraction: FFmpeg converts video to WAV format.
- Transcription: Uses Faster Whisper (base model) for reliable speech-to-text.
- Accent classification: SpeechBrain’s pretrained ECAPA-TDNN model for accent detection.
- Summary: Hugging Face’s T5-small model for text summarization.
- Noise Reduction: Optional, using noisereduce to improve audio clarity.
- UI: Built with Streamlit for a lightweight, interactive web interface.
- Hosting: Deployed on Azure VM with port 8502 exposed.

## Supported Accents for Accent Detection

The system currently supports the following English accents:

| Code          | Accent / Region               |
| ------------- | ---------------------------- |
| **us**        | American (English - USA)      |
| **england**   | British (English - England)   |
| **australia** | Australian (English - Australia) |
| **canada**    | Canadian (English - Canada)   |
| **india**     | Indian (English - India)      |
| **ireland**   | Irish (English - Ireland)     |
| **scotland**  | Scottish (English - Scotland) |
| **wales**     | Welsh (English - Wales)       |
| **bermuda**   | Bermudian (English - Bermuda) |
| **hongkong**  | Hong Kong English             |
| **malaysia**  | Malaysian English             |
| **newzealand**| New Zealander (English - NZ)  |
| **philippines**| Filipino English             |
| **singapore** | Singaporean English           |
| **southatlantic** | South Atlantic English     |

> **Note:** These accents are based on the classification model trained primarily on English accents. For other languages or accents, additional models or training may be required.

## 📦 Installation & 🚀 Launch
1. Clone the repository:
```bash
    git clone https://github.com/HsounaKOBBI/AIAgent.git
    cd AIAgent
```
2. Install required Python packages:
```bash
    pip install -r requirements.txt
```
3. Launch the Streamlit app:
```bash
    streamlit run Task.py --server.port 8502 --server.enableCORS false --server.address 0.0.0.0
```
## ☁️ Hosting & Demo
The app is hosted on Microsoft Azure.
Try the live demo here: http://52.170.134.226:8502/

## How it Works
1. User uploads a video file or inputs a video URL.
2. The app extracts the audio track from the video.
3. Optionally applies noise reduction to improve transcription accuracy.
4. Uses Faster Whisper to transcribe audio to text.
5. Runs an accent classification model to detect the speaker’s accent.
6. Generates a brief summary of the transcription.
7. Shows detailed statistics about the transcription including speaking speed and keywords.
