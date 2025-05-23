# AIAgent
# Video Transcription & Accent Analysis App

This project is a web application that extracts audio from videos, transcribes speech using Whisper, detects the speaker’s accent with SpeechBrain, applies noise reduction, and generates a summary of the transcription. The app is built with Streamlit for an easy-to-use interface.

## Features

- Upload or download videos from URL (including YouTube)
- Extract audio track from video using FFmpeg
- Optional noise reduction on audio
- Transcribe speech to text with Whisper (faster_whisper)
- Detect speaker’s accent using SpeechBrain pretrained model
- Summarize transcription with Hugging Face transformers (T5-small)
- Provide transcription stats: word count, speaking speed, keywords, and frequent words
## Installation
1. Clone the repository:
```bash
    git clone https://github.com/HsounaKOBBI/AIAgent.git
    cd AIAgent
    ```
2. Install required Python packages:
```bash
    pip install -r requirements.txt
    ```
