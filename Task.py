import streamlit as st
import tempfile
import requests
import ffmpeg
import os
from faster_whisper import WhisperModel
import time
from speechbrain.inference import EncoderClassifier
from transformers import pipeline
from pytubefix import YouTube
import torch
from pydub import AudioSegment
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from yake import KeywordExtractor
import noisereduce as nr
import librosa
import soundfile as sf
import random


model = WhisperModel("base", device="cpu", compute_type="int8")

# Load the accent classifier and summarizer
classifier = EncoderClassifier.from_hparams(
    source="pretrained_models/accent-id-commonaccent_ecapa",
    savedir="pretrained_models/accent-id-commonaccent_ecapa"
)
summarizer = pipeline("summarization", model="t5-small")


# Map for displaying accent labels
accent_map = {
    "us": "American",
    "england": "British",
    "australia": "Australian",
    "canada": "Canadian",
    "india": "Indian",
    "ireland": "Irish",
    "scotland": "Scottish",
    "wales": "Welsh",
    "bermuda": "Bermudian",
    "hongkong": "Hong Kong English",
    "malaysia": "Malaysian",
    "newzealand": "New Zealander",
    "philippines": "Filipino",
    "singapore": "Singaporean",
    "southatlantic": "South Atlantic",
}

def get_accent_label(code):
    return accent_map.get(code.lower().strip(), code.capitalize())

def format_confidence(score_tensor):
    if isinstance(score_tensor, torch.Tensor):
        score = score_tensor.item()
    else:
        score = float(score_tensor)
    return f"{score:.2f}%"

def extract_audio(video_path: str, output_path: str, format: str = "wav") -> bool:
    codec_map = {"mp3": "libmp3lame", "wav": "pcm_s16le"}
    if format not in codec_map:
        st.error(f"❌ Unsupported format: {format}")
        return False
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_path, format=format, acodec=codec_map[format])
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        st.success("✅ Audio extracted successfully")
        return True
    except ffmpeg.Error as e:
        st.error("❌ FFmpeg extraction error:")
        st.error(e.stderr.decode())
        return False

def denoise_audio(input_path: str, output_path: str):
    y, sr = librosa.load(input_path, sr=None)
    reduced = nr.reduce_noise(y=y, sr=sr)
    sf.write(output_path, reduced, sr)
    return output_path

def transcribe(audio_path: str):
    segments, _ = model.transcribe(audio_path, beam_size=1, language="en")
    return " ".join([seg.text for seg in segments])

def classify_accent(audio_path: str):
    out_prob, score, index, label = classifier.classify_file(audio_path)
    return out_prob, score, index, label


def extract_random_segment(audio_path: str, segment_duration=20):
    audio = AudioSegment.from_file(audio_path)
    total_duration_sec = len(audio) / 1000

    if total_duration_sec <= segment_duration:
        # If audio shorter than segment duration, return original path
        return audio_path

    # Random start point (en ms), pour segment_duration secondes
    max_start = int((total_duration_sec - segment_duration) * 1000)
    start_ms = random.randint(0, max_start)
    end_ms = start_ms + segment_duration * 1000

    segment = audio[start_ms:end_ms]

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        segment.export(temp_audio.name, format="wav")
        return temp_audio.name


def summarize(text: str) -> str:
    return summarizer(text, max_length=15, min_length=5, do_sample=False)[0]["summary_text"]

def get_duration(audio_path):
    audio = AudioSegment.from_file(audio_path)
    return len(audio) / 1000.0

def transcription_stats(text, duration):
    words = word_tokenize(text.lower())
    filtered = [w for w in words if w.isalpha()]
    stop_words = set(stopwords.words('english'))
    content_words = [w for w in filtered if w not in stop_words]

    count = len(filtered)
    wpm = count / (duration / 60) if duration > 0 else 0
    common = Counter(content_words).most_common(3)
    keywords = [kw for kw, _ in KeywordExtractor(top=3, stopwords=stop_words).extract_keywords(text)]

    return count, wpm, common, keywords

def speech_rate_description(wpm):
    if wpm < 110:
        return "🧘 Very slow"
    elif wpm < 140:
        return "🗣️ Slow"
    elif wpm < 180:
        return "🎤 Moderate"
    elif wpm < 220:
        return "🎧 Fast"
    else:
        return "⚡ Very fast"

# App UI
st.title("🎙️ Video Transcription & Analysis")

video_source = st.radio("Choose video source:", ["📤 Upload video", "🔗 Video URL (.mp4)"])
denoise_toggle = st.toggle("🔉 Apply noise reduction")

video_path = None
if video_source == "📤 Upload video":
    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            video_path = tmp.name
elif video_source == "🔗 Video URL (.mp4)":
    url = st.text_input("Enter video URL (.mp4)")
    if url:
        try:
            if "youtu.be" in url:
                yt = YouTube(url)
                stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    stream.download(output_path=os.path.dirname(tmp.name), filename=os.path.basename(tmp.name))
                    video_path = tmp.name
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                    response = requests.get(url, stream=True)
                    if response.status_code == 200:
                        for chunk in response.iter_content(chunk_size=8192):
                            temp_video.write(chunk)
                        video_path = temp_video.name
                        st.success("✅ Vidéo téléchargée.")
                    else:
                        st.error("❌ Échec du téléchargement.")
            st.success("✅ Video downloaded")
        except Exception as e:
            st.error(f"Download error: {e}")

if video_path:
    with st.spinner("🔄 Extracting audio..."):
        audio_path = video_path.replace(".mp4", ".wav")
        success = extract_audio(video_path, audio_path)

    if denoise_toggle:
        with st.spinner("🔄 Reducing noise..."):
            denoise_audio(audio_path, audio_path)
            st.success("✅ Noise reduction complete")

    if success:
        duration = get_duration(audio_path)

        st.audio(audio_path, format="audio/wav")

        with st.spinner("📝 Transcribing..."):
            transcription = transcribe(audio_path)

        st.subheader("📝 Transcription")
        t_holder = st.empty()
        display_text = ""
        for word in transcription.split():
            display_text += word + " "
            t_holder.markdown(f"{display_text}")
            time.sleep(0.05)

        with st.spinner("🌍 Detecting accent..."):
            if duration > 120:
                audio_path = extract_random_segment(audio_path, 20)

            _, score, _, label = classify_accent(audio_path)
        st.write(f"🌍 **Detected Accent**: `{get_accent_label(label[0])}`")
        st.write(f"🔢 Confidence Score: `{format_confidence(score)}`")

        with st.spinner("📚 Summarizing..."):
            summary = summarize(transcription)
        s_holder = st.empty()
        s_text = ""
        for word in summary.split():
            s_text += word + " "
            s_holder.markdown(f"**{s_text}**")
            time.sleep(0.05)

        wc, wpm, common_words, keywords = transcription_stats(transcription, duration)

        st.subheader("📊 Transcription Stats")
        st.write(f"📝 Word Count: **{wc}**")
        st.write(f"🕒 Duration: **{duration:.1f} sec**")
        st.write(f"🚀 Speaking Speed: **{wpm:.1f} wpm** ({speech_rate_description(wpm)})")

        st.write("🔁 Most Frequent Words:")
        for word, count in common_words:
            st.write(f"• **{word}**: {count} times")

        st.write("🗝️ Keywords:")
        for kw in keywords:
            st.write(f"• {kw}")