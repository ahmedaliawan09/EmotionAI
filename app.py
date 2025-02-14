import streamlit as st  
import tempfile  
import numpy as np  
import soundfile as sf  
from streamlit_mic_recorder import mic_recorder  
import whisper  
import google.generativeai as genai   
from gtts import gTTS  
import os  
import re  
import torch
import warnings
from deep_translator import GoogleTranslator 
import dotenv
import shutil

# ✅ Configure Gemini API  
dotenv.load_dotenv()
genai.configure(api_key="GEMINI_API_KEY")  
model = genai.GenerativeModel("gemini-2.0-flash")  

warnings.filterwarnings("ignore", category=UserWarning, module="torch")  
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")  
warnings.filterwarnings("ignore", category=RuntimeWarning, module="torch")

# ✅ Setup FFmpeg Properly  
# Manually set FFmpeg path
ffmpeg_path = shutil.which("ffmpeg")

if ffmpeg_path:
    os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)
    st.success("✅ FFmpeg is properly configured!")
else:
    st.error("🚨 FFmpeg not found! Try adding 'ffmpeg' to your requirements file.")


# ✅ Load Whisper Model  
device = "cuda" if torch.cuda.is_available() else "cpu"
model_whisper = whisper.load_model("tiny", device=device)

st.title("🧘 AI-Powered Mental Health Journal")  
st.write("Record your thoughts and get AI-generated insights!")  

# ✅ Choose Input Method  
option = st.radio("Choose Input Method:", ["Upload Audio", "Record Voice"])  

# ✅ Language Selection  
language_map = {"English": "en", "Urdu": "ur", "Spanish": "es", "French": "fr"}  
language_choice = st.selectbox("Choose Language for Audio Feedback:", list(language_map.keys()))  
language_code = language_map[language_choice]  

translator = GoogleTranslator(source='auto', target=language_code)  
audio_path = None  

# ✅ Handle Audio Upload or Recording  
if option == "Upload Audio":  
    uploaded_audio = st.file_uploader("🎙️ Upload your voice journal (MP3/WAV)", type=["mp3", "wav"])  
    if uploaded_audio:  
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:  
            temp_audio.write(uploaded_audio.read())  
            audio_path = temp_audio.name  

elif option == "Record Voice":  
    st.write("🎤 Press the button below to record your voice")  
    recorded_audio = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording")  
    if recorded_audio and "bytes" in recorded_audio:  
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:  
            temp_audio.write(recorded_audio["bytes"])  
            audio_path = temp_audio.name  

# ✅ Process Audio  
if audio_path:  
    st.audio(audio_path, format="audio/wav")  
    st.write("🔄 Transcribing... Please wait.")  

    try:
        result = model_whisper.transcribe(audio_path)  
        transcribed_text = result.get("text", "").strip()  

        if transcribed_text:
            st.write("📝 **Your Journal Entry:**")  
            st.write(transcribed_text)  

            # ✅ Generate AI Insights  
            st.write("🧠 **AI Insights & Suggestions:**")  
            prompt = f"""
            Analyze this journal entry in **less than 150 words**. Summarize in 3 parts:
            1. **Emotional Tone** - Identify key emotions in a few words.
            2. **Possible Reasons** - Explain briefly why the person might feel this way.
            3. **Quick Self-Care Tips** - Suggest 4-5 practical actions in bullet points.
            
            Entry: {transcribed_text}
            """  

            response = model.generate_content(prompt)  
            insights = response.text if hasattr(response, "text") else "No insights generated."

            # ✅ Ensure insights aren't cut-off mid-sentence  
            sentences = re.split(r'(?<=[.!?]) +', insights)  
            insights_clean = " ".join(sentences[:5])  

            st.write(insights_clean)

            # ✅ Translate & Convert to Speech  
            translated_text = translator.translate(insights_clean)
            cleaned_insights = re.sub(r'[*_`]', '', translated_text)  
            tts = gTTS(text=cleaned_insights, lang=language_code)  
            audio_file = "insights.mp3"  
            tts.save(audio_file)  
            st.audio(audio_file, format="audio/mp3")  
            
            # Cleanup  
            if os.path.exists(audio_file):
                os.remove(audio_file)
        else:
            st.error("⚠️ Transcription failed. Please try again with a clearer audio.")

    except Exception as e:
        st.error(f"🚨 Error in processing: {str(e)}")  
