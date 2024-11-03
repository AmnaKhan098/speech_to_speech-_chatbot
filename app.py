import os
import streamlit as st
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from groq import Groq
from gtts import gTTS
import tempfile
from pydub import AudioSegment
from streamlit_audio_recorder import audio_recorder  # Import the audio recorder

# Set your Groq API key here
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Load Whisper model and processor for audio transcription
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")

# Function to process audio input
def process_audio(audio_file_path):
    # Convert MP3 to WAV format compatible with Whisper
    audio = AudioSegment.from_mp3(audio_file_path)
    wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(wav_file.name, format="wav")

    # Read the converted WAV file and process it with Whisper
    audio_input, sample_rate = sf.read(wav_file.name)
    inputs = whisper_processor(audio_input, sampling_rate=sample_rate, return_tensors="pt")
    predicted_ids = whisper_model.generate(**inputs)
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # Generate response using Groq's LLaMA model
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": transcription,
            }
        ],
        model="llama3-8b-8192",  # Update this with the specific model you want to use
    )

    response = chat_completion.choices[0].message.content

    # Convert the response text to speech
    tts = gTTS(response)
    tts_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tts_file.name)

    return transcription, tts_file.name

# Streamlit interface
st.title("Voice-to-Voice Chatbot")
st.write("Record audio to interact with the chatbot.")

# Record audio using the audio recorder
audio_data = audio_recorder("record",  # label for the recorder
                             button_label="Start Recording",
                             stop_button_label="Stop Recording",
                             audio_format="mp3")

if audio_data is not None:
    # Save recorded audio temporarily
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    audio_path.write(audio_data)
    audio_path.close()

    # Process the audio file
    transcription, response_audio_path = process_audio(audio_path.name)

    # Display transcription and response audio
    st.write("Transcription:", transcription)
    st.audio(response_audio_path, format="audio/mp3")

