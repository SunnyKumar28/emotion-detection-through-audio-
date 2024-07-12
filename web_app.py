import streamlit as st
import numpy as np
import soundfile as sf
import librosa
import requests
from tensorflow.keras.models import load_model

# Load the emotion recognition model
model_path = 'my_model.keras'  # Ensure this file is in the same directory as this script
model = load_model(model_path)

# Function to predict emotion from audio
def predict_emotion(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    mfccs_scaled = mfccs_scaled.reshape(1, 40, 1)  # Adjust the shape as per the model input
    prediction = model.predict(mfccs_scaled)
    return np.argmax(prediction, axis=1)[0]  # Return the index of the highest probability

# Function to get text-to-speech with emotion
def text_to_speech(text, emotion):
    typecast_api_key = 'YOUR_TYPECAST_API_KEY'  # Replace with your actual Typecast API key
    typecast_url = 'https://api.typecast.ai/v1/synthesize'
    headers = {'Authorization': f'Bearer {typecast_api_key}', 'Content-Type': 'application/json'}
    data = {
        'text': text,
        'emotion': emotion,
        'voice': 'en_us_001'  # Replace with the desired voice
    }
    response = requests.post(typecast_url, headers=headers, json=data)
    if response.status_code == 200:
        audio_content = response.content
        with open('output.wav', 'wb') as audio_file:
            audio_file.write(audio_content)
        return 'output.wav'
    else:
        st.error('Error in generating speech')
        return None

# Streamlit app layout
st.title('Emotion Recognition and Text-to-Speech App')
st.write('This app can predict emotions from audio files and convert text to speech with specified emotions.')

# Emotion prediction section
st.header('Predict Emotion from Audio')
audio_file = st.file_uploader('Upload an audio file', type=['wav'])
if audio_file is not None:
    emotion = predict_emotion(audio_file)
    target_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant_surprise', 'sad']
    st.write(f'Predicted Emotion: {target_names[emotion]}')

# Text-to-speech section
st.header('Text-to-Speech with Emotion')
text = st.text_area('Enter text')
emotion = st.selectbox('Select emotion', ['happy', 'sad', 'angry', 'neutral'])
if st.button('Generate Speech'):
    output_audio_file = text_to_speech(text, emotion)
    if output_audio_file:
        audio_file = open(output_audio_file, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')

# Footer
st.markdown('---')
st.write('Developed by CodeWrapper')

# Run the app with: streamlit run app.py
