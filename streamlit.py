import streamlit as st
import requests
import base64
from audio_recorder_streamlit import audio_recorder
import json 

# Streamlit app setup
st.set_page_config(page_title="Mental Health Chatbot", page_icon="💬", layout="centered")

# Title of the app
st.title("💬 Mental Health Chatbot")
st.write("Talk to the Mental Health Expert. Feel free to ask anything.")

# Function to get response from the Flask backend
def get_response(user_input=None, audio_bytes=None):
    url = "http://127.0.0.1:5001/chat"  # Use the port where your Flask app is running

    if user_input:
        payload = {"message": user_input}
    elif audio_bytes:
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                # Save the base64 string to a JSON file
        audio_data = {"audio_base64": audio_base64}
        with open('audio.json', 'w') as json_file:
            json.dump(audio_data, json_file)
        payload = {"audio_base64": audio_base64}  # Specify you want audio output as well
    else:
        return "No valid input provided.", None

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        data = response.json()
        with open('data.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)
        output_transcription = data.get("llm_text_response")
        audio_base64 = data.get("audio_response")
        input_transcription=data.get("transcription")
        return output_transcription, input_transcription, audio_base64
    else:
        return "Error communicating with the backend.", None

# Initialize the session state to store chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

audio_data = audio_recorder()

if audio_data:
    # Play the recorded audio
    st.audio(audio_data, format="audio/wav")
    # Send audio to the backend for transcription and response
    # input_transcription, output_transcription, audio_base64 = get_response(audio_bytes=audio_data)
    # response = get_response(audio_bytes=audio_data)
    # with open('response_final.json', 'w') as json_file:
    #     json.dump(response, json_file, indent=4)

    #     # Optionally display the response in the Streamlit app
    # st.json(response)

    output_transcription, input_transcription, audio_base64 = get_response(audio_bytes=audio_data)
    st.session_state.chat_history.append(("You (transcribed)", input_transcription))
    st.session_state.chat_history.append(("Your_wise_buddy", output_transcription))

    if audio_base64:
        audio_response_bytes = base64.b64decode(audio_base64)
        st.audio(audio_response_bytes, format='audio/mpeg')

# Input form for the user to type a message
with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("You:", "")
    submit_button = st.form_submit_button(label='Send')

# When the user submits a message
if submit_button and user_input:
    text_response, audio_base64 = get_response(user_input=user_input)
    # Append user message to the chat history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Your_wise_buddy", text_response))

    if audio_base64:
        # Decode the base64 audio string and play it
        audio_bytes = base64.b64decode(audio_base64)
        st.audio(audio_bytes, format='audio/mpeg')

# Display chat history
for speaker, message in st.session_state.chat_history:
    if speaker == "You" or speaker == "You (transcribed)":
        st.markdown(f"<div style='text-align: right'><strong>{speaker}:</strong> {message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: left'><strong>{speaker}:</strong> {message}</div>", unsafe_allow_html=True)
