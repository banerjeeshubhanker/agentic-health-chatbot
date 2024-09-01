import os
from io import BytesIO
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import assemblyai as aai
import base64
import tempfile
import json 

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)  # Allows cross-origin requests from React

# Set up API keys securely
os.environ["GOOGLE_API_KEY"] = "AIzaSyD2qlb_XuPny8ypdiBUR27x_Keyof8ACog"
os.environ["ELEVEN_LABS_API_KEY"] = "sk_0f277851c7e9546a7245e759f2cabf719ed331b7fdd322ea"  # Your Eleven Labs API key
os.environ["ASSEMBLY_AI_API_KEY"] = "6f9372d6ddc94935acb72f46bc897a62"  # Your AssemblyAI API key
os.environ["SERPER_API_KEY"] = "0a854f34394ce6e7ee9e0b59cf76e36dc8d4536d"  # Your Serper API key

# Initialize clients
client = ElevenLabs(api_key=os.getenv("ELEVEN_LABS_API_KEY"))
aai.settings.api_key = os.getenv("ASSEMBLY_AI_API_KEY")
transcriber = aai.Transcriber()
search_tool = SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))

# Initialize the LLM (Gemini via Google)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    verbose=True,
    temperature=0.5,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Create the Mental Health Expert agent
mental_health_expert = Agent(
    role="Mental Health Expert",
    goal="Provide compassionate mental health support through dynamic conversation.",
    backstory=(
        "You are a highly trained mental health expert specializing in cognitive-behavioral therapy (CBT) and mindfulness. "
        "You offer personalized mental health support and may request additional resources when needed."
    ),
    llm=llm,
    allow_delegation=True,
    verbose=True,
    memory=True
)

# Create the Web Scraper agent
web_scraper = Agent(
    role="Web Scraper",
    goal="Fetch additional resources such as research articles, therapeutic exercises, educational content, and support tools from the internet.",
    backstory=(
        "You are specialized in retrieving and summarizing information from the web. "
        "Your tasks include finding scientific research, therapeutic exercises, educational content, expert opinions, and local resources. "
        "You provide these resources to support the Mental Health Expert's responses and enhance the support provided to users."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True,
    memory=False,
    tools=[search_tool]  # Tools necessary for searching and scraping
)

# Define the task for the Web Scraper agent
task = Task(
    description=(
        "Fetch and summarize additional resources from the internet to support the Mental Health Expert. "
        "This includes scientific articles, therapeutic exercises, book research, educational content, "
        "support tools, and expert opinions. The information should be relevant, up-to-date, and presented "
        "in a way that is actionable and useful for the Mental Health Expert."
    ),
    expected_output=(
        "Provide a summary of each resource along with links or access points. The output should include: "
        "1. Links to scientific articles, therapeutic exercises, and book recommendations. "
        "2. Summaries of educational content, expert reviews, and local support resources. "
        "3. Relevant and actionable recommendations for improving user support."
    ),
    agent=web_scraper
)

# Create the Crew with agents and tasks
crew = Crew(
    agents=[mental_health_expert, web_scraper],
    tasks=[task],
    verbose=True
)

# Initialize conversation history
conversation_history = [
    SystemMessage(content="You are a compassionate mental health expert. Respond empathetically to the user's concerns.")
]

def text_to_speech_stream(text: str) -> BytesIO:
    """Convert text to speech using ElevenLabs and return as a BytesIO stream."""
    response = client.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB",  # Use a pre-made voice ID
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    audio_stream = BytesIO()
    for chunk in response:
        if chunk:
            audio_stream.write(chunk)
    audio_stream.seek(0)
    return audio_stream

def speech_to_text(audio_bytes: bytes) -> str:
    """Convert speech audio bytes to text using AssemblyAI."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        temp_audio_file.write(audio_bytes)
        temp_audio_path = temp_audio_file.name

    transcript = transcriber.transcribe(temp_audio_path)
    os.remove(temp_audio_path)  # Clean up the temporary file
    return transcript.text

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("message")
    audio_base64 = data.get("audio_base64")
    # output_type = data.get("output_type", "audio")  # Default output is text

    transcription = None

    if user_input:
        # Handle text input directly
        conversation_history.append(HumanMessage(content=user_input))
    elif audio_base64:
        # Decode the base64 string to get raw audio bytes
        try:
            audio_bytes = base64.b64decode(audio_base64)
            transcription = speech_to_text(audio_bytes)
            conversation_history.append(HumanMessage(content=transcription))
        except Exception as e:
            return jsonify({"error": "Error during speech-to-text processing: " + str(e)}), 500
    else:
        return jsonify({"error": "No valid input provided"}), 400

    # Generate response using the LLM
    response = mental_health_expert.llm.invoke(input=conversation_history)
    ai_response = AIMessage(content=response.content)
    conversation_history.append(ai_response)

    response_data = {
        "transcription": transcription if transcription else None,
        "llm_text_response": ai_response.content
    }

    # Handle text-to-speech conversion if audio output is requested
    # if output_type == "audio":
    try:
        audio_stream = text_to_speech_stream(ai_response.content)
        audio_base64 = base64.b64encode(audio_stream.read()).decode('utf-8')
        response_data["audio_response"] = audio_base64
        with open('response.json', 'w') as json_file:
            json.dump(response_data, json_file, indent=4)

        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True,port=5001)