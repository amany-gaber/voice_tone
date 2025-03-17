import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
import onnxruntime as ort
import soundfile as sf
import io
from pydub import AudioSegment
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add a simple root endpoint for healthcheck
@app.get("/")
async def root():
    return {"status": "online", "message": "API is running"}

# Load the ONNX model with better error handling
try:
    # Try different paths to find the model
    possible_paths = [
        "wav2vec2_emotion.onnx",  # Direct path
        os.path.join(os.path.dirname(__file__), "wav2vec2_emotion.onnx"),  # Relative to file
        os.path.join(os.getcwd(), "wav2vec2_emotion.onnx"),  # Current working directory
    ]
    
    model_path = None
    for path in possible_paths:
        logger.info(f"Checking path: {path}")
        if os.path.exists(path):
            model_path = path
            logger.info(f"Found model at: {path}")
            break
    
    if model_path is None:
        logger.error("Model file not found in any of the expected locations")
        raise FileNotFoundError("Model file not found")
    
    # Check file size
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
    logger.info(f"Model file size: {file_size:.2f} MB")
    
    # Load the model
    logger.info("Loading ONNX model...")
    session = ort.InferenceSession(model_path)
    logger.info("Model loaded successfully")
    
    # Ensure labels are correctly mapped
    id2label = {0: "calm", 1: "neutral", 2: "anxiety", 3: "confidence"}
    
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    # Create a placeholder session that will raise a clear error when used
    session = None

# Function to convert MP3/OGG to WAV if needed
def convert_audio_to_wav(audio_bytes, format):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=format)
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    return wav_io.getvalue()

# Function to preprocess the audio file
def preprocess_audio(audio_bytes, file_extension):
    if file_extension in ["mp3", "ogg"]:
        audio_bytes = convert_audio_to_wav(audio_bytes, file_extension)

    audio, samplerate = sf.read(io.BytesIO(audio_bytes))
    
    if len(audio.shape) > 1:  
        audio = np.mean(audio, axis=1)  # Convert stereo to mono
    
    input_tensor = np.expand_dims(audio, axis=0).astype(np.float32)
    return input_tensor

# Softmax function to convert logits to probabilities
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # Improve numerical stability
    return exp_logits / np.sum(exp_logits)

@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
    file_extension = file.filename.split(".")[-1].lower()
    supported_formats = ["wav", "mp3", "ogg"]

    if file_extension not in supported_formats:
        return {"error": "Unsupported file format. Please upload WAV, MP3, or OGG."}

    audio_bytes = await file.read()
    input_tensor = preprocess_audio(audio_bytes, file_extension)
    
    inputs = {session.get_inputs()[0].name: input_tensor}
    outputs = session.run(None, inputs)
    
    predicted_logits = outputs[0][0]
    probabilities = softmax(predicted_logits)

    # Get top 2 emotions
    top_2_indices = np.argsort(probabilities)[-2:][::-1]  # Get indices of top 2 emotions
    top_2_emotions = {id2label[i]: f"{round(probabilities[i] * 100)}%" for i in top_2_indices}

    # Create response string in the required format
    response_text = f"Top 2 emotions ({', '.join([f'{k}: {v}' for k, v in top_2_emotions.items()])})"

    return {"top_emotions": response_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
