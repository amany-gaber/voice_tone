import os
import io
import logging
import numpy as np
import onnxruntime as ort
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydub import AudioSegment

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Root endpoint for health check
@app.get("/")
async def root():
    return {"status": "online", "message": "API is running"}

# Load the ONNX model with error handling
try:
    possible_paths = [
        "wav2vec2_emotion.onnx",
        os.path.join(os.path.dirname(__file__), "wav2vec2_emotion.onnx"),
        os.path.join(os.getcwd(), "wav2vec2_emotion.onnx"),
    ]
    
    model_path = next((path for path in possible_paths if os.path.exists(path)), None)
    
    if model_path is None:
        raise FileNotFoundError("Model file not found in any expected location")
    
    logger.info(f"Loading ONNX model from {model_path}")
    session = ort.InferenceSession(model_path)
    logger.info("Model loaded successfully")
    
    id2label = {0: "calm", 1: "neutral", 2: "anxiety", 3: "confidence"}
    
except Exception as e:
    logger.error(f"Error loading model: {e}")
    session = None

# Function to convert MP3/OGG to WAV
def convert_audio_to_wav(audio_bytes, format):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=format)
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    return wav_io.getvalue()

# Function to preprocess audio
def preprocess_audio(audio_bytes, file_extension):
    if file_extension in ["mp3", "ogg"]:
        audio_bytes = convert_audio_to_wav(audio_bytes, file_extension)
    
    audio, samplerate = sf.read(io.BytesIO(audio_bytes))
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)  # Convert stereo to mono
    
    return np.expand_dims(audio, axis=0).astype(np.float32)

# Softmax function
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # Improve numerical stability
    return exp_logits / np.sum(exp_logits)

# Prediction endpoint
@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in ["wav", "mp3", "ogg"]:
        return {"error": "Unsupported file format. Please upload WAV, MP3, or OGG."}
    
    audio_bytes = await file.read()
    input_tensor = preprocess_audio(audio_bytes, file_extension)
    
    inputs = {session.get_inputs()[0].name: input_tensor}
    outputs = session.run(None, inputs)
    
    probabilities = softmax(outputs[0][0])
    top_2_indices = np.argsort(probabilities)[-2:][::-1]
    top_2_emotions = {id2label[i]: f"{round(probabilities[i] * 100)}%" for i in top_2_indices}
    
    return {"top_emotions": ", ".join([f"{k}: {v}" for k, v in top_2_emotions.items()])}

# Debugging endpoint
@app.get("/debug")
async def debug():
    model_info = {"exists": os.path.exists("wav2vec2_emotion.onnx"),
                  "size_mb": os.path.getsize("wav2vec2_emotion.onnx") / (1024 * 1024) if os.path.exists("wav2vec2_emotion.onnx") else None,
                  "path": os.path.abspath("wav2vec2_emotion.onnx") if os.path.exists("wav2vec2_emotion.onnx") else None}
    
    return {"files_in_directory": os.listdir("."),
            "model_info": model_info,
            "current_directory": os.getcwd(),
            "python_version": os.sys.version}

# Main execution block
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
