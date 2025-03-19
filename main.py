from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import onnxruntime
import io
import soundfile as sf
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Voice Tone Analysis API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the ONNX model
try:
    onnx_session = onnxruntime.InferenceSession("wav2vec2_emotion.onnx")
    input_name = onnx_session.get_inputs()[0].name
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Define emotion labels
emotion_labels = ["angry", "happy", "neutral", "sad"]

# Model response type
class PredictionResult(BaseModel):
    emotion: str
    confidence: float
    all_scores: dict

@app.get("/")
def read_root():
    return {"message": "Voice Tone Analysis API", "status": "online"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

def preprocess_audio(audio_data, sample_rate):
    # Resample if needed (wav2vec2 expects 16kHz)
    if sample_rate != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
    
    # Convert to float32 if not already
    audio_data = audio_data.astype(np.float32)
    
    # Normalize audio (if needed)
    if np.abs(audio_data).max() > 1.0:
        audio_data = audio_data / np.abs(audio_data).max()
        
    return audio_data

@app.post("/predict", response_model=PredictionResult)
async def predict_emotion(file: UploadFile = File(...)):
    try:
        # Read audio file
        content = await file.read()
        audio_data, sample_rate = sf.read(io.BytesIO(content))
        
        # Preprocess audio
        processed_audio = preprocess_audio(audio_data, sample_rate)
        
        # Run inference
        model_input = {input_name: np.expand_dims(processed_audio, axis=0)}
        raw_prediction = onnx_session.run(None, model_input)
        
        # Process prediction
        scores = raw_prediction[0][0]
        predicted_class = np.argmax(scores)
        predicted_emotion = emotion_labels[predicted_class]
        
        # Convert scores to probabilities using softmax
        scores_exp = np.exp(scores - np.max(scores))
        probabilities = scores_exp / scores_exp.sum()
        
        # Create response
        emotion_scores = {emotion: float(prob) for emotion, prob in zip(emotion_labels, probabilities)}
        
        return PredictionResult(
            emotion=predicted_emotion,
            confidence=float(probabilities[predicted_class]),
            all_scores=emotion_scores
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

# If running the script directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
