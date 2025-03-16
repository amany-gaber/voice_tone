import numpy as np
from fastapi import FastAPI, File, UploadFile
import onnxruntime as ort
import soundfile as sf
import io
from pydub import AudioSegment

app = FastAPI()

# Load the model
model_path = r"wav2vec2_emotion.onnx"
session = ort.InferenceSession(model_path)

emotion_labels = ["calm", "neutral", "anxiety", "confidence"]

# Convert MP3/OGG to WAV if needed
def convert_audio_to_wav(audio_bytes, format):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=format)
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    return wav_io.getvalue()

# Preprocess the audio file
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

    # Format results as percentages
    top_2_emotions = {
        emotion_labels[i]: f"{round(probabilities[i] * 100)}%" for i in top_2_indices
    }

    # Create response text
    result_text = ", ".join([f"{k}: {v}" for k, v in top_2_emotions.items()])

    return [
         "Top 2 predicted emotions",
         result_text
    ]