import os
import io
import logging
import numpy as np
import onnxruntime as ort
import soundfile as sf
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydub import AudioSegment
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Global variable for model session
session = None

# Function to download model from cloud storage if needed
def download_model_if_needed():
    model_path = os.environ.get("MODEL_PATH", "wav2vec2_emotion.onnx")
    model_url = os.environ.get("MODEL_URL", None)
    
    # If model doesn't exist locally and URL is provided, download it
    if not os.path.exists(model_path) and model_url:
        try:
            logger.info(f"Model not found locally. Downloading from {model_url}")
            os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
            
            with open(model_path, "wb") as f:
                response = requests.get(model_url, stream=True)
                if not response.ok:
                    logger.error(f"Download failed with status code: {response.status_code}")
                    return False
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                chunk_size = 8192
                
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            logger.info(f"Downloaded {downloaded}/{total_size} bytes ({(downloaded/total_size)*100:.1f}%)")
            
            logger.info(f"Model downloaded successfully to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            return False
    return os.path.exists(model_path)

# Load the ONNX model with improved error handling
def load_model():
    global session
    try:
        # First check if we need to download the model
        if not download_model_if_needed():
            logger.warning("Model not available for download, continuing with local search")
        
        # List all directories and files recursively to help debug
        logger.info("Starting model loading process")
        all_files = []
        for root, dirs, files in os.walk('.', topdown=True):
            for file in files:
                if file.endswith('.onnx'):
                    all_files.append(os.path.join(root, file))
        
        logger.info(f"Found ONNX files: {all_files}")
        
        # Try multiple possible locations
        possible_paths = [
            os.environ.get("MODEL_PATH", "wav2vec2_emotion.onnx"),
            "wav2vec2_emotion.onnx",
            "./wav2vec2_emotion.onnx",
            os.path.join(os.path.dirname(__file__), "wav2vec2_emotion.onnx"),
            os.path.join(os.getcwd(), "wav2vec2_emotion.onnx"),
            "/app/wav2vec2_emotion.onnx",  # Common Railway path
            *all_files  # Add any .onnx files found during recursive search
        ]
        
        # Log all paths we're checking
        logger.info(f"Checking these paths: {possible_paths}")
        
        # Check which paths exist and log their size
        for path in possible_paths:
            try:
                exists = os.path.exists(path)
                size = os.path.getsize(path) if exists else "N/A"
                logger.info(f"Path: {path}, Exists: {exists}, Size: {size if exists else 'N/A'} bytes")
            except Exception as e:
                logger.warning(f"Error checking path {path}: {str(e)}")
        
        # Find first valid path
        model_path = next((path for path in possible_paths if os.path.exists(path)), None)
        
        if model_path is None:
            raise FileNotFoundError("Model file not found in any expected location")
        
        logger.info(f"Loading ONNX model from {model_path}")
        
        # Create inference session with additional logging
        try:
            # Try CPU execution provider first
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            logger.info("Model loaded successfully with CPU provider")
        except Exception as cpu_error:
            logger.warning(f"Failed to load model with CPU provider: {cpu_error}")
            # Try default providers
            session = ort.InferenceSession(model_path)
            logger.info("Model loaded successfully with default providers")
        
        # Log model inputs and outputs for debugging
        input_names = [input.name for input in session.get_inputs()]
        output_names = [output.name for output in session.get_outputs()]
        logger.info(f"Model inputs: {input_names}")
        logger.info(f"Model outputs: {output_names}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

# Root endpoint for health check
@app.get("/")
async def root():
    return {
        "status": "online", 
        "message": "API is running", 
        "model_loaded": session is not None
    }

# Try to load the model on startup
model_loaded = load_model()
id2label = {0: "calm", 1: "neutral", 2: "anxiety", 3: "confidence"}

# Function to convert MP3/OGG to WAV
def convert_audio_to_wav(audio_bytes, format):
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=format)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        return wav_io.getvalue()
    except Exception as e:
        logger.error(f"Error converting audio: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Audio conversion failed: {str(e)}")

# Function to preprocess audio
def preprocess_audio(audio_bytes, file_extension):
    try:
        if file_extension in ["mp3", "ogg"]:
            audio_bytes = convert_audio_to_wav(audio_bytes, file_extension)
        
        audio, samplerate = sf.read(io.BytesIO(audio_bytes))
        logger.info(f"Audio loaded: shape={audio.shape}, samplerate={samplerate}")
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)  # Convert stereo to mono
            logger.info(f"Converted to mono: shape={audio.shape}")
        
        return np.expand_dims(audio, axis=0).astype(np.float32)
    except Exception as e:
        logger.error(f"Error preprocessing audio: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Audio preprocessing failed: {str(e)}")

# Softmax function
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # Improve numerical stability
    return exp_logits / np.sum(exp_logits)

# Prediction endpoint
@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    global session
    
    # Try to load model if not loaded
    if session is None:
        logger.warning("Model not loaded, attempting to load...")
        model_loaded = load_model()
        if not model_loaded or session is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
    try:
        file_extension = file.filename.split(".")[-1].lower()
        logger.info(f"Processing file: {file.filename}, extension: {file_extension}")
        
        if file_extension not in ["wav", "mp3", "ogg"]:
            return {"error": "Unsupported file format. Please upload WAV, MP3, or OGG."}
        
        audio_bytes = await file.read()
        logger.info(f"File size: {len(audio_bytes)} bytes")
        
        input_tensor = preprocess_audio(audio_bytes, file_extension)
        logger.info(f"Preprocessed tensor shape: {input_tensor.shape}")

        # Get input name from model
        input_name = session.get_inputs()[0].name
        logger.info(f"Using input name: {input_name}")
        
        inputs = {input_name: input_tensor}
        logger.info("Running inference...")
        
        outputs = session.run(None, inputs)
        logger.info(f"Inference complete, output shape: {outputs[0].shape}")
        
        probabilities = softmax(outputs[0][0])
        top_2_indices = np.argsort(probabilities)[-2:][::-1]
        top_2_emotions = {id2label[i]: f"{round(probabilities[i] * 100)}%" for i in top_2_indices}
        
        logger.info(f"Prediction results: {top_2_emotions}")
        return {"top_emotions": ", ".join([f"{k}: {v}" for k, v in top_2_emotions.items()])}
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Enhanced debugging endpoint
@app.get("/debug")
async def debug():
    try:
        # Get environment variables (hide sensitive info)
        env_vars = {k: "***" if "SECRET" in k or "KEY" in k or "TOKEN" in k or "PASSWORD" in k or "URL" in k else v 
                   for k, v in os.environ.items()}
        
        # Check for model in various locations
        possible_paths = [
            os.environ.get("MODEL_PATH", "wav2vec2_emotion.onnx"),
            "wav2vec2_emotion.onnx",
            "./wav2vec2_emotion.onnx",
            os.path.join(os.path.dirname(__file__), "wav2vec2_emotion.onnx"),
            os.path.join(os.getcwd(), "wav2vec2_emotion.onnx"),
            "/app/wav2vec2_emotion.onnx",
        ]
        
        model_checks = {}
        for path in possible_paths:
            try:
                exists = os.path.exists(path)
                size = os.path.getsize(path) if exists else None
                is_file = os.path.isfile(path) if exists else None
                permissions = None
                if exists:
                    try:
                        permissions = oct(os.stat(path).st_mode & 0o777)
                    except:
                        permissions = "Failed to get permissions"
                
                model_checks[path] = {
                    "exists": exists,
                    "size_mb": size / (1024 * 1024) if size else None,
                    "is_file": is_file,
                    "permissions": permissions
                }
            except Exception as e:
                model_checks[path] = {"error": str(e)}
        
        # Get recursive file listing
        all_files = []
        try:
            for root, dirs, files in os.walk('.', topdown=True):
                for file in files:
                    if file.endswith('.onnx'):
                        full_path = os.path.join(root, file)
                        all_files.append({
                            "path": full_path,
                            "size_mb": os.path.getsize(full_path) / (1024 * 1024) if os.path.exists(full_path) else None
                        })
        except Exception as e:
            all_files = [f"Error walking directory: {str(e)}"]
        
        # Check disk space
        disk_info = {}
        try:
            import shutil
            disk = shutil.disk_usage("/")
            disk_info = {
                "total_gb": disk.total / (1024**3),
                "used_gb": disk.used / (1024**3),
                "free_gb": disk.free / (1024**3),
                "percent_used": disk.used / disk.total * 100
            }
        except:
            disk_info = {"error": "Could not get disk info"}
        
        # Check memory usage
        memory_info = {}
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = {
                "rss_mb": process.memory_info().rss / (1024 * 1024),
                "vms_mb": process.memory_info().vms / (1024 * 1024)
            }
        except:
            memory_info = {"error": "psutil not available"}
            
        # Get ONNX Runtime info
        ort_info = {
            "version": ort.__version__,
            "available_providers": ort.get_available_providers(),
            "device": ort.get_device()
        }
        
        return {
            "files_in_directory": os.listdir("."),
            "onnx_files": all_files,
            "model_checks": model_checks,
            "current_directory": os.getcwd(),
            "env_vars": env_vars,
            "python_version": os.sys.version,
            "model_loaded": session is not None,
            "memory_usage": memory_info,
            "disk_info": disk_info,
            "onnxruntime_info": ort_info
        }
    except Exception as e:
        return {"error": str(e)}

# Add endpoint to explicitly reload model
@app.post("/reload-model")
async def reload_model():
    global session
    session = None
    success = load_model()
    if success and session is not None:
        return {"status": "success", "message": "Model reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")

# Main execution block
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
