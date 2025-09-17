from fastapi import FastAPI, UploadFile, File
import requests
import tempfile

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Ayah Tafseer API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(await file.read())
        temp_path = temp.name

    # TODO: Run Whisper ASR, normalize, match, fetch tafsir
    # For now, return a dummy response
    return {
        "surah": 2,
        "ayah": 255,
        "tafsir": "This is where Tafsir Ibn Kathir would go.",
        "debug_audio_path": temp_path
    }

