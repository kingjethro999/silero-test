from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import torch
import io
import numpy as np
from src.silero.silero import silero_stt, silero_tts
import soundfile as sf

app = FastAPI()

device = torch.device('cpu')

# Load STT model
stt_model, stt_decoder, stt_utils = silero_stt(language='en', device=device)
read_batch, split_into_batches, read_audio, prepare_model_input = stt_utils

# Load TTS model (default English)
tts_model, tts_symbols, tts_sample_rate, tts_example_text, tts_apply = silero_tts(language='en', speaker='v3_en')
tts_model = tts_model.to(device)

@app.post('/stt')
async def stt_api(file: UploadFile = File(...)):
    # Read audio file into numpy array
    audio_bytes = await file.read()
    with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
        audio = f.read(dtype='float32')
        sample_rate = f.samplerate
    # Prepare input for model
    # Silero expects mono, 16kHz
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    if sample_rate != 16000:
        import torchaudio
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, 16000)
        audio = audio_tensor.squeeze(0).numpy()
        sample_rate = 16000
    batch = [audio]
    input_tensor = prepare_model_input(batch, device=device)
    output = stt_model(input_tensor)
    text = stt_decoder(output[0].cpu())
    return JSONResponse({"text": text})

class TTSRequest(BaseModel):
    text: str
    speaker: str = 'en_0'  # default speaker
    sample_rate: int = 48000

@app.post('/tts')
async def tts_api(req: TTSRequest):
    # Generate audio from text
    audio = tts_apply(
        texts=[req.text],
        model=tts_model,
        sample_rate=req.sample_rate,
        symbols=tts_symbols,
        device=device
    )[0]
    # Return as WAV
    buf = io.BytesIO()
    sf.write(buf, audio, req.sample_rate, format='WAV')
    buf.seek(0)
    return StreamingResponse(buf, media_type='audio/wav', headers={
        'Content-Disposition': 'attachment; filename="output.wav"'
    })

@app.get('/')
def root():
    return {"message": "Silero Models API. Use /stt for speech-to-text and /tts for text-to-speech."} 