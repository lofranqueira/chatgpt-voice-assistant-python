import sounddevice as sd
import numpy as np
import whisper
import scipy.io.wavfile as wav
from openai import OpenAI

import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("Gravando... fale algo!")

duracao = 5
taxa = 44100

audio = sd.rec(int(duracao * taxa), samplerate=taxa, channels=1)
sd.wait()

wav.write("audio.wav", taxa, audio)

print("Áudio gravado!")

model = whisper.load_model("small")

resultado = model.transcribe(
    "audio.wav",
    language="pt",
    task="transcribe"
)

texto = resultado["text"]

print("Você disse:")
print(texto)

print("Enviando para o ChatGPT...")

resposta = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {"role": "system", "content": "Responda sempre em português."},
    {"role": "user", "content": texto}])

print("ChatGPT respondeu:")
print(resposta.choices[0].message.content)