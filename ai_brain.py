import asyncio
import json
import logging
import time
import threading
import queue
import sys
import os
from typing import Optional, Dict, Any

# Core libraries for AI model
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    pipeline
)

# Speech recognition and TTS
import speech_recognition as sr
import pyttsx3
import whisper

# VTube Studio integration
import pyvts
import websockets

# Audio processing
import pyaudio
import wave

class AIVTuberCore:
    """Core AI VTuber implementation with local model support"""

    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 quantization_type: str = "4bit",
                 whisper_model_size: str = "base",
                 use_local_whisper: bool = True,
                 vts_enabled: bool = True):

        self.model_name = model_name
        self.quantization_type = quantization_type
        self.whisper_model_size = whisper_model_size
        self.use_local_whisper = use_local_whisper
        self.vts_enabled = vts_enabled

        # Initialize components
        self.model = None
        self.tokenizer = None
        self.tts_engine = None
        self.speech_recognizer = None
        self.whisper_model = None
        self.vts_client = None

        # Audio settings
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.is_speaking = False

        # Conversation context
        self.conversation_history = []
        self.max_history = 10

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.initialize_components()

    def initialize_components(self):
        self.logger.info("Initializing components...")
        self.load_quantized_model()
        self.setup_tts()
        self.setup_speech_recognition()
        if self.use_local_whisper:
            self.load_whisper_model()
        if self.vts_enabled:
            self.setup_vts()

    def load_quantized_model(self):
        self.logger.info(f"Loading model: {self.model_name} with quantization: {self.quantization_type}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.quantization_type == "4bit",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )

    def setup_tts(self):
        self.logger.info("Setting up TTS engine...")
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty("rate", 175)

    def setup_speech_recognition(self):
        self.logger.info("Setting up speech recognition...")
        self.speech_recognizer = sr.Recognizer()

    def load_whisper_model(self):
        self.logger.info(f"Loading Whisper model: {self.whisper_model_size}")
        self.whisper_model = whisper.load_model(self.whisper_model_size)

    def setup_vts(self):
        self.logger.info("Initializing VTube Studio client...")
        self.vts_client = pyvts.vts()
        asyncio.create_task(self.vts_client.connect())

    def listen_and_transcribe(self):
        self.logger.info("Listening...")
        with sr.Microphone() as source:
            audio = self.speech_recognizer.listen(source)
            try:
                if self.use_local_whisper:
                    audio_data = audio.get_wav_data()
                    with open("temp.wav", "wb") as f:
                        f.write(audio_data)
                    result = self.whisper_model.transcribe("temp.wav")
                    return result['text']
                else:
                    return self.speech_recognizer.recognize_google(audio)
            except Exception as e:
                self.logger.error(f"Speech recognition failed: {e}")
                return ""

    def generate_response(self, prompt: str) -> str:
        self.logger.info("Generating response...")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_length=100, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def speak(self, text: str):
        self.logger.info("Speaking...")
        self.is_speaking = True
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
        self.is_speaking = False

    def conversation_loop(self):
        self.logger.info("Starting conversation loop...")
        while True:
            text = self.listen_and_transcribe()
            if text:
                self.logger.info(f"User said: {text}")
                self.conversation_history.append(text)
                if len(self.conversation_history) > self.max_history:
                    self.conversation_history.pop(0)

                prompt = "\n".join(self.conversation_history)
                response = self.generate_response(prompt)
                self.conversation_history.append(response)
                self.speak(response)

    def start(self):
        self.logger.info("Starting AI VTuber Core...")
        conversation_thread = threading.Thread(target=self.conversation_loop)
        conversation_thread.start()

if __name__ == "__main__":
    core = AIVTuberCore(
        model_name="microsoft/DialoGPT-medium",
        quantization_type="4bit",
        whisper_model_size="base",
        use_local_whisper=True,
        vts_enabled=False  # Set to True if using pyvts with VTube Studio
    )
    core.start()
