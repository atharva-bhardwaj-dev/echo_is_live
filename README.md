# echo_is_live
this is a fully local AI-powered VTuber assistant. It listens to your voice, transcribes it using OpenAI's Whisper, generates responses using a quantized large language model (like DialoGPT), and speaks back with text-to-speech. Optionally, it can integrate with VTube Studio to animate your model in sync with the conversation.

Ideal for VTubers, Twitch streamers, or anyone who wants a talking AI companion on-screen — completely offline and customizable.

---

## ✨ Features

- 🎙️ Voice-to-text using [Whisper](https://github.com/openai/whisper) (local or online fallback)
- 🧠 AI dialogue generation via quantized transformers (4-bit with BitsAndBytes)
- 🔊 Text-to-speech with `pyttsx3` (offline TTS)
- 🧍 VTube Studio support via `pyvts` (toggleable)
- 🧵 Multithreaded real-time loop (listen → respond → speak)
- 💻 Fully local, no API keys or cloud required

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/AI-VTuber-Core.git
cd AI-VTuber-Core
