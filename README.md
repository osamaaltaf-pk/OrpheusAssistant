# 🧠 Orpheus AI Assistant

An advanced **offline AI voice assistant** that combines real-time WebRTC streaming, Orpheus TTS, Whisper STT, LLaMA LLM, and n8n workflow integration. Designed for private, low-latency, and fully autonomous voice interaction.

---

## 🚀 Features

### 🎤 Real-Time Communication
- WebRTC-based audio streaming
- Bi-directional voice input/output
- Auto-reconnect and fallback handling

### 🔊 Audio Processing
- PipeCat-powered audio pipeline
- Noise reduction and normalization
- GPU acceleration (when available)

### 🤖 AI Models
- **Orpheus TTS 3B 4-bit Quant** — natural speech synthesis
- **Whisper Large** — high-accuracy speech recognition
- **LLaMA 3.2 3B Instruct** — intelligent language reasoning

### 🔌 Tool Integration
- **n8n** automation for:
  - Calendar and reminders
  - Email actions
  - Custom workflow execution

---

## ⚙️ Setup

### 1. Create Environment

```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
.\venv\Scripts\activate        # Windows

pip install -r requirements.txt

2. Configure .env
Create a .env file in the root directory:

SERVER_BASE_URL=http://127.0.0.1:1234
N8N_BASE_URL=http://localhost:5678
N8N_API_KEY=your_n8n_api_key

3. Model Setup
Install LM Studio and load LLaMA-3.2-3B-Instruct

Configure Orpheus TTS 3B 4bit Quant

Install Whisper Large via openai-whisper or faster-whisper

4. n8n Setup
Install and run n8n:

n8n start

Import included workflow templates (/n8n_workflows)

Connect calendar, email, and other integrations

▶️ Run the App
Start the WebRTC server:

bash
Copy
Edit

python webrtc_server.py

Start the n8n server:

n8n start

Launch the main interface:

python RT_Orpheus_Gradio.py

Open your browser: http://localhost:7860

🧬 Architecture

[ Mic Input ]
     ↓
[ WebRTC Server ]
     ↓
[ PipeCat Audio ]
     ↓
[ Whisper STT ] → [ LLaMA LLM ] → [ n8n Tool Integration ]
                                         ↓
                                   Calendar / Email / Reminders
                                         ↓
                                [ Orpheus TTS Output ]


🧪 Troubleshooting
Issue	Solution
WebRTC not working	Check browser support and network config
Low audio quality	Tune filters in pipecat_audio.py
n8n not triggering	Verify API key and .env configuration

📹 Demos
🔴 Real-Time Demo & Walkthrough: YouTube

🧪 Code Walkthrough (4_11_25_Version2.py): YouTube

🕰️ Original Demo: YouTube

🤝 Contributing
Fork this repository

Create a new feature branch

Submit a pull request

📄 License
This project is licensed under the MIT License.
