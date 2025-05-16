# Orpheus AI Assistant

An advanced AI assistant that combines Orpheus TTS, Whisper STT, and Llama LLM with WebRTC streaming, PipeCat audio processing, and n8n integration for tool calling capabilities.

## Features

- **Real-time Voice Communication**
  - WebRTC-based audio streaming
  - Low-latency voice input/output
  - Automatic connection management and fallback

- **Advanced Audio Processing**
  - PipeCat-based audio pipeline
  - Noise reduction and normalization
  - GPU-accelerated processing when available

- **AI Models**
  - Orpheus TTS 3b 4 Quant for natural speech synthesis
  - Whisper Large for accurate speech recognition
  - Meta-Llama/Llama-3.2-3B-Instruct for intelligent responses

- **Tool Integration**
  - n8n workflow automation
  - Calendar and appointment management
  - Email and reminder capabilities

## Setup

1. **Environment Setup**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Configuration**
   Create a `.env` file with the following settings:
   ```env
   SERVER_BASE_URL=http://127.0.0.1:1234
   N8N_BASE_URL=http://localhost:5678
   N8N_API_KEY=your_n8n_api_key
   ```

3. **Model Setup**
   - Download and set up LMStudio with Llama-3.2-3B-Instruct
   - Configure Orpheus TTS 3b 4 Quant
   - Install Whisper Large model

4. **n8n Setup**
   - Install and configure n8n
   - Import provided workflow templates
   - Set up required integrations (calendar, email, etc.)

## Running the Application

1. **Start the WebRTC Server**
   ```bash
   python webrtc_server.py
   ```

2. **Start n8n**
   ```bash
   n8n start
   ```

3. **Launch the Main Application**
   ```bash
   python RT_Orpheus_Gradio.py
   ```

4. Open your browser and navigate to `http://localhost:7860`

## Architecture

The application uses a modular architecture with the following components:

- **WebRTC Server**: Handles real-time audio streaming
- **PipeCat Pipeline**: Processes audio streams with configurable components
- **n8n Integration**: Manages external tool calls and automation
- **Gradio Interface**: Provides the web-based user interface

## Performance Optimization

- GPU acceleration for audio processing when available
- WebRTC optimization for low-latency communication
- Efficient audio pipeline with minimal processing overhead

## Troubleshooting

- **WebRTC Issues**: Check browser WebRTC support and network connectivity
- **Audio Quality**: Adjust PipeCat pipeline parameters in `pipecat_audio.py`
- **n8n Connection**: Verify n8n server status and API key configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - see LICENSE file for details

# Orpheus - Consider the 4_11_25_Version2.py file

# Advanced Dev? --- try the real-time version RT_Orpheus_Gradio.py

Real-time demo and code walk:  https://youtu.be/QjfcbqCfyfM
![image](https://github.com/user-attachments/assets/49b65654-0206-486f-9f9a-5b4a816292bb)

Canopyai Orpheus &amp; LMStudio: 100% Uncensored Private Offline chat 

Current 4_11_25_Version2.py code walk:  https://youtu.be/80PlrvhpMzI

Old Demo:  https://youtu.be/HYmszx3jO5g

![image](https://github.com/user-attachments/assets/abd1e544-2868-4300-aa5c-156649d7291e)

