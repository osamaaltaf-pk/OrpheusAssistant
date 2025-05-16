#!/bin/bash
set -e

# Check if environment variables are set
if [ -z "$SERVER_BASE_URL" ]; then
  echo "WARNING: SERVER_BASE_URL is not set. Using default: http://127.0.0.1:1234"
  export SERVER_BASE_URL="http://127.0.0.1:1234"
fi

if [ -z "$N8N_BASE_URL" ]; then
  echo "WARNING: N8N_BASE_URL is not set. Using default: http://localhost:5678"
  export N8N_BASE_URL="http://localhost:5678"
fi

if [ -z "$N8N_API_KEY" ]; then
  echo "WARNING: N8N_API_KEY is not set. Using a placeholder value."
  export N8N_API_KEY="your_n8n_api_key"
fi

# Check if CUDA is available
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'Device count: {torch.cuda.device_count()}')"
python3 -c "import torch; print(f'Current device: {torch.cuda.current_device()}')"
python3 -c "import torch; print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Start WebRTC server in the background
echo "Starting WebRTC signaling server..."
python3 webrtc_server.py &
WEBRTC_PID=$!

# Wait for WebRTC server to start
sleep 2

# Check if WebRTC server is running
if ps -p $WEBRTC_PID > /dev/null; then
  echo "WebRTC signaling server started successfully."
else
  echo "WARNING: WebRTC signaling server failed to start. Continuing without it."
fi

# Start main application
echo "Starting Orpheus AI Assistant..."
exec python3 RT_Orpheus_Gradio.py 