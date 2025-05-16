#!/bin/bash
set -e

# Print banner
echo "================================================================="
echo "         Orpheus AI Assistant Deployment Script                  "
echo "================================================================="

# Check if docker and docker-compose are available
if ! command -v docker &> /dev/null; then
    echo "Error: docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create .env file if not exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp env.template .env
    echo "Please edit the .env file with your configuration before continuing."
    echo "Press Enter to continue or Ctrl+C to exit and edit the file."
    read
fi

# Create directories
echo "Creating necessary directories..."
mkdir -p models
mkdir -p coturn
mkdir -p n8n-workflows
mkdir -p assets

# Check if assets exist
if [ ! -f assets/user.png ] || [ ! -f assets/bot.png ]; then
    echo "Creating default avatar images..."
    # Creating dummy pixel as base64
    PIXEL="iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAMAAAD04JH5AAAAA1BMVEX///+nxBvIAAAASElEQVR4nO3BMQEAAADCoPVPbQhfoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABeA8XKAAFZcBBuAAAAAElFTkSuQmCC"
    echo $PIXEL | base64 -d > assets/user.png
    echo $PIXEL | base64 -d > assets/bot.png
fi

# Check if SSL certificates exist
if [ ! -f coturn/cert.pem ] || [ ! -f coturn/key.pem ]; then
    echo "Generating self-signed SSL certificates for TURN server..."
    openssl req -x509 -newkey rsa:4096 -keyout coturn/key.pem -out coturn/cert.pem -days 365 -nodes -subj "/CN=orpheus.ai"
fi

# Prepare n8n workflows
echo "Installing n8n workflows..."
cp -r n8n-workflows/* n8n-workflows/

# Check for GPU support
echo "Checking for GPU support..."
if [ -f /proc/driver/nvidia/version ]; then
    echo "NVIDIA driver detected. Using GPU configuration."
    GPU_SUPPORT=true
else
    echo "No NVIDIA driver detected. Using CPU configuration."
    GPU_SUPPORT=false
fi

# Build Docker images
echo "Building Docker images..."
docker-compose build

# Start services
echo "Starting services..."
docker-compose up -d

echo "================================================================="
echo "Orpheus AI Assistant deployment completed!"
echo ""
echo "Access the components at:"
echo "- Gradio Web Interface: http://localhost:7860"
echo "- n8n Workflow Editor: http://localhost:5678"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop: docker-compose down"
echo "=================================================================" 