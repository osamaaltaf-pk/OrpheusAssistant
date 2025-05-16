"""
Orpheus AI Assistant Integration Module

This module integrates all components of the Orpheus AI Assistant:
- WebRTC for real-time audio communication
- PipeCat for audio processing
- n8n for tool calling capabilities
- Orpheus TTS and Whisper STT for voice interaction
- Llama model for natural language understanding
"""

import asyncio
import logging
import os
import threading
import time
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

# Import components
from pipecat_audio import AudioPipeline, AudioBlock, NoiseReducer, AudioNormalizer, AudioResampler
from webrtc_client import WebRTCAudioProcessor
from n8n_integration import N8NIntegration, ToolCaller, CalendarTool, AppointmentTool

# Configure logging
logging.basicConfig(
    level=logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger('orpheus_integration')

class OrpheusIntegration:
    """Main integration class for Orpheus AI Assistant"""
    
    def __init__(self):
        # Load environment variables
        self.server_base_url = os.environ.get("SERVER_BASE_URL", "http://127.0.0.1:1234")
        self.n8n_base_url = os.environ.get("N8N_BASE_URL", "http://localhost:5678")
        self.n8n_api_key = os.environ.get("N8N_API_KEY", "your_n8n_api_key")
        self.webrtc_url = os.environ.get("WEBRTC_URL", "ws://localhost:8765")
        
        # Initialize components
        self._init_audio_pipeline()
        self._init_webrtc()
        self._init_n8n()
        
        # State variables
        self.is_running = False
        self.thread = None
        
    def _init_audio_pipeline(self):
        """Initialize PipeCat audio pipeline"""
        logger.info("Initializing PipeCat audio pipeline...")
        self.audio_pipeline = AudioPipeline()
        self.audio_pipeline.add_processor(NoiseReducer(threshold=0.02))
        self.audio_pipeline.add_processor(AudioNormalizer(target_db=-20))
        self.audio_pipeline.add_processor(AudioResampler(target_sr=16000))
        logger.info("PipeCat audio pipeline initialized with 3 processors")
        
    def _init_webrtc(self):
        """Initialize WebRTC audio processor"""
        logger.info(f"Initializing WebRTC client with signaling server at {self.webrtc_url}...")
        self.webrtc = WebRTCAudioProcessor(self.webrtc_url)
        
    def _init_n8n(self):
        """Initialize n8n tool integration"""
        logger.info(f"Initializing n8n integration with server at {self.n8n_base_url}...")
        try:
            self.n8n = N8NIntegration(self.n8n_base_url, self.n8n_api_key)
            self.tool_caller = ToolCaller(self.n8n)
            self.calendar_tool = CalendarTool(self.tool_caller)
            self.appointment_tool = AppointmentTool(self.tool_caller)
            logger.info(f"n8n integration initialized, found {len(self.tool_caller.tools)} tools")
        except Exception as e:
            logger.error(f"Failed to initialize n8n integration: {e}")
            self.n8n = None
            self.tool_caller = None
            self.calendar_tool = None
            self.appointment_tool = None
    
    def start(self):
        """Start all components"""
        if self.is_running:
            logger.warning("Integration is already running")
            return
        
        logger.info("Starting Orpheus integration...")
        
        # Start audio pipeline
        self.audio_pipeline.start()
        
        # Start WebRTC
        self.webrtc.start()
        
        # Set running state
        self.is_running = True
        
        # Start background thread for continuous processing
        self.thread = threading.Thread(target=self._background_processing, daemon=True)
        self.thread.start()
        
        logger.info("Orpheus integration started successfully")
    
    def stop(self):
        """Stop all components"""
        if not self.is_running:
            logger.warning("Integration is not running")
            return
        
        logger.info("Stopping Orpheus integration...")
        
        # Set running state to False to stop background thread
        self.is_running = False
        
        # Wait for background thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        # Stop WebRTC
        self.webrtc.stop()
        
        # Stop audio pipeline
        self.audio_pipeline.stop()
        
        logger.info("Orpheus integration stopped successfully")
    
    def connect_webrtc(self, peer_id: Optional[str] = None):
        """Connect to a WebRTC peer"""
        if not self.is_running:
            logger.warning("Cannot connect - integration is not running")
            return False
        
        try:
            self.webrtc.connect(peer_id)
            logger.info(f"WebRTC connection initiated to peer {peer_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebRTC peer: {e}")
            return False
    
    def disconnect_webrtc(self):
        """Disconnect from WebRTC peer"""
        if not self.is_running:
            logger.warning("Cannot disconnect - integration is not running")
            return
        
        self.webrtc.disconnect()
        logger.info("WebRTC disconnected")
    
    def process_audio(self, audio_data: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        """Process audio through the pipeline"""
        if not self.is_running:
            logger.warning("Cannot process audio - integration is not running")
            return None
        
        # Process through PipeCat pipeline
        result = self.audio_pipeline.process(audio_data, sample_rate)
        
        if result:
            # Send processed audio to WebRTC if connected
            if self.webrtc.client.connected:
                self.webrtc.process_input(result.data, result.sample_rate)
            
            return result.data
        else:
            logger.warning("Audio processing returned no result")
            return None
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict:
        """Call an n8n tool"""
        if not self.is_running or not self.tool_caller:
            logger.warning("Cannot call tool - integration is not running or n8n not initialized")
            return {"status": "error", "message": "n8n integration not available"}
        
        try:
            result = await self.tool_caller.call_tool(tool_name, parameters)
            logger.info(f"Tool '{tool_name}' called with result status: {result.get('status')}")
            return result
        except Exception as e:
            logger.error(f"Failed to call tool '{tool_name}': {e}")
            return {"status": "error", "message": str(e)}
    
    async def create_calendar_event(self, title: str, start_time, end_time, description: str = "") -> Dict:
        """Create a calendar event"""
        if not self.is_running or not self.calendar_tool:
            logger.warning("Cannot create event - integration is not running or calendar tool not initialized")
            return {"status": "error", "message": "Calendar integration not available"}
        
        try:
            result = await self.calendar_tool.create_event(title, start_time, end_time, description)
            logger.info(f"Calendar event '{title}' created with result status: {result.get('status')}")
            return result
        except Exception as e:
            logger.error(f"Failed to create calendar event: {e}")
            return {"status": "error", "message": str(e)}
    
    async def schedule_appointment(self, title: str, date, duration_minutes: int, attendees: List[str]) -> Dict:
        """Schedule an appointment"""
        if not self.is_running or not self.appointment_tool:
            logger.warning("Cannot schedule appointment - integration is not running or appointment tool not initialized")
            return {"status": "error", "message": "Appointment integration not available"}
        
        try:
            result = await self.appointment_tool.schedule_appointment(title, date, duration_minutes, attendees)
            logger.info(f"Appointment '{title}' scheduled with result status: {result.get('status')}")
            return result
        except Exception as e:
            logger.error(f"Failed to schedule appointment: {e}")
            return {"status": "error", "message": str(e)}
    
    def _background_processing(self):
        """Background thread for continuous processing"""
        logger.info("Background processing thread started")
        
        while self.is_running:
            try:
                # Check for audio from WebRTC
                received = self.webrtc.process_output()
                if received:
                    audio_data, sample_rate = received
                    
                    # Process through PipeCat pipeline
                    result = self.audio_pipeline.process(audio_data, sample_rate)
                    
                    # Here you would typically forward the processed audio to the STT module
                    # and then to the LLM for processing
                    logger.debug(f"Processed WebRTC audio: {audio_data.shape} -> {result.data.shape if result else 'None'}")
            
            except Exception as e:
                logger.error(f"Error in background processing: {e}")
            
            # Sleep to avoid high CPU usage
            time.sleep(0.01)
        
        logger.info("Background processing thread stopped")

# Example usage
if __name__ == "__main__":
    # Initialize integration
    integration = OrpheusIntegration()
    
    # Start all components
    integration.start()
    
    try:
        # Connect to WebRTC peer
        integration.connect_webrtc("server")
        
        # Generate some test audio
        sample_rate = 16000
        duration_sec = 1.0
        test_audio = np.random.randn(int(sample_rate * duration_sec)).astype(np.float32)
        
        # Process audio
        processed_audio = integration.process_audio(test_audio, sample_rate)
        if processed_audio is not None:
            print(f"Processed audio shape: {processed_audio.shape}")
        
        # Let it run for a while
        print("Press Ctrl+C to stop...")
        import time
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Stop all components
        integration.stop() 