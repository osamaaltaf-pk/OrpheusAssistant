import asyncio
import json
import logging
import os
import websockets
import threading
import queue
import uuid
import numpy as np
from typing import Optional, Dict, Any, Callable, List

# Configure logging
logging.basicConfig(
    level=logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger('webrtc_client')

class WebRTCClient:
    """WebRTC client for audio streaming with WebSocket signaling"""
    
    def __init__(self, signaling_url: str = "ws://localhost:8765"):
        self.signaling_url = signaling_url
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.peer_id: Optional[str] = None
        self.target_peer_id: Optional[str] = None
        self.ice_servers: List[Dict[str, Any]] = []
        
        # Audio data queues
        self.input_audio_queue = queue.Queue()  # Local audio to send
        self.output_audio_queue = queue.Queue()  # Received audio to play
        
        # Connection state
        self.connected = False
        self.connection_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Callbacks
        self.on_connected: Optional[Callable] = None
        self.on_disconnected: Optional[Callable] = None
        self.on_audio_received: Optional[Callable[[np.ndarray, int], None]] = None
        
        # Create a separate thread for asyncio event loop
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.ready_event = threading.Event()
    
    def start(self) -> None:
        """Start the WebRTC client"""
        if not self.thread.is_alive():
            self.thread.start()
            logger.info("WebRTC client thread started")
            # Wait for event loop to be ready
            if not self.ready_event.wait(timeout=5.0):
                logger.error("WebRTC client event loop not ready after 5 seconds")
                raise RuntimeError("WebRTC client failed to start")
    
    def stop(self) -> None:
        """Stop the WebRTC client"""
        if self.event_loop and self.thread.is_alive():
            asyncio.run_coroutine_threadsafe(self._disconnect(), self.event_loop)
            logger.info("WebRTC client stop requested")
    
    def connect(self, target_peer_id: Optional[str] = None) -> None:
        """Connect to a peer"""
        if self.event_loop:
            self.target_peer_id = target_peer_id
            asyncio.run_coroutine_threadsafe(self._connect(), self.event_loop)
            logger.info(f"WebRTC client connect requested to peer {target_peer_id}")
    
    def disconnect(self) -> None:
        """Disconnect from the current peer"""
        if self.event_loop:
            asyncio.run_coroutine_threadsafe(self._disconnect(), self.event_loop)
            logger.info("WebRTC client disconnect requested")
    
    def send_audio(self, audio_data: np.ndarray, sample_rate: int) -> None:
        """Send audio data to the peer"""
        if self.connected:
            # Convert audio data to format suitable for WebRTC
            # This is a simplified version - in real implementation,
            # audio would be encoded using a codec like Opus
            self.input_audio_queue.put((audio_data, sample_rate))
        else:
            logger.warning("Cannot send audio - not connected")
    
    def get_received_audio(self) -> Optional[tuple]:
        """Get received audio data if available"""
        try:
            return self.output_audio_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _run_event_loop(self) -> None:
        """Run asyncio event loop in a separate thread"""
        try:
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)
            # Signal that the event loop is ready
            self.ready_event.set()
            # Run the event loop
            self.event_loop.run_until_complete(self._run())
        except Exception as e:
            logger.exception(f"Error in WebRTC client event loop: {e}")
        finally:
            if self.event_loop:
                self.event_loop.close()
            logger.info("WebRTC client event loop stopped")
    
    async def _run(self) -> None:
        """Main coroutine for WebRTC client"""
        # Connect to signaling server
        await self._connect_to_signaling()
        # Start heartbeat
        self.heartbeat_task = asyncio.create_task(self._send_heartbeat())
        # Process signaling messages
        try:
            while True:
                await asyncio.sleep(0.1)  # Prevent CPU spinning
                if not self.websocket:
                    # Try to reconnect if disconnected
                    await self._connect_to_signaling()
                    continue
                
                # Check for audio data to send
                if self.connected:
                    try:
                        audio_data, sample_rate = self.input_audio_queue.get_nowait()
                        await self._send_audio_data(audio_data, sample_rate)
                    except queue.Empty:
                        pass  # No audio data to send
        except asyncio.CancelledError:
            logger.info("WebRTC client run task cancelled")
        except Exception as e:
            logger.exception(f"Error in WebRTC client run task: {e}")
        finally:
            # Clean up
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
            if self.websocket:
                await self.websocket.close()
    
    async def _connect_to_signaling(self) -> None:
        """Connect to the signaling server"""
        try:
            logger.info(f"Connecting to signaling server at {self.signaling_url}")
            self.websocket = await websockets.connect(self.signaling_url)
            
            # Register with the signaling server
            await self.websocket.send(json.dumps({
                'type': 'register',
                'peerId': self.peer_id or str(uuid.uuid4())
            }))
            
            # Handle the registration response
            response_text = await self.websocket.recv()
            response = json.loads(response_text)
            if response.get('type') == 'registered':
                self.peer_id = response.get('peerId')
                logger.info(f"Registered with signaling server, assigned peer ID: {self.peer_id}")
                
                # Start message handling
                asyncio.create_task(self._handle_signaling_messages())
                
                # If we have a target peer, connect to it
                if self.target_peer_id:
                    await self._connect()
            else:
                logger.error(f"Failed to register with signaling server: {response}")
                if self.websocket:
                    await self.websocket.close()
                    self.websocket = None
        except Exception as e:
            logger.exception(f"Error connecting to signaling server: {e}")
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
    
    async def _handle_signaling_messages(self) -> None:
        """Handle messages from the signaling server"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get('type')
                    
                    if msg_type == 'ice-config':
                        # Store ICE server configuration
                        self.ice_servers = data.get('config', {}).get('iceServers', [])
                        logger.info(f"Received ICE server configuration: {len(self.ice_servers)} servers")
                    
                    elif msg_type == 'offer':
                        # Handle incoming offer (someone wants to connect to us)
                        offer = data.get('offer')
                        from_peer_id = data.get('peerId')
                        logger.info(f"Received offer from peer {from_peer_id}")
                        
                        # In a real implementation, we would:
                        # 1. Create a RTCPeerConnection
                        # 2. Set the remote description using the offer
                        # 3. Create an answer
                        # 4. Set the local description
                        # 5. Send the answer to the peer
                        
                        # For now, let's just acknowledge the offer
                        await self.websocket.send(json.dumps({
                            'type': 'answer',
                            'answer': {'type': 'answer', 'sdp': 'dummy'},
                            'targetId': from_peer_id,
                            'peerId': self.peer_id
                        }))
                        
                        # Set connected state
                        self.connected = True
                        self.target_peer_id = from_peer_id
                        if self.on_connected:
                            self.on_connected()
                    
                    elif msg_type == 'answer':
                        # Handle incoming answer (response to our offer)
                        answer = data.get('answer')
                        from_peer_id = data.get('peerId')
                        logger.info(f"Received answer from peer {from_peer_id}")
                        
                        # In a real implementation, we would:
                        # 1. Set the remote description using the answer
                        
                        # Set connected state
                        self.connected = True
                        if self.on_connected:
                            self.on_connected()
                    
                    elif msg_type == 'ice-candidate':
                        # Handle ICE candidate
                        candidate = data.get('candidate')
                        from_peer_id = data.get('peerId')
                        logger.debug(f"Received ICE candidate from peer {from_peer_id}")
                        
                        # In a real implementation, we would:
                        # 1. Add the ICE candidate to the RTCPeerConnection
                    
                    elif msg_type == 'peer-disconnected':
                        # Handle peer disconnection
                        from_peer_id = data.get('peerId')
                        logger.info(f"Peer {from_peer_id} disconnected")
                        
                        if from_peer_id == self.target_peer_id:
                            self.connected = False
                            self.target_peer_id = None
                            if self.on_disconnected:
                                self.on_disconnected()
                    
                    elif msg_type == 'error':
                        # Handle error
                        error_msg = data.get('message')
                        logger.error(f"Received error from signaling server: {error_msg}")
                    
                    elif msg_type == 'heartbeat-ack':
                        # Heartbeat acknowledgement - no action needed
                        pass
                    
                    else:
                        logger.warning(f"Received unknown message type: {msg_type}")
                
                except json.JSONDecodeError:
                    logger.error(f"Received invalid JSON from signaling server: {message}")
                except Exception as e:
                    logger.exception(f"Error handling signaling message: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("Signaling server connection closed")
        except Exception as e:
            logger.exception(f"Error in signaling message handler: {e}")
        finally:
            # Clean up
            self.connected = False
            self.websocket = None
            if self.on_disconnected:
                self.on_disconnected()
    
    async def _connect(self) -> None:
        """Connect to a peer"""
        if not self.websocket or not self.peer_id:
            logger.error("Cannot connect - not connected to signaling server")
            return
        
        if not self.target_peer_id:
            logger.error("Cannot connect - no target peer specified")
            return
        
        try:
            logger.info(f"Initiating connection to peer {self.target_peer_id}")
            
            # In a real implementation, we would:
            # 1. Create a RTCPeerConnection
            # 2. Add audio tracks/transceivers
            # 3. Create an offer
            # 4. Set the local description
            # 5. Send the offer to the peer
            
            # For now, let's just send a dummy offer
            await self.websocket.send(json.dumps({
                'type': 'offer',
                'offer': {'type': 'offer', 'sdp': 'dummy'},
                'targetId': self.target_peer_id,
                'peerId': self.peer_id
            }))
            
        except Exception as e:
            logger.exception(f"Error connecting to peer: {e}")
    
    async def _disconnect(self) -> None:
        """Disconnect from the current peer"""
        self.connected = False
        self.target_peer_id = None
        
        if self.on_disconnected:
            self.on_disconnected()
        
        # In a real implementation, we would:
        # 1. Close the RTCPeerConnection
    
    async def _send_heartbeat(self) -> None:
        """Send periodic heartbeats to keep the connection alive"""
        try:
            while True:
                if self.websocket and self.peer_id:
                    try:
                        await self.websocket.send(json.dumps({
                            'type': 'heartbeat',
                            'peerId': self.peer_id
                        }))
                    except Exception as e:
                        logger.error(f"Error sending heartbeat: {e}")
                await asyncio.sleep(10)  # Send heartbeat every 10 seconds
        except asyncio.CancelledError:
            logger.info("Heartbeat task cancelled")
    
    async def _send_audio_data(self, audio_data: np.ndarray, sample_rate: int) -> None:
        """Send audio data to the peer"""
        # In a real implementation, we would:
        # 1. Encode the audio data using a codec like Opus
        # 2. Send it via the RTCPeerConnection data channel
        
        # For now, let's just log it
        logger.debug(f"Would send audio data: {audio_data.shape}, {sample_rate} Hz")
    
    async def _receive_audio_data(self, encoded_data: bytes) -> None:
        """Process received audio data"""
        # In a real implementation, we would:
        # 1. Decode the audio data using a codec like Opus
        # 2. Queue it for playback
        
        # For now, let's just create a dummy audio chunk
        sample_rate = 16000
        duration_sec = 0.1
        dummy_audio = np.zeros(int(sample_rate * duration_sec), dtype=np.float32)
        
        # Put it in the output queue
        self.output_audio_queue.put((dummy_audio, sample_rate))
        
        # Call the callback if registered
        if self.on_audio_received:
            self.on_audio_received(dummy_audio, sample_rate)

# Integration with PipeCat audio pipeline
class WebRTCAudioProcessor:
    """Integrates WebRTC with PipeCat audio pipeline"""
    
    def __init__(self, signaling_url: str = "ws://localhost:8765"):
        self.client = WebRTCClient(signaling_url)
        self.client.on_audio_received = self._on_audio_received
        self.last_received_audio = None
        self.last_received_sample_rate = None
    
    def start(self) -> None:
        """Start the WebRTC client"""
        self.client.start()
    
    def stop(self) -> None:
        """Stop the WebRTC client"""
        self.client.stop()
    
    def connect(self, peer_id: Optional[str] = None) -> None:
        """Connect to a peer"""
        self.client.connect(peer_id)
    
    def disconnect(self) -> None:
        """Disconnect from the current peer"""
        self.client.disconnect()
    
    def process_input(self, audio_data: np.ndarray, sample_rate: int) -> None:
        """Process audio input (send to peer)"""
        if self.client.connected:
            self.client.send_audio(audio_data, sample_rate)
    
    def process_output(self) -> Optional[tuple]:
        """Process audio output (receive from peer)"""
        return self.client.get_received_audio()
    
    def _on_audio_received(self, audio_data: np.ndarray, sample_rate: int) -> None:
        """Callback for received audio"""
        self.last_received_audio = audio_data
        self.last_received_sample_rate = sample_rate

# Example usage
if __name__ == "__main__":
    # Create WebRTC audio processor
    processor = WebRTCAudioProcessor()
    processor.start()
    
    # Connect to a peer (server in this case)
    processor.connect("server")
    
    # Generate some audio data
    sample_rate = 16000
    duration_sec = 1.0
    audio_data = np.random.randn(int(sample_rate * duration_sec)).astype(np.float32)
    
    # Send it to the peer
    processor.process_input(audio_data, sample_rate)
    
    # Receive audio from the peer
    received = processor.process_output()
    if received:
        audio, sr = received
        print(f"Received audio: {audio.shape}, {sr} Hz")
    
    # Clean up
    import time
    time.sleep(2)  # Wait for any pending operations
    processor.disconnect()
    processor.stop() 