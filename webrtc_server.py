import asyncio
import json
import logging
import os
import ssl
import uuid
import websockets
from dataclasses import dataclass, asdict, field
from typing import Dict, Set, Optional, List

# Configure logging
logging.basicConfig(
    level=logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger('webrtc_server')

# Load environment variables
TURN_USERNAME = os.environ.get("TURN_USERNAME", "orpheus")
TURN_PASSWORD = os.environ.get("TURN_PASSWORD", "orpheus_turn_password")
TURN_REALM = os.environ.get("TURN_REALM", "orpheus.ai")
TURN_SERVER = os.environ.get("TURN_SERVER", "coturn:3478")
STUN_SERVERS = os.environ.get("STUN_SERVERS", "stun:stun.l.google.com:19302,stun:stun1.l.google.com:19302").split(",")

@dataclass
class RTCPeer:
    id: str
    websocket: websockets.WebSocketServerProtocol
    ice_candidates: List[dict] = field(default_factory=list)
    sdp_offer: Optional[dict] = None
    sdp_answer: Optional[dict] = None
    connected_to: Optional[str] = None
    last_heartbeat: float = 0

@dataclass
class IceConfig:
    stun_servers: List[str]
    turn_server: str
    turn_username: str
    turn_password: str
    turn_realm: str

    def to_dict(self):
        ice_servers = []
        
        # Add STUN servers
        for stun in self.stun_servers:
            if stun.strip():
                ice_servers.append({
                    "urls": stun.strip()
                })
        
        # Add TURN server if configured
        if self.turn_server and self.turn_username and self.turn_password:
            ice_servers.append({
                "urls": f"turn:{self.turn_server}",
                "username": self.turn_username,
                "credential": self.turn_password,
                "credentialType": "password"
            })
            
        return {
            "iceServers": ice_servers
        }

class WebRTCSignalingServer:
    def __init__(self):
        self.peers: Dict[str, RTCPeer] = {}
        self.connections: Set[str] = set()
        self.ice_config = IceConfig(
            stun_servers=STUN_SERVERS,
            turn_server=TURN_SERVER,
            turn_username=TURN_USERNAME,
            turn_password=TURN_PASSWORD,
            turn_realm=TURN_REALM
        )
        
        # Start heartbeat check
        asyncio.create_task(self.check_heartbeats())
        
    async def register(self, websocket: websockets.WebSocketServerProtocol, peer_id: str):
        if peer_id in self.peers:
            raise ValueError(f"Peer ID {peer_id} already exists")
        
        if not peer_id:
            peer_id = str(uuid.uuid4())
            
        import time
        peer = RTCPeer(
            id=peer_id, 
            websocket=websocket, 
            ice_candidates=[],
            last_heartbeat=time.time()
        )
        self.peers[peer_id] = peer
        logger.info(f"Registered peer {peer_id}")
        
        # Send ICE config
        await websocket.send(json.dumps({
            "type": "ice-config",
            "config": self.ice_config.to_dict()
        }))
        
        return peer_id
        
    async def unregister(self, peer_id: str):
        if peer_id in self.peers:
            # Notify connected peer about disconnection
            connected_to = self.peers[peer_id].connected_to
            if connected_to and connected_to in self.peers:
                try:
                    await self.peers[connected_to].websocket.send(json.dumps({
                        "type": "peer-disconnected",
                        "peerId": peer_id
                    }))
                except Exception as e:
                    logger.error(f"Error notifying peer {connected_to} about disconnection: {e}")
            
            # Remove connection tracking
            if (peer_id, connected_to) in self.connections:
                self.connections.remove((peer_id, connected_to))
            if (connected_to, peer_id) in self.connections:
                self.connections.remove((connected_to, peer_id))
                
            # Remove peer
            del self.peers[peer_id]
            logger.info(f"Unregistered peer {peer_id}")
            
    async def check_heartbeats(self):
        """Check for stale connections and remove them"""
        import time
        while True:
            try:
                current_time = time.time()
                stale_peers = []
                
                for peer_id, peer in self.peers.items():
                    # If no heartbeat for 30 seconds, consider the peer disconnected
                    if current_time - peer.last_heartbeat > 30:
                        stale_peers.append(peer_id)
                
                for peer_id in stale_peers:
                    logger.info(f"Removing stale peer {peer_id} due to heartbeat timeout")
                    await self.unregister(peer_id)
                    
                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error in heartbeat check: {e}")
                await asyncio.sleep(10)  # Continue checking even if there's an error
            
    async def handle_message(self, websocket: websockets.WebSocketServerProtocol, message: str):
        try:
            import time
            data = json.loads(message)
            msg_type = data.get('type')
            peer_id = data.get('peerId')
            
            # Update heartbeat timestamp for this peer
            if peer_id in self.peers:
                self.peers[peer_id].last_heartbeat = time.time()
            
            if msg_type == 'register':
                registered_id = await self.register(websocket, peer_id)
                return {'type': 'registered', 'peerId': registered_id}
                
            elif msg_type == 'heartbeat':
                return {'type': 'heartbeat-ack'}
                
            elif msg_type == 'offer':
                target_id = data.get('targetId')
                if target_id in self.peers:
                    self.peers[peer_id].connected_to = target_id
                    self.peers[target_id].connected_to = peer_id
                    self.connections.add((peer_id, target_id))
                    
                    self.peers[target_id].sdp_offer = data.get('offer')
                    await self.peers[target_id].websocket.send(json.dumps({
                        'type': 'offer',
                        'offer': data.get('offer'),
                        'peerId': peer_id
                    }))
                else:
                    return {'type': 'error', 'message': f"Target peer {target_id} not found"}
                    
            elif msg_type == 'answer':
                target_id = data.get('targetId')
                if target_id in self.peers:
                    self.peers[target_id].sdp_answer = data.get('answer')
                    await self.peers[target_id].websocket.send(json.dumps({
                        'type': 'answer',
                        'answer': data.get('answer'),
                        'peerId': peer_id
                    }))
                else:
                    return {'type': 'error', 'message': f"Target peer {target_id} not found"}
                    
            elif msg_type == 'ice-candidate':
                target_id = data.get('targetId')
                if target_id in self.peers:
                    candidate = data.get('candidate')
                    self.peers[target_id].ice_candidates.append(candidate)
                    await self.peers[target_id].websocket.send(json.dumps({
                        'type': 'ice-candidate',
                        'candidate': candidate,
                        'peerId': peer_id
                    }))
                else:
                    return {'type': 'error', 'message': f"Target peer {target_id} not found"}
                    
            elif msg_type == 'list-peers':
                # Return list of available peers (excluding self)
                available_peers = [
                    {'id': pid, 'connected': bool(peer.connected_to)}
                    for pid, peer in self.peers.items()
                    if pid != peer_id
                ]
                return {'type': 'peers-list', 'peers': available_peers}
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON in message")
            return {'type': 'error', 'message': "Invalid JSON format"}
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return {'type': 'error', 'message': str(e)}

    async def connection_handler(self, websocket: websockets.WebSocketServerProtocol, path: str):
        peer_id = None
        try:
            async for message in websocket:
                response = await self.handle_message(websocket, message)
                if response:
                    await websocket.send(json.dumps(response))
                    # If this is a registration response, capture the peer_id
                    if response.get('type') == 'registered':
                        peer_id = response.get('peerId')
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed for peer {peer_id}")
            # If we know the peer_id, unregister it
            if peer_id:
                await self.unregister(peer_id)
            # Otherwise, find and remove the disconnected peer by websocket reference
            else:
                for pid, peer in list(self.peers.items()):
                    if peer.websocket == websocket:
                        await self.unregister(pid)
                        break
        except Exception as e:
            logger.error(f"Error in connection handler: {e}")
            if peer_id:
                await self.unregister(peer_id)

async def start_server():
    server = WebRTCSignalingServer()
    
    # Create SSL context for secure WebSocket
    ssl_context = None
    cert_path = os.environ.get("SSL_CERT_PATH", "cert.pem")
    key_path = os.environ.get("SSL_KEY_PATH", "key.pem")
    
    if os.path.exists(cert_path) and os.path.exists(key_path):
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        try:
            ssl_context.load_cert_chain(cert_path, key_path)
            logger.info(f"SSL certificates loaded from {cert_path} and {key_path}")
        except Exception as e:
            logger.error(f"Error loading SSL certificates: {e}")
            ssl_context = None
    else:
        logger.warning(f"SSL certificates not found at {cert_path} and {key_path}, running in unsecure mode")
    
    host = os.environ.get("WEBRTC_HOST", "0.0.0.0")
    port = int(os.environ.get("WEBRTC_PORT", "8765"))
    
    async with websockets.serve(
        server.connection_handler,
        host,
        port,
        ssl=ssl_context
    ):
        protocol = "wss" if ssl_context else "ws"
        logger.info(f"WebRTC Signaling Server started on {protocol}://{host}:{port}")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}") 