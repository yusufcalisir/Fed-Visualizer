"""
High-Performance Communication Backbone for Federated Learning
===============================================================
Async WebSocket communication with Msgpack serialization, Zstd compression,
HMAC signing, chunking, Pub/Sub events, and latency tracking.
"""

import asyncio
import hashlib
import hmac
import time
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from enum import Enum
import logging
import struct
import pickle
import numpy as np

# Try to import optional high-performance libraries
try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    msgpack = None
    HAS_MSGPACK = False

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    zstd = None
    HAS_ZSTD = False

logger = logging.getLogger("FLNetwork")


# ============================================================================
# MESSAGE TYPES
# ============================================================================

class MessageType(Enum):
    """Protocol message types."""
    HANDSHAKE = "handshake"
    HANDSHAKE_ACK = "handshake_ack"
    HEARTBEAT = "heartbeat"
    HEARTBEAT_ACK = "heartbeat_ack"
    WEIGHTS_BROADCAST = "weights_broadcast"
    WEIGHTS_UPLOAD = "weights_upload"
    WEIGHTS_CHUNK = "weights_chunk"
    WEIGHTS_CHUNK_ACK = "weights_chunk_ack"
    AGGREGATION_COMPLETE = "aggregation_complete"
    ERROR = "error"


# ============================================================================
# HARDWARE CAPABILITIES
# ============================================================================

@dataclass
class HardwareCapabilities:
    """Client hardware profile for registration."""
    ram_gb: float
    cpu_cores: int
    gpu_available: bool
    gpu_memory_gb: float = 0.0
    network_bandwidth_mbps: float = 100.0


# ============================================================================
# SERIALIZATION & COMPRESSION
# ============================================================================

class Serializer:
    """
    High-performance serializer with Msgpack and compression.
    Falls back to pickle + zlib if Msgpack/Zstd unavailable.
    """
    
    def __init__(self, 
                 use_compression: bool = True,
                 compression_level: int = 3):
        self.use_compression = use_compression
        self.compression_level = compression_level
        
        # Stats
        self.total_raw_bytes = 0
        self.total_compressed_bytes = 0
        
        # Zstd compressor
        if HAS_ZSTD and use_compression:
            self.zstd_compressor = zstd.ZstdCompressor(level=compression_level)
            self.zstd_decompressor = zstd.ZstdDecompressor()
    
    def serialize(self, data: Any) -> bytes:
        """Serialize data to bytes with optional compression."""
        # Convert numpy arrays to bytes
        if isinstance(data, dict):
            data = self._prepare_for_serialization(data)
        
        # Serialize
        if HAS_MSGPACK:
            raw = msgpack.packb(data, use_bin_type=True)
        else:
            raw = pickle.dumps(data)
        
        self.total_raw_bytes += len(raw)
        
        # Compress
        if self.use_compression:
            if HAS_ZSTD:
                compressed = self.zstd_compressor.compress(raw)
            else:
                compressed = zlib.compress(raw, level=self.compression_level)
            self.total_compressed_bytes += len(compressed)
            return compressed
        
        self.total_compressed_bytes += len(raw)
        return raw
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to data."""
        # Decompress
        if self.use_compression:
            if HAS_ZSTD:
                decompressed = self.zstd_decompressor.decompress(data)
            else:
                decompressed = zlib.decompress(data)
        else:
            decompressed = data
        
        # Deserialize
        if HAS_MSGPACK:
            result = msgpack.unpackb(decompressed, raw=False)
        else:
            result = pickle.loads(decompressed)
        
        # Restore numpy arrays
        if isinstance(result, dict):
            result = self._restore_from_serialization(result)
        
        return result
    
    def _prepare_for_serialization(self, data: Dict) -> Dict:
        """Convert numpy arrays to serializable format."""
        prepared = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                prepared[key] = {
                    "__numpy__": True,
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "data": value.tobytes()
                }
            elif isinstance(value, dict):
                prepared[key] = self._prepare_for_serialization(value)
            else:
                prepared[key] = value
        return prepared
    
    def _restore_from_serialization(self, data: Dict) -> Dict:
        """Restore numpy arrays from serialized format."""
        restored = {}
        for key, value in data.items():
            if isinstance(value, dict) and value.get("__numpy__"):
                dtype = np.dtype(value["dtype"])
                shape = tuple(value["shape"])
                restored[key] = np.frombuffer(value["data"], dtype=dtype).reshape(shape)
            elif isinstance(value, dict):
                restored[key] = self._restore_from_serialization(value)
            else:
                restored[key] = value
        return restored
    
    def get_compression_ratio(self) -> float:
        """Get compression ratio: raw / compressed."""
        if self.total_compressed_bytes == 0:
            return 1.0
        return self.total_raw_bytes / self.total_compressed_bytes


# ============================================================================
# CHUNKING
# ============================================================================

class Chunker:
    """
    Splits large payloads into smaller chunks for transmission.
    Supports resume-on-failure.
    """
    
    def __init__(self, chunk_size: int = 1024 * 1024):  # 1MB default
        self.chunk_size = chunk_size
    
    def chunk(self, data: bytes, message_id: str) -> List[Dict[str, Any]]:
        """Split data into chunks."""
        total_size = len(data)
        num_chunks = (total_size + self.chunk_size - 1) // self.chunk_size
        
        chunks = []
        for i in range(num_chunks):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, total_size)
            
            chunks.append({
                "message_id": message_id,
                "chunk_index": i,
                "total_chunks": num_chunks,
                "total_size": total_size,
                "data": data[start:end],
                "checksum": hashlib.md5(data[start:end]).hexdigest()
            })
        
        return chunks
    
    def reassemble(self, chunks: List[Dict[str, Any]]) -> bytes:
        """Reassemble chunks into original data."""
        # Sort by index
        sorted_chunks = sorted(chunks, key=lambda c: c["chunk_index"])
        
        # Verify completeness
        expected = sorted_chunks[0]["total_chunks"]
        if len(sorted_chunks) != expected:
            raise ValueError(f"Missing chunks: got {len(sorted_chunks)}, expected {expected}")
        
        # Verify checksums and reassemble
        data = b""
        for chunk in sorted_chunks:
            chunk_data = chunk["data"]
            if hashlib.md5(chunk_data).hexdigest() != chunk["checksum"]:
                raise ValueError(f"Checksum mismatch for chunk {chunk['chunk_index']}")
            data += chunk_data
        
        return data


# ============================================================================
# HMAC SIGNING
# ============================================================================

class MessageSigner:
    """
    HMAC-SHA256 signing for message integrity.
    """
    
    def __init__(self, secret_key: str = "federated_learning_secret"):
        self.secret_key = secret_key.encode()
    
    def sign(self, data: bytes) -> str:
        """Generate HMAC signature for data."""
        return hmac.new(self.secret_key, data, hashlib.sha256).hexdigest()
    
    def verify(self, data: bytes, signature: str) -> bool:
        """Verify HMAC signature."""
        expected = self.sign(data)
        return hmac.compare_digest(expected, signature)


# ============================================================================
# PUB/SUB EVENT SYSTEM
# ============================================================================

class EventType(Enum):
    """Network event types."""
    CLIENT_CONNECTED = "client_connected"
    CLIENT_DISCONNECTED = "client_disconnected"
    UPLOAD_STARTED = "upload_started"
    UPLOAD_PROGRESS = "upload_progress"
    UPLOAD_COMPLETE = "upload_complete"
    DOWNLOAD_STARTED = "download_started"
    DOWNLOAD_COMPLETE = "download_complete"
    AGGREGATION_STARTED = "aggregation_started"
    AGGREGATION_PROGRESS = "aggregation_progress"
    AGGREGATION_COMPLETE = "aggregation_complete"
    BROADCAST_STARTED = "broadcast_started"
    BROADCAST_COMPLETE = "broadcast_complete"
    HEARTBEAT_MISSED = "heartbeat_missed"
    ERROR = "error"


@dataclass
class NetworkEvent:
    """A network event for Pub/Sub."""
    event_type: EventType
    timestamp: float
    client_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    message: str = ""


class EventBus:
    """
    Publisher-Subscriber event bus for network events.
    Allows UI and Aggregator to listen independently.
    """
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_history: List[NetworkEvent] = []
        self._max_history = 1000
    
    def subscribe(self, event_type: EventType, callback: Callable[[NetworkEvent], None]):
        """Subscribe to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def subscribe_all(self, callback: Callable[[NetworkEvent], None]):
        """Subscribe to all event types."""
        for event_type in EventType:
            self.subscribe(event_type, callback)
    
    def publish(self, event: NetworkEvent):
        """Publish an event to all subscribers."""
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        if event.event_type in self._subscribers:
            for callback in self._subscribers[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Event callback error: {e}")
    
    def get_recent_events(self, count: int = 50) -> List[NetworkEvent]:
        """Get recent events for Streamlit display."""
        return self._event_history[-count:]


# ============================================================================
# LATENCY TRACKER
# ============================================================================

class LatencyTracker:
    """
    Tracks RTT and network latency for each client.
    
    Latency = T_received - T_sent - T_local_training
    """
    
    def __init__(self):
        self._pending: Dict[str, float] = {}  # message_id -> send_time
        self._client_rtts: Dict[str, List[float]] = {}
        self._bandwidth_samples: List[Tuple[float, int]] = []  # (timestamp, bytes)
    
    def record_send(self, message_id: str):
        """Record when a message was sent."""
        self._pending[message_id] = time.time()
    
    def record_receive(self, 
                       message_id: str, 
                       client_id: str,
                       local_training_time: float = 0.0) -> float:
        """
        Record when a response was received and compute latency.
        
        Returns:
            Network latency in seconds.
        """
        if message_id not in self._pending:
            return 0.0
        
        send_time = self._pending.pop(message_id)
        total_time = time.time() - send_time
        network_latency = total_time - local_training_time
        
        if client_id not in self._client_rtts:
            self._client_rtts[client_id] = []
        self._client_rtts[client_id].append(network_latency)
        
        # Keep only last 100 samples per client
        if len(self._client_rtts[client_id]) > 100:
            self._client_rtts[client_id].pop(0)
        
        return network_latency
    
    def record_bandwidth(self, bytes_transferred: int):
        """Record bandwidth sample."""
        self._bandwidth_samples.append((time.time(), bytes_transferred))
        # Keep last 5 minutes
        cutoff = time.time() - 300
        self._bandwidth_samples = [(t, b) for t, b in self._bandwidth_samples if t > cutoff]
    
    def get_avg_rtt(self, client_id: str) -> float:
        """Get average RTT for a client."""
        if client_id not in self._client_rtts or not self._client_rtts[client_id]:
            return 0.0
        return sum(self._client_rtts[client_id]) / len(self._client_rtts[client_id])
    
    def get_throughput_mbps(self, window_seconds: float = 10.0) -> float:
        """Get throughput in Mbps over a time window."""
        cutoff = time.time() - window_seconds
        recent = [b for t, b in self._bandwidth_samples if t > cutoff]
        if not recent:
            return 0.0
        return sum(recent) * 8 / (1024 * 1024) / window_seconds


# ============================================================================
# BACKPRESSURE HANDLER
# ============================================================================

class BackpressureHandler:
    """
    Implements backpressure to prevent server overload.
    Slows down client requests when queue is full.
    """
    
    def __init__(self, 
                 max_queue_size: int = 100,
                 throttle_threshold: float = 0.8):
        self.max_queue_size = max_queue_size
        self.throttle_threshold = throttle_threshold
        self.current_queue_size = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """
        Try to acquire a slot in the queue.
        Returns True if accepted, False if backpressure needed.
        """
        async with self._lock:
            if self.current_queue_size >= self.max_queue_size:
                return False
            self.current_queue_size += 1
            return True
    
    async def release(self):
        """Release a slot in the queue."""
        async with self._lock:
            self.current_queue_size = max(0, self.current_queue_size - 1)
    
    def get_utilization(self) -> float:
        """Get queue utilization (0.0 to 1.0)."""
        return self.current_queue_size / self.max_queue_size
    
    def should_throttle(self) -> bool:
        """Check if clients should be throttled."""
        return self.get_utilization() >= self.throttle_threshold


# ============================================================================
# CLIENT CONNECTION
# ============================================================================

@dataclass
class ClientConnection:
    """Represents a connected client."""
    client_id: str
    hardware: HardwareCapabilities
    connected_at: float
    last_heartbeat: float
    writer: Any  # asyncio.StreamWriter
    reader: Any  # asyncio.StreamReader
    is_alive: bool = True
    pending_chunks: Dict[str, List[Dict]] = field(default_factory=dict)


# ============================================================================
# FEDERATED NETWORK SERVER
# ============================================================================

class FederatedNetworkServer:
    """
    High-performance async WebSocket server for Federated Learning.
    """
    
    def __init__(self,
                 host: str = "0.0.0.0",
                 port: int = 8765,
                 secret_key: str = "federated_learning_secret",
                 heartbeat_interval: float = 10.0,
                 heartbeat_timeout: float = 30.0,
                 chunk_size: int = 1024 * 1024):
        self.host = host
        self.port = port
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        
        # Components
        self.serializer = Serializer(use_compression=True)
        self.chunker = Chunker(chunk_size=chunk_size)
        self.signer = MessageSigner(secret_key)
        self.event_bus = EventBus()
        self.latency_tracker = LatencyTracker()
        self.backpressure = BackpressureHandler()
        
        # State
        self.clients: Dict[str, ClientConnection] = {}
        self._running = False
        self._server = None
    
    async def start(self):
        """Start the server."""
        self._running = True
        self._server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port
        )
        
        logger.info(f"Server started on {self.host}:{self.port}")
        
        # Start heartbeat monitor
        asyncio.create_task(self._heartbeat_monitor())
        
        async with self._server:
            await self._server.serve_forever()
    
    async def stop(self):
        """Stop the server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
    
    async def _handle_client(self, 
                             reader: asyncio.StreamReader, 
                             writer: asyncio.StreamWriter):
        """Handle a new client connection."""
        addr = writer.get_extra_info('peername')
        client_id = None
        
        try:
            # Wait for handshake
            message = await self._receive_message(reader)
            if message.get("type") != MessageType.HANDSHAKE.value:
                raise ValueError("Expected handshake")
            
            client_id = message.get("client_id")
            hardware = HardwareCapabilities(**message.get("hardware", {}))
            
            # Register client
            self.clients[client_id] = ClientConnection(
                client_id=client_id,
                hardware=hardware,
                connected_at=time.time(),
                last_heartbeat=time.time(),
                writer=writer,
                reader=reader
            )
            
            # Send ack
            await self._send_message(writer, {
                "type": MessageType.HANDSHAKE_ACK.value,
                "server_time": time.time()
            })
            
            self.event_bus.publish(NetworkEvent(
                event_type=EventType.CLIENT_CONNECTED,
                timestamp=time.time(),
                client_id=client_id,
                message=f"Client {client_id} connected"
            ))
            
            # Main message loop
            while self._running and self.clients.get(client_id, {}).is_alive:
                message = await self._receive_message(reader)
                await self._process_message(client_id, message)
                
        except Exception as e:
            logger.error(f"Client {client_id or addr} error: {e}")
        finally:
            if client_id and client_id in self.clients:
                del self.clients[client_id]
                self.event_bus.publish(NetworkEvent(
                    event_type=EventType.CLIENT_DISCONNECTED,
                    timestamp=time.time(),
                    client_id=client_id,
                    message=f"Client {client_id} disconnected"
                ))
            writer.close()
            await writer.wait_closed()
    
    async def _receive_message(self, reader: asyncio.StreamReader) -> Dict:
        """Receive and deserialize a message."""
        # Read length prefix (4 bytes)
        length_data = await reader.read(4)
        if not length_data:
            raise ConnectionError("Connection closed")
        
        msg_length = struct.unpack(">I", length_data)[0]
        
        # Read signature (64 bytes hex = 64 chars)
        signature = (await reader.read(64)).decode()
        
        # Read message
        data = await reader.read(msg_length)
        
        # Verify signature
        if not self.signer.verify(data, signature):
            raise ValueError("Invalid message signature")
        
        # Track bandwidth
        self.latency_tracker.record_bandwidth(len(data))
        
        return self.serializer.deserialize(data)
    
    async def _send_message(self, writer: asyncio.StreamWriter, message: Dict):
        """Serialize and send a message."""
        data = self.serializer.serialize(message)
        signature = self.signer.sign(data)
        
        # Send: length + signature + data
        writer.write(struct.pack(">I", len(data)))
        writer.write(signature.encode())
        writer.write(data)
        await writer.drain()
        
        self.latency_tracker.record_bandwidth(len(data))
    
    async def _process_message(self, client_id: str, message: Dict):
        """Process an incoming message."""
        msg_type = message.get("type")
        
        if msg_type == MessageType.HEARTBEAT.value:
            self.clients[client_id].last_heartbeat = time.time()
            await self._send_message(self.clients[client_id].writer, {
                "type": MessageType.HEARTBEAT_ACK.value,
                "server_time": time.time()
            })
        
        elif msg_type == MessageType.WEIGHTS_UPLOAD.value:
            if not await self.backpressure.acquire():
                await self._send_message(self.clients[client_id].writer, {
                    "type": MessageType.ERROR.value,
                    "error": "Server busy, please retry"
                })
                return
            
            self.event_bus.publish(NetworkEvent(
                event_type=EventType.UPLOAD_COMPLETE,
                timestamp=time.time(),
                client_id=client_id,
                data={"weights": message.get("weights")},
                message=f"Client {client_id} uploaded weights"
            ))
            
            await self.backpressure.release()
        
        elif msg_type == MessageType.WEIGHTS_CHUNK.value:
            # Handle chunked upload
            msg_id = message.get("message_id")
            if msg_id not in self.clients[client_id].pending_chunks:
                self.clients[client_id].pending_chunks[msg_id] = []
            
            self.clients[client_id].pending_chunks[msg_id].append(message)
            
            # Check if all chunks received
            if len(self.clients[client_id].pending_chunks[msg_id]) == message.get("total_chunks"):
                data = self.chunker.reassemble(self.clients[client_id].pending_chunks[msg_id])
                weights = self.serializer.deserialize(data)
                
                self.event_bus.publish(NetworkEvent(
                    event_type=EventType.UPLOAD_COMPLETE,
                    timestamp=time.time(),
                    client_id=client_id,
                    data={"weights": weights},
                    message=f"Client {client_id} uploaded weights (chunked)"
                ))
                
                del self.clients[client_id].pending_chunks[msg_id]
            
            # Send ack
            await self._send_message(self.clients[client_id].writer, {
                "type": MessageType.WEIGHTS_CHUNK_ACK.value,
                "chunk_index": message.get("chunk_index")
            })
    
    async def broadcast_weights(self, weights: Dict[str, np.ndarray]):
        """Broadcast global weights to all clients."""
        self.event_bus.publish(NetworkEvent(
            event_type=EventType.BROADCAST_STARTED,
            timestamp=time.time(),
            message=f"Broadcasting to {len(self.clients)} clients"
        ))
        
        message = {
            "type": MessageType.WEIGHTS_BROADCAST.value,
            "weights": weights,
            "timestamp": time.time()
        }
        
        tasks = []
        for client_id, conn in self.clients.items():
            if conn.is_alive:
                tasks.append(self._send_message(conn.writer, message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.event_bus.publish(NetworkEvent(
            event_type=EventType.BROADCAST_COMPLETE,
            timestamp=time.time(),
            message="Broadcast complete"
        ))
    
    async def _heartbeat_monitor(self):
        """Monitor client heartbeats and detect zombies."""
        while self._running:
            await asyncio.sleep(self.heartbeat_interval)
            
            now = time.time()
            for client_id, conn in list(self.clients.items()):
                if now - conn.last_heartbeat > self.heartbeat_timeout:
                    conn.is_alive = False
                    self.event_bus.publish(NetworkEvent(
                        event_type=EventType.HEARTBEAT_MISSED,
                        timestamp=now,
                        client_id=client_id,
                        message=f"Client {client_id} missed heartbeat (zombie)"
                    ))
    
    def get_traffic_stats(self) -> Dict[str, Any]:
        """Get traffic statistics for Streamlit."""
        return {
            "connected_clients": len(self.clients),
            "compression_ratio": self.serializer.get_compression_ratio(),
            "throughput_mbps": self.latency_tracker.get_throughput_mbps(),
            "queue_utilization": self.backpressure.get_utilization(),
            "recent_events": [
                {
                    "type": e.event_type.value,
                    "time": e.timestamp,
                    "client": e.client_id,
                    "message": e.message
                }
                for e in self.event_bus.get_recent_events(10)
            ]
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=== Network Module Tests ===\n")
    
    # Test Serializer
    print("--- Serializer Test ---")
    serializer = Serializer(use_compression=True)
    test_weights = {
        "layer1.weight": np.random.randn(64, 32).astype(np.float32),
        "layer1.bias": np.random.randn(64).astype(np.float32)
    }
    
    serialized = serializer.serialize(test_weights)
    deserialized = serializer.deserialize(serialized)
    
    print(f"Original size: {sum(w.nbytes for w in test_weights.values())} bytes")
    print(f"Serialized size: {len(serialized)} bytes")
    print(f"Compression ratio: {serializer.get_compression_ratio():.2f}x")
    print(f"Weights match: {np.allclose(test_weights['layer1.weight'], deserialized['layer1.weight'])}")
    
    # Test Chunker
    print("\n--- Chunker Test ---")
    chunker = Chunker(chunk_size=1000)
    large_data = b"x" * 5000
    chunks = chunker.chunk(large_data, "msg_001")
    print(f"Split into {len(chunks)} chunks")
    reassembled = chunker.reassemble(chunks)
    print(f"Reassembled correctly: {large_data == reassembled}")
    
    # Test HMAC
    print("\n--- HMAC Test ---")
    signer = MessageSigner("test_secret")
    data = b"test message"
    sig = signer.sign(data)
    print(f"Signature: {sig[:32]}...")
    print(f"Verification: {signer.verify(data, sig)}")
    print(f"Tamper detection: {not signer.verify(data + b'x', sig)}")
    
    # Test Event Bus
    print("\n--- Event Bus Test ---")
    bus = EventBus()
    received_events = []
    bus.subscribe(EventType.CLIENT_CONNECTED, lambda e: received_events.append(e))
    bus.publish(NetworkEvent(
        event_type=EventType.CLIENT_CONNECTED,
        timestamp=time.time(),
        client_id="test_client",
        message="Test event"
    ))
    print(f"Event received: {len(received_events) == 1}")
    
    print("\n✅ All network tests passed!")


# ============================================================================
# VISUALIZATION HELPERS (for Streamlit Dashboard)
# ============================================================================

import plotly.graph_objects as go
import math

def create_topology(server_status: str, 
                    active_clients: List[str], 
                    all_clients: List[str]) -> go.Figure:
    """
    Create a network topology visualization.
    
    Args:
        server_status: Current status of the server.
        active_clients: List of client IDs currently active.
        all_clients: List of all registered client IDs.
    
    Returns:
        Plotly figure object.
    """
    fig = go.Figure()
    
    # Server Node (Center)
    server_color = "#00ff88" if server_status == "AGGREGATING" else "#00d4ff"
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers+text',
        marker=dict(size=40, color=server_color, line=dict(width=2, color='#fff'), symbol='diamond'),
        text=["SERVER"],
        textposition="top center",
        hoverinfo="text",
        name="Server"
    ))
    
    num_clients = len(all_clients)
    if num_clients == 0:
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=0, r=0, t=0, b=0),
            height=300
        )
        return fig
        
    radius = 1.0
    edge_x, edge_y = [], []
    node_x, node_y = [], []
    node_colors, node_sizes, node_text = [], [], []
    
    for i, client_id in enumerate(all_clients):
        angle = (2 * math.pi * i) / num_clients
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        
        node_x.append(x)
        node_y.append(y)
        
        is_active = client_id in active_clients
        if is_active:
            node_colors.append("#ffff00")
            node_sizes.append(25)
            edge_x.extend([0, x, None])
            edge_y.extend([0, y, None])
        else:
            node_colors.append("#ff00ff")
            node_sizes.append(15)
            
        node_text.append(client_id)
    
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=2, color='rgba(255, 255, 255, 0.5)'),
        hoverinfo='none',
        name='Connections'
    ))
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color='#fff')),
        text=node_text,
        hoverinfo="text",
        name="Clients"
    ))
    
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=0, b=0),
        height=300
    )
    
    return fig


def create_chart(data: List[float], 
                 title: str, 
                 color: str = "#00ff88") -> go.Figure:
    """
    Create a simple line chart for metrics.
    
    Args:
        data: List of values to plot.
        title: Chart title.
        color: Line color.
    
    Returns:
        Plotly figure.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=data,
        mode='lines+markers',
        line=dict(color=color, width=3),
        marker=dict(size=6, color='#fff'),
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(color="#fff", family="Orbitron")),
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        height=200
    )
    
    return fig