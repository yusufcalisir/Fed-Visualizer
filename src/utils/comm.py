"""
Async Communication Layer for Federated Learning Server
========================================================
Implements WebSocket-based communication for low-latency weight transmission.
"""

import asyncio
import pickle
import struct
from typing import Dict, Any, Optional, Callable, Set
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger("FedComm")


@dataclass
class ClientConnection:
    """Represents a connected client."""
    client_id: str
    writer: asyncio.StreamWriter
    reader: asyncio.StreamReader
    connected_at: float
    last_activity: float


class FederatedCommServer:
    """
    Async TCP Server for Federated Learning communication.
    
    Uses a simple length-prefixed protocol:
    - 4 bytes: message length (big-endian)
    - N bytes: pickled message data
    """
    
    def __init__(self, 
                 host: str = "0.0.0.0",
                 port: int = 8765,
                 timeout: float = 30.0,
                 on_client_update: Optional[Callable] = None):
        """
        Initialize the communication server.
        
        Args:
            host: Bind address.
            port: Port number.
            timeout: Client timeout in seconds.
            on_client_update: Callback when client sends updates.
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.on_client_update = on_client_update
        
        self.clients: Dict[str, ClientConnection] = {}
        self.pending_updates: Dict[str, Dict[str, Any]] = {}
        self._server: Optional[asyncio.Server] = None
        self._running = False
        
        self.logs: list = []
    
    def _log(self, message: str):
        """Add a log entry."""
        entry = {"timestamp": time.strftime("%H:%M:%S"), "message": message}
        self.logs.append(entry)
        logger.info(message)
    
    async def start(self):
        """Start the async server."""
        self._server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port
        )
        self._running = True
        self._log(f"Server started on {self.host}:{self.port}")
        
        async with self._server:
            await self._server.serve_forever()
    
    async def stop(self):
        """Stop the server gracefully."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        self._log("Server stopped.")
    
    async def _handle_client(self, 
                              reader: asyncio.StreamReader, 
                              writer: asyncio.StreamWriter):
        """Handle a new client connection."""
        addr = writer.get_extra_info('peername')
        self._log(f"New connection from {addr}")
        
        client_id = None
        try:
            while self._running:
                # Read message length (4 bytes)
                length_data = await asyncio.wait_for(
                    reader.read(4), 
                    timeout=self.timeout
                )
                if not length_data:
                    break
                
                msg_length = struct.unpack(">I", length_data)[0]
                
                # Read message data
                data = await asyncio.wait_for(
                    reader.read(msg_length),
                    timeout=self.timeout
                )
                
                message = pickle.loads(data)
                
                # Process message
                response = await self._process_message(message, reader, writer)
                if message.get("type") == "register":
                    client_id = message.get("client_id")
                
                # Send response
                if response:
                    await self._send_message(writer, response)
                    
        except asyncio.TimeoutError:
            self._log(f"Client {client_id or addr} timed out.")
        except Exception as e:
            self._log(f"Error handling client {client_id or addr}: {e}")
        finally:
            if client_id and client_id in self.clients:
                del self.clients[client_id]
            writer.close()
            await writer.wait_closed()
            self._log(f"Client {client_id or addr} disconnected.")
    
    async def _process_message(self, 
                                message: Dict[str, Any],
                                reader: asyncio.StreamReader,
                                writer: asyncio.StreamWriter) -> Optional[Dict[str, Any]]:
        """Process an incoming message from a client."""
        msg_type = message.get("type")
        
        if msg_type == "register":
            client_id = message["client_id"]
            self.clients[client_id] = ClientConnection(
                client_id=client_id,
                writer=writer,
                reader=reader,
                connected_at=time.time(),
                last_activity=time.time()
            )
            self._log(f"Client '{client_id}' registered.")
            return {"type": "ack", "status": "registered"}
        
        elif msg_type == "update":
            client_id = message["client_id"]
            weights = message["weights"]
            self.pending_updates[client_id] = weights
            
            if client_id in self.clients:
                self.clients[client_id].last_activity = time.time()
            
            self._log(f"Received update from '{client_id}'.")
            
            if self.on_client_update:
                await self.on_client_update(client_id, weights)
            
            return {"type": "ack", "status": "received"}
        
        elif msg_type == "request_weights":
            # Client is requesting global weights
            return {"type": "weights_pending"}
        
        return None
    
    async def _send_message(self, 
                             writer: asyncio.StreamWriter, 
                             message: Dict[str, Any]):
        """Send a message to a client."""
        data = pickle.dumps(message)
        length = struct.pack(">I", len(data))
        writer.write(length + data)
        await writer.drain()
    
    async def broadcast_weights(self, weights: Dict[str, Any]):
        """Broadcast global weights to all connected clients."""
        self._log(f"Broadcasting weights to {len(self.clients)} clients...")
        
        message = {"type": "global_weights", "weights": weights}
        
        tasks = []
        for client_id, conn in self.clients.items():
            tasks.append(self._send_message(conn.writer, message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self._log("Broadcast complete.")
    
    def get_pending_updates(self) -> Dict[str, Dict[str, Any]]:
        """Get all pending client updates and clear the buffer."""
        updates = self.pending_updates.copy()
        self.pending_updates.clear()
        return updates
    
    def get_connected_clients(self) -> Set[str]:
        """Return set of currently connected client IDs."""
        return set(self.clients.keys())


def serialize_weights(weights: Dict[str, Any]) -> bytes:
    """Serialize model weights using pickle."""
    return pickle.dumps(weights)


def deserialize_weights(data: bytes) -> Dict[str, Any]:
    """Deserialize model weights from bytes."""
    return pickle.loads(data)


# Example client code for testing
async def example_client(client_id: str, host: str = "127.0.0.1", port: int = 8765):
    """Example client for testing the communication server."""
    import numpy as np
    
    reader, writer = await asyncio.open_connection(host, port)
    
    # Register
    register_msg = {"type": "register", "client_id": client_id}
    data = pickle.dumps(register_msg)
    writer.write(struct.pack(">I", len(data)) + data)
    await writer.drain()
    
    # Read ack
    length_data = await reader.read(4)
    msg_length = struct.unpack(">I", length_data)[0]
    response = pickle.loads(await reader.read(msg_length))
    print(f"[{client_id}] Registered: {response}")
    
    # Send update
    mock_weights = {
        "layer1.weight": np.random.randn(64, 32),
        "layer1.bias": np.random.randn(64)
    }
    update_msg = {"type": "update", "client_id": client_id, "weights": mock_weights}
    data = pickle.dumps(update_msg)
    writer.write(struct.pack(">I", len(data)) + data)
    await writer.drain()
    
    # Read ack
    length_data = await reader.read(4)
    msg_length = struct.unpack(">I", length_data)[0]
    response = pickle.loads(await reader.read(msg_length))
    print(f"[{client_id}] Update ack: {response}")
    
    writer.close()
    await writer.wait_closed()


if __name__ == "__main__":
    # Run server
    server = FederatedCommServer(port=8765)
    asyncio.run(server.start())
