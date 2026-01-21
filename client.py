import asyncio
import websockets
import json
import numpy as np
import sounddevice as sd
import threading
import queue
import sys

# Audio configuration (matches server: 24kHz, 16-bit PCM)
SAMPLE_RATE = 24000
CHANNELS = 1


class AudioPlayer:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._play_thread, daemon=True)
        self.thread.start()

    def _play_thread(self):
        """Thread to play audio from the queue to avoid blocking networking"""
        with sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16"
        ) as stream:
            while not self.stop_event.is_set():
                try:
                    data = self.audio_queue.get(timeout=0.1)
                    stream.write(data)
                except queue.Empty:
                    continue

    def add_audio(self, data):
        self.audio_queue.put(np.frombuffer(data, dtype=np.int16))


async def listen_to_server(ws, player):
    """Wait for audio chunks or status messages from the server"""
    try:
        async for message in ws:
            if isinstance(message, bytes):
                # It's an audio chunk
                player.add_audio(message)
            else:
                # It's a status message (JSON)
                try:
                    status = json.loads(message)
                    if status.get("status") == "completed":
                        print(
                            f"\n[Server] Synthesis completed in {status.get('time')} ({status.get('chunks')} chunks received)"
                        )
                        print("Enter text: ", end="", flush=True)
                    elif "error" in status:
                        print(f"\n[Error] {status['error']}")
                except json.JSONDecodeError:
                    print(f"\n[Server] {message}")
    except websockets.exceptions.ConnectionClosed:
        print("\n[Client] Connection closed by server.")


async def send_text(ws):
    """Loop to get user input and send to server"""
    while True:
        # aioconsole.ainput would be better but keeping it simple with standard loop
        text = await asyncio.get_event_loop().run_in_executor(
            None, input, "Enter text: "
        )
        if text.strip().lower() in ["exit", "quit"]:
            break
        if text.strip():
            await ws.send(text)


async def main():
    uri = "ws://localhost:8000/ws/tts"
    player = AudioPlayer()

    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected! Type your text and press Enter (type 'exit' to quit).")

            # Run listener and sender concurrently
            await asyncio.gather(
                listen_to_server(websocket, player), send_text(websocket)
            )
    except ConnectionRefusedError:
        print("Failed to connect. Is the server running at localhost:8000?")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Ensure sounddevice and websockets are installed:
    # pip install sounddevice websockets numpy
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
