import asyncio
import websockets
import json
import sys
import wave
import time

# Audio configuration
SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2  # int16


class AudioRecorder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.buffers = []
        self.send_time = None
        self.first_audio_time = None

    def mark_send(self):
        self.send_time = time.perf_counter()
        self.first_audio_time = None

    def add_audio(self, data: bytes):
        if self.first_audio_time is None and self.send_time is not None:
            self.first_audio_time = time.perf_counter()
            ttfb = (self.first_audio_time - self.send_time) * 1000
            print(f"[TTFB] {ttfb:.2f} ms (first audio byte received)")

        self.buffers.append(data)

    def save_wav(self, filename: str):
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(self.buffers))


async def listen_to_server(ws, recorder: AudioRecorder):
    try:
        async for message in ws:
            if isinstance(message, bytes):
                recorder.add_audio(message)

            else:
                try:
                    status = json.loads(message)

                    if status.get("status") == "completed":
                        filename = f"tts_{int(time.time())}.wav"
                        recorder.save_wav(filename)

                        total_time = (
                            time.perf_counter() - recorder.send_time
                            if recorder.send_time
                            else None
                        )

                        if total_time:
                            print(f"[Total] {total_time*1000:.2f} ms")

                        recorder.reset()
                        print("Enter text: ", end="", flush=True)

                    elif "error" in status:
                        print(f"[Server Error] {status['error']}")

                except json.JSONDecodeError:
                    print(f"[Server] {message}")

    except websockets.exceptions.ConnectionClosed:
        print("[Client] Connection closed.")


async def send_text(ws, recorder: AudioRecorder):
    while True:
        text = await asyncio.get_event_loop().run_in_executor(
            None, input, "Enter text: "
        )

        if text.strip().lower() in ("exit", "quit"):
            break

        if text.strip():
            recorder.mark_send()
            await ws.send(text)


async def main():
    uri = "ws://localhost:8000/ws/tts"
    recorder = AudioRecorder()

    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected. Type text to synthesize.")

            await asyncio.gather(
                listen_to_server(websocket, recorder),
                send_text(websocket, recorder),
            )

    except ConnectionRefusedError:
        print("Server not running.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
