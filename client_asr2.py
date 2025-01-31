import websockets
import asyncio
import wave
import pyaudio
import os


full_audio = []
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                output=True,
                frames_per_buffer=6400)


def export_audio_using_pywav(data, path, **kwargs):
    audio = pyaudio.PyAudio()

    # Save audio file
    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(data)
    wf.close()
    return path


class StreamingASR:
    def __init__(self):
        self.uri = "ws://bangalore-wired-mtqpbgjjtk.dynamic-m.com:8558/asr/predict"
        # self.uri = "ws://localhost:8558/asr/predict"
        self.websocket = None

    async def recv_data(self, reader, writer):
        if self.websocket == None:
            self.websocket = await websockets.connect(self.uri)
        while True:
            audio_chunk = await reader.read(6400)
            # stream.write(audio_chunk)
            # print(audio_chunk)
            await self.websocket.send(audio_chunk)
            # await asyncio.sleep(0.001)

    async def main(self):
        server = await asyncio.start_server(self.recv_data, '192.168.242.41', 8888)
        recv_data = asyncio.create_task(server.serve_forever())
        receive_task = asyncio.create_task(self.receive_responses())
        await asyncio.gather(recv_data, receive_task)

    async def receive_responses(self):
        if self.websocket == None:
            self.websocket = await websockets.connect(self.uri)
        try:
            async for message in self.websocket:
                stream.write(message)
                # full_audio.append(message)
                # audio_data = b"".join(full_audio)
                # audio_path = os.path.join("/tmp/", f"tmp.wav")
                # _ = export_audio_using_pywav(audio_data, audio_path)
                # print(audio_path)
                # print(f"Received from server: {message}")
        except websockets.ConnectionClosed as e:
            print(f"WebSocket connection closed: {e}")
            self.websocket = None  # Reset the connection


receiver = StreamingASR()
asyncio.run(receiver.main())