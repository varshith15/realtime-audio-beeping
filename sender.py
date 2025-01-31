import asyncio
import struct
import time

import pyaudio

class VoiceChatSender:
    def __init__(self):
        self.chunk = 1600
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunk)

    async def send_data(self, writer):
        while True:
            # timestamp = time.time()
            data = self.stream.read(self.chunk)
            writer.write(data)
            await writer.drain()

    async def main(self):
        reader, writer = await asyncio.open_connection('192.168.242.41', 8888)
        await self.send_data(writer)


sender = VoiceChatSender()
asyncio.run(sender.main())