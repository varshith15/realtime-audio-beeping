import os
import time
import logging
import json
from bson.objectid import ObjectId
import asyncio
import base64
import tornado.ioloop
import tornado.web
import tornado.websocket
import websockets
import traceback


START_TIMESTAMP = "start_timestamp"

def export_audio_using_pywav(data, path, **kwargs):
    import pywav

    if not isinstance(data, bytes):
        raise Exception(f"Expected bytes, instead got {str(type(data))}")
    logging.error("kwargs: {}".format(kwargs))
    wave_write = pywav.WavWrite(path, **kwargs)
    wave_write.write(data)
    wave_write.close()
    return path


class AudioBuffer:
    """
    Class representing an audio buffer for managing chunks of audio data.

    Attributes:
    - max_size (int): Maximum number of chunks that can be stored in the buffer.
    - duration_ms (float): Total duration of the audio data in the buffer, in milliseconds.
    - num_chunks (int): Number of chunks currently in the buffer.
    - buffer (list): The audio data stored in the buffer.
    """

    def __init__(self, max_duration_ms, **kwargs):
        self.audio_buffer_id = str(ObjectId())
        self.buffer = []
        self.duration_ms = 0
        self.max_duration_ms = max_duration_ms
        self.sample_rate = kwargs.get("sample_rate", 8000)
        self.audio_exporter_kwargs = kwargs.get("audio_exporter_kwargs", {})

    def add_chunk(self, chunk):
        """
        Adds a chunk of audio data to the buffer.
        The chunk is not added if the buffer is already full.
        Updates the duration and increments the chunk counter.
        """
        if self.is_full():
            return
        if isinstance(chunk, bytes):
            self.buffer.append(chunk)
        else:
            self.buffer.extend(chunk)
        self.duration_ms += (len(chunk) / self.sample_rate) * 1000

    def get_data(self):
        """
        Returns the audio data in the buffer as a byte string.
        It assumes the buffer is a List[int16] array of samples
        """
        audio_data = self.buffer
        if isinstance(self.buffer[0], bytes):
            audio_data = b"".join(self.buffer)
        audio_path = os.path.join("/tmp/", f"{self.audio_buffer_id}.wav")
        _ = export_audio_using_pywav(
            audio_data, audio_path, **self.audio_exporter_kwargs
        )
        return open(audio_path, "rb").read()

    def is_full(self):
        return self.get_duration() >= self.max_duration_ms

    def clear(self):
        self.buffer = []
        self.duration_ms = 0

    def get_duration(self):
        return self.duration_ms

    def __del__(self):
        audio_path = os.path.join("/tmp/", f"{self.audio_buffer_id}.wav")
        if os.path.exists(audio_path):
            os.remove(audio_path)


class ASRWebSocketHandler(tornado.websocket.WebSocketHandler):
    def __init__(self, application, request, **kwargs):
        super().__init__(application, request, **kwargs)
        self.timeout_checker = None
        self.telemetry = None
        self.session_info = None
        self.audio_buffer = None

    def check_timeout(self):
        # if time.time() * 1000 - self.session_info[START_TIMESTAMP] >= 15000:
        #     logging.error("Idle timeout exceeded, closing connection")
        #     self._close()
        #     return
        try:
            if not self.session_info.get("is_pipeline_running", False) and self.ws_connection:
                self.ping()
        except (websockets.ConnectionClosed, tornado.websocket.WebSocketClosedError):
            logging.error(f"Client connection closed")
            self._close()
        except Exception as e:
            logging.error(f"Ping exception: {e}")
            self._close()

    async def open(self):
        print("WebSocket opened")
        self.audio_buffer = AudioBuffer(
            max_duration_ms=5000,
            sample_rate=8000,
            audio_exporter_kwargs={
                "channels": 1,
                "samplerate": 8000,
                "bitspersample": 8,
                "audioformat": 7,
            },
        )
        # self._initialize_audio_buffer()
        self.session_info = {START_TIMESTAMP: round(time.time() * 1000)}
        self.telemetry = {START_TIMESTAMP: round(time.time() * 1000)}
        self.timeout_checker = tornado.ioloop.PeriodicCallback(self.check_timeout, 300)
        self.timeout_checker.start()

    def on_message(self, audio_chunk):
        # print(f"Received message: {audio_chunk}")
        # Add a callback to process the message
        tornado.ioloop.IOLoop.current().add_callback(self.process_message, audio_chunk)

    async def process_message(self, audio_chunk):
        try:
            await self._process_message(audio_chunk)
        except Exception as e:
            logging.error(
                f"Exception while processing the message: {str(e)}\n{str(traceback.format_exc())}"
            )
            self._close()

    async def _run_pipeline(self):
        try:
            self.session_info["is_pipeline_running"] = True

            start_time = round(time.time() * 1000)
            duration = self.audio_buffer.get_duration()
            # audio_data = self.audio_buffer.get_data()
            # TODO: process audio data
            # response = await audio_data
            print("Audio Chunks got:", duration)
            await asyncio.sleep(0.05)
            pipeline_execution_time = round(time.time() * 1000) - start_time
            logging.info(
                f"Audio Buffer [{duration}]: Pipeline execution time: {pipeline_execution_time} ms"
            )
            if "pipeline_runs" not in self.telemetry:
                self.telemetry["pipeline_runs"] = []
            # self.telemetry["pipeline_runs"].append(
            #     {
            #         "transcript": response["LM"],
            #     }
            # )
            await self.write_message(self.telemetry)
        except Exception as e:
            logging.error(
                f"Error in run pipeline: {str(e)}\n{str(traceback.format_exc())}"
            )
        finally:
            self.session_info["is_pipeline_running"] = False

    async def _process_message(self, audio_chunk):
        # message = json.loads(message)
        if "first_media_message_received" not in self.session_info:
            self.session_info["first_media_message_received"] = True
            self.telemetry["first_media_message_received"] = (
                round(time.time() * 1000) - self.telemetry[START_TIMESTAMP]
            )
        # audio_chunk = base64.b64decode(message)
        self.audio_buffer.add_chunk(audio_chunk)
        if (
            not self.session_info.get("is_pipeline_running", False)
            and self.audio_buffer.is_full()
        ):
            logging.error("Audio buffer is full. Running the pipeline")
            await self._run_pipeline()
            # self._close()
        elif self._is_ready():
            await self._run_pipeline()

    def _is_ready(self):
        return (
            self.audio_buffer.get_duration()
            and self.audio_buffer.get_duration() >= 300
            and not self.session_info.get("is_pipeline_running", False)
        )

    def on_close(self):
        print("WebSocket closed")

    def check_origin(self, origin):
        # Override to allow connections from any origin
        return True

    def _close(self):
        try:
            self.write_message(self.telemetry)
        except Exception as e:
            logging.error(
                f"Exception in sending message to client: {str(e)}\nTraceback: {str(traceback.format_exc())}"
            )
        self.close()


def make_app():
    return tornado.web.Application([
        (r"/asr/predict", ASRWebSocketHandler),
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(8889)
    print("WebSocket server started on port 8889")
    tornado.ioloop.IOLoop.current().start()