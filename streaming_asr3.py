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
import numpy as np
import wave
import io
import soundfile as sf
import wave
import pyaudio
import librosa
import numpy as np

import os

from pydub.generators import Sine

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from pyctcdecode import build_ctcdecoder
from spr_ctc_decoder import Scorer
from spr_ctc_decoder import pyctc_beam_search_decoder

import soundfile as sf
from io import BytesIO


def get_audio_duration_from_bytes(audio_bytes):
    # Create a BytesIO object from the audio bytes
    audio_stream = BytesIO(audio_bytes)

    # Open the audio file as a SoundFile object
    with sf.SoundFile(audio_stream) as sound_file:
        # Get the number of samples and the sample rate
        sample_count = sound_file.frames
        sample_rate = sound_file.samplerate

        # Calculate the duration in seconds
        duration_seconds = sample_count / float(sample_rate)

        return duration_seconds


def log_softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """Logarithm of softmax function, following implementation of scipy.special."""
    x_max = np.amax(x, axis=axis, keepdims=True)
    if x_max.ndim > 0:
        x_max[~np.isfinite(x_max)] = 0
    elif not np.isfinite(x_max):
        x_max = 0  # pylint: disable=R0204
    tmp = x - x_max
    exp_tmp = np.exp(tmp)
    # suppress warnings about log of zero
    with np.errstate(divide="ignore"):
        s = np.sum(exp_tmp, axis=axis, keepdims=True)
        out: np.ndarray = np.log(s)
    out = tmp - out
    return out


def _add_timestamps(text, frames):
    timestamps = []
    words = text.split()
    for idx, (start_frame, end_frame) in enumerate(frames):
        word = words[idx] if idx < len(words) else ""
        timestamps.append(
            {
                "CONTENT": word,
                "START_TIME": round(start_frame * 20, 2),
                "END_TIME": round(end_frame * 20, 2),
            }
        )
    return timestamps


def load_vocab(file_path, encoding="UTF-8"):
    lines = []
    with open(file_path, "r", encoding=encoding) as file:
        for line in file:
            lines.append(line.strip("\n"))
    return lines


device = torch.device('cuda')

processor = Wav2Vec2Processor.from_pretrained(
    "/DATA2/Abhinav/Generalised_English_Retail/models/production/v0/wav2vec2_contextual/wav2vec2_processor/")
model = Wav2Vec2ForCTC.from_pretrained("/DATA2/Abhinav/Generalised_English_Retail/models/deployed/v0/").to(device)

# shipt
hot_words = ["fuck", "fucking", "stupid", "idiot", "bloody"]
weights = [(2, (False, True)), (2, (False, True)), (1, (False, True)), (1, (False, True)), (1, (False, True))]
# hot_words = ["adobe"]
# weights = [(5, (False, True))]

lm_path = "/DATA2/Abhinav/Generalised_English_Retail/models/production/v0/spr_ctc_decoder/lm.arpa"

scorer_alpha = 0.5
scorer_beta = 1.5
score_lm_boundary = True
unk_logp_offset = -10.0
avg_token_len = 6

beam_size = 512
prune_logp = -10.0
min_token_logp = -10.0
prune_history = False
vocab = load_vocab("/DATA2/Abhinav/Generalised_English_Retail/models/production/v0/spr_ctc_decoder/vocab.txt")

# hot_words = {}  # Assuming hot words are empty based on the log

space_id = 0
blank_id = 30
frequent_word_prefix_token = 2
return_timestamp = True

scorer = Scorer(
    scorer_alpha,  # alpha
    scorer_beta,  # beta
    lm_path,  # lm arpa file path
    vocab,  # vocab
    score_lm_boundary,  # score boundary true/false
    unk_logp_offset,  # unknown score offset value
    space_id,
    avg_token_len,  # Avg Token Length
)


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


BAD_WORDS = ["fucking", "fuck", "stupid", "idiot", "bloody", "adobe"]

START_TIMESTAMP = "start_timestamp"


def split_array(arr, chunk_size):
    return [arr[i:i + chunk_size] for i in range(0, len(arr), chunk_size)]


def beep_audio(audio_data, x, y):
    duration = (y - x) / 1000
    t = np.linspace(0, duration, int(16000 * duration), endpoint=False)
    sine_wave = np.sin(2 * np.pi * 600 * t)
    int_wave = (sine_wave * 32767).astype(np.int16)
    audio = np.concatenate([audio_data[:int(x * 16)], int_wave, audio_data[int(y * 16):]])
    return audio


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
        self.sample_rate = kwargs.get("sample_rate", 16000)
        self.audio_exporter_kwargs = kwargs.get("audio_exporter_kwargs", {})
        self.curr_idx = 0

    def add_chunk(self, chunk):
        """
        Adds a chunk of audio data to the buffer.
        The chunk is not added if the buffer is already full.
        Updates the duration and increments the chunk counter.
        """
        # if self.is_full():
        #     # TODO: Empty the buffer from starting
        #     return
        if isinstance(chunk, bytes):
            self.buffer.append(chunk)
        else:
            self.buffer.extend(chunk)
        self.duration_ms += (len(chunk) / self.sample_rate) * 1000
        # print(self.duration_ms)

    def get_data(self):
        """
        Returns the audio data in the buffer as a byte string.
        It assumes the buffer is a List[int16] array of samples
        """
        prev_idx = max(0, self.curr_idx - 6)
        next_idx = self.curr_idx + 4
        audio_list = self.buffer[prev_idx: next_idx + 1]
        audio_data = audio_list
        if isinstance(audio_list[0], bytes):
            audio_data = b"".join(audio_list)

        # print(len(audio_data))
        # print(get_audio_duration_from_bytes(audio_list[0]))
        audio_path = os.path.join("/tmp/", f"{self.audio_buffer_id}.wav")
        _ = export_audio_using_pywav(
            audio_data, audio_path, **self.audio_exporter_kwargs
        )
        audio, sample_rate = sf.read(audio_path)
        # print(len(audio) / 16000)
        timestamps = [0, 9600, 14400, 16000]
        # print(audio)
        # for _audio in audio_list:
        #     timestamps.append(timestamps[-1] + get_audio_duration_from_bytes(_audio))
        # print(timestamps)
        return audio, timestamps

    def is_full(self):
        return self.get_duration() >= self.max_duration_ms

    def clear(self):
        self.buffer = []
        self.duration_ms = 0

    def get_duration(self):
        return self.duration_ms

    def get_number_of_chunks(self):
        return self.duration_ms / self.chunk_ms_size

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
        try:
            if not self.session_info.get("is_pipeline_running", False) and self.ws_connection:
                self.ping()
        except (websockets.ConnectionClosed, tornado.websocket.WebSocketClosedError):
            print(f"Client connection closed")
            self._close()
        except Exception as e:
            print(f"Ping exception: {e}")
            self._close()

    async def open(self):
        print("WebSocket opened")
        self.audio_buffer = AudioBuffer(
            max_duration_ms=120000,
            sample_rate=16000,
            audio_exporter_kwargs={
                "channels": 1,
                "samplerate": 16000,
                "bitspersample": 16,
                "audioformat": 7,
            },
        )
        self.session_info = {START_TIMESTAMP: round(time.time() * 1000)}
        self.telemetry = {START_TIMESTAMP: round(time.time() * 1000)}
        self.timeout_checker = tornado.ioloop.PeriodicCallback(self.check_timeout, 300)
        self.timeout_checker.start()

    def on_message(self, audio_chunk):
        tornado.ioloop.IOLoop.current().add_callback(self.process_message, audio_chunk)

    async def process_message(self, audio_chunk):
        try:
            await self._process_message(audio_chunk)
        except Exception as e:
            print(
                f"Exception while processing the message: {str(e)}\n{str(traceback.format_exc())}"
            )
            self._close()

    async def _run_pipeline(self):
        global processor, model
        try:
            self.session_info["is_pipeline_running"] = True

            start_time = round(time.time() * 1000)
            audio_data, timestamps = self.audio_buffer.get_data()
            # print(audio_data)

            features = processor(
                [audio_data],
                sampling_rate=16000,
                padding=True,
                return_tensors="pt",
            )
            input_values = features.input_values.to(device)
            attention_mask = features.attention_mask.to(device)

            with torch.no_grad():
                output = model(
                    input_values.float(), attention_mask=attention_mask
                ).logits

            output = output.cpu().detach().numpy()

            prob = np.clip(
                log_softmax(np.array(output[0]), axis=1),
                np.log(1e-15),
                0,
            ).astype(np.double)

            text = pyctc_beam_search_decoder(
                prob.tolist(),  # logits
                beam_size,  # Beam Width
                prune_logp,  # Beam prune logp
                min_token_logp,  # Min Token logp
                prune_history,  # Prune History
                vocab,  # Vocab List
                hot_words,  # List of hot words
                weights,  # List of weights
                scorer,  # LM Scorer
                space_id,  # Space Id
                blank_id,  # Blank Id
                frequent_word_prefix_token,  # Frequent Word Prefix Token @
                return_timestamp,
            )

            output_final = _add_timestamps(text[0].replace("@", ""), text[1])
            # words = []
            # for _asr_word in output_final:

            # for _asr_word in output_final:
            #     x = _asr_word['START_TIME']*16
            #     y = _asr_word['END_TIME']*16
            #     t1 = timestamps[1]
            #     t2 = timestamps[2]
            #     # print(x, y, timestamps[1], timestamps[2])
            #     if y <= t2 and y > t1:
            #         words.append(_asr_word['CONTENT'])

            # print(" ".join(words), end=" ", flush=True)
            # print("Output Final:", output_final)
            for _asr_word in output_final:
                for bad_word in BAD_WORDS:
                    if bad_word.lower() in _asr_word['CONTENT'].lower():
                        audio_data = beep_audio(audio_data, _asr_word['START_TIME'], _asr_word['END_TIME'])
                        break
                        # print("Beeped", _asr_word, bad_word, timestamps[1], timestamps[2])
            # print(timestamps)
            self.audio_buffer.curr_idx += 3
            # print(timestamps[1], timestamps[2])
            audio_data_middle = audio_data[timestamps[1]:timestamps[2]] * 32767.0
            audio_data_middle = audio_data_middle.astype(np.int16)
            # print(audio_data_middle)
            # print(audio_data_middle)
            pipeline_execution_time = round(time.time() * 1000) - start_time
            # print(
            #     f"Pipeline execution time: {pipeline_execution_time} ms"
            # )
            await self.write_message(audio_data_middle.tobytes(), binary=True)
        except Exception as e:
            print(
                f"Error in run pipeline: {str(e)}\n{str(traceback.format_exc())}"
            )
        finally:
            self.session_info["is_pipeline_running"] = False

    async def _process_message(self, audio_chunk):
        if "first_media_message_received" not in self.session_info:
            self.session_info["first_media_message_received"] = True
            self.telemetry["first_media_message_received"] = (
                    round(time.time() * 1000) - self.telemetry[START_TIMESTAMP]
            )
        self.audio_buffer.add_chunk(audio_chunk)
        if self._is_ready():
            await self._run_pipeline()

    def _is_ready(self):
        return (
                len(self.audio_buffer.buffer) > self.audio_buffer.curr_idx + 3
                and not self.session_info.get("is_pipeline_running", False)
        )

    def on_close(self):
        print("WebSocket closed")

    def check_origin(self, origin):
        return True

    def _close(self):
        # self.write_message(self.telemetry)
        self.close()


def make_app():
    return tornado.web.Application([
        (r"/asr/predict", ASRWebSocketHandler),
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(8558)
    print("WebSocket server started on port 8558")
    tornado.ioloop.IOLoop.current().start()