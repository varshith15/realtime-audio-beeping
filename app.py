import numpy as np
import io
import wave
import time
from flask import Flask, request
from flask_socketio import SocketIO
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
dg_connection = None
answer = []


def on_message(self, result, **kwargs):
    global answer
    print("2")

    words = result.channel.alternatives[0].words
    print(words)
    answer.append((words, time.time()))
    print(answer)


def on_metadata(self, metadata, **kwargs):
    print(f"\n\n{metadata}\n\n")


def on_error(self, error, **kwargs):
    print(f"\n\n{error}\n\n")


def initialize():
    global dg_connection

    DEEPGRAM_API_KEY = '744999acddd69dad56913c0915b6cbbb4ffd38f2'

    # example of setting up a client config. logging values: WARNING, VERBOSE, DEBUG, SPAM
    config: DeepgramClientOptions = DeepgramClientOptions(
        # verbose=logging.DEBUG,
        options={"keepalive": "false"}
    )
    deepgram: DeepgramClient = DeepgramClient(DEEPGRAM_API_KEY, config)
    dg_connection = deepgram.listen.live.v("1")

    dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
    dg_connection.on(LiveTranscriptionEvents.Metadata, on_metadata)
    dg_connection.on(LiveTranscriptionEvents.Error, on_error)

    # connect to websocket
    options: LiveOptions = LiveOptions(
        model="nova-2",
        language="en-US",
        interim_results=True,
    )

    dg_connection.start(options)


@app.route('/process_audio_chunk', methods=['POST'])
def process_audio_chunk():
    print("1")
    global dg_connection

    if dg_connection == None:
        initialize()
    # Simulate audio processing (replace with actual processing logic or API call)
    audio_chunk = request.files['file'].read()
    dg_connection.send(audio_chunk)
    # processed_audio_chunk = audio_chunk  # Placeholder for actual processing
    #
    # # Emit the processed audio chunk to all connected clients
    # socketio.emit('audio_chunk', {'data': processed_audio_chunk})
    return 'OK', 200


if __name__ == '__main__':
    # global dg_connection
    socketio.run(app, port=10000)