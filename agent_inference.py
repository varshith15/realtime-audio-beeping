import gradio as gr
import requests
import wave
import numpy as np
import io


def audio_array_to_wav_bytes(audio_array, sample_rate):
    """
    Convert an audio numpy array to WAV file bytes.

    Parameters:
    audio_array (numpy.ndarray): The audio data as a numpy array.
    sample_rate (int): The sample rate of the audio data.

    Returns:
    bytes: The WAV file bytes.
    """
    # Ensure the array is of type int16
    audio_array = np.asarray(audio_array, dtype=np.int16)

    # Create an in-memory file-like object
    buffer = io.BytesIO()

    # Initialize the WAV file parameters
    num_channels = 1 if len(audio_array.shape) == 1 else audio_array.shape[1]
    sample_width = audio_array.dtype.itemsize
    num_frames = audio_array.shape[0]

    # Write the WAV file
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.setnframes(num_frames)
        wav_file.writeframes(audio_array.tobytes())

    # Get the bytes data from the buffer
    wav_bytes = buffer.getvalue()

    # Close the buffer
    buffer.close()

    return wav_bytes


# Define the function to send audio to the backend for processing
def send_audio_to_backend(audio):
    sample_rate, audio_array = audio
    response = requests.post(
        "http://localhost:10000/process_audio_chunk",
        files={"file": audio_array_to_wav_bytes(audio_array, sample_rate)}
    )
    if response.ok:
        # Return the processed audio file
        return response.content
    else:
        raise Exception(f"Failed to process audio chunk: {response.status_code}, {response.reason}")


# Create the Gradio interface
iface = gr.Interface(
    fn=send_audio_to_backend,
    inputs=gr.Audio(sources="microphone", type="numpy", streaming=True, format="wav"),
    outputs="audio",
    live=True
)

# Launch the Gradio app on port 8554
iface.launch(server_port=8554)