import time
import traceback
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
hot_words = []
weights = []

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

BAD_WORDS = ["fuck"]

START_TIMESTAMP = "start_timestamp"

def split_array(arr, chunk_size):
    return [arr[i:i + chunk_size] for i in range(0, len(arr), chunk_size)]


class AudioBuffer:
    def __init__(self, **kwargs):
        self.buffer = [np.zeros(300*16)]
        self.sample_rate = kwargs.get("sample_rate", 16000)
        self.audio_exporter_kwargs = kwargs.get("audio_exporter_kwargs", {})
        self.curr_idx = 1

    def get_data(self):
        prev_idx = max(0, self.curr_idx - 1)
        next_idx = self.curr_idx + 1
        audio_list = self.buffer[prev_idx: next_idx + 1]
        timestamps = [0]
        for _audio in audio_list:
            timestamps.append(timestamps[-1] + len(_audio))
        audio_data = np.concatenate(audio_list)
        return audio_data, timestamps


    def add_chunk(self, chunks):
        if isinstance(chunks, list):
            self.buffer.extend(chunks)
        else:
            self.buffer.append(chunks)

input_audio, sr = librosa.load("./../big_16.wav", sr=16000)
input_audio = split_array(input_audio, 16*300)
audio_buffer = AudioBuffer()
audio_buffer.add_chunk(chunks=input_audio)

def generate_sine_wave(duration):
    duration = duration/1000
    t = np.linspace(0, duration, int(16000*duration), endpoint=False)
    sine_wave = np.sin(2 * np.pi * 600 * t)
    int_wave = (sine_wave * 32767).astype(np.int16)
    return int_wave


def beep_audio(audio_data, x, y):
    duration = (y-x)/1000
    t = np.linspace(0, duration, int(16000*duration), endpoint=False)
    sine_wave = np.sin(2 * np.pi * 600 * t)
    int_wave = (sine_wave * 32767).astype(np.int16)
    audio = np.concatenate([audio_data[:int(x*16)], int_wave, audio_data[int(y*16):]])
    return audio


FINAL_ANSWER = []

def run_pipeline():
    while True:
        if audio_buffer.curr_idx > len(audio_buffer.buffer)-1:
            break
        start_time = round(time.time() * 1000)

        audio_data, timestamps = audio_buffer.get_data()
        try:
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

            print("Output Final:", output_final)
            for bad_word in BAD_WORDS:
                for _asr_word in output_final:
                    if bad_word.lower() in _asr_word['CONTENT'].lower():
                        audio_data = beep_audio(audio_data, _asr_word['START_TIME'], _asr_word['END_TIME'])
                        # print("Beeped", _asr_word, bad_word, timestamps[1], timestamps[2])
        except Exception as e:
            print(
                f"Error in run pipeline: {str(e)}\n{str(traceback.format_exc())}"
            )
        # print(timestamps)
        audio_buffer.curr_idx += 1
        # print(timestamps[1], timestamps[2])
        audio_data_middle = audio_data[timestamps[1]:timestamps[2]]
        pipeline_execution_time = round(time.time() * 1000) - start_time
        print(
            f"Pipeline execution time: {pipeline_execution_time} ms"
        )
        FINAL_ANSWER.append(audio_data_middle)


run_pipeline()
final_audio = np.concatenate(FINAL_ANSWER)
import soundfile as sf

sf.write("./out.wav", final_audio, 16000)
print("Done!")