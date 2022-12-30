import pyaudio
import webrtcvad
from live_inference import LiveInference
import numpy as np
import threading
import copy
import time
from sys import exit
import contextvars
from queue import  Queue


class LiveRecognition():
    exit_event = threading.Event()
    def __init__(self, model_name, device_name="default"):
        self.model_name = model_name
        self.device_name = device_name

    def stop(self):
        LiveRecognition.exit_event.set()
        self.recognition_input_queue.put("close")
        print("Stopping listening process")

    def start(self):
        self.recognition_output_queue = Queue()
        self.recognition_input_queue = Queue()
        self.recognition_process = threading.Thread(target=LiveRecognition.recognition_process, args=(
            self.model_name, self.recognition_input_queue, self.recognition_output_queue,))
        self.recognition_process.start()
        time.sleep(5)  # start vad after recognition model is loaded
        self.vad_process = threading.Thread(target=LiveRecognition.vad_process, args=(
            self.device_name, self.recognition_input_queue,))
        self.vad_process.start()

    def vad_process(device_name, recognition_input_queue):
        vad = webrtcvad.Vad()
        vad.set_mode(1)

        audio = pyaudio.PyAudio()
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        FRAME_DURATION = 30
        CHUNK = int(RATE * FRAME_DURATION / 1000)
        RECORD_SECONDS = 50

        microphones = LiveRecognition.list_microphones(audio)
        selected_input_device_id = LiveRecognition.get_input_device_id(
            device_name, microphones)

        stream = audio.open(input_device_index=selected_input_device_id,
                            format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

        frames = b''
        while True:
            if LiveRecognition.exit_event.is_set():
                break
            frame = stream.read(CHUNK, exception_on_overflow=False)
            is_speech = vad.is_speech(frame, RATE)
            if is_speech:
                frames += frame
            else:
                if len(frames) > 1:
                    recognition_input_queue.put(frames)
                frames = b''
        stream.stop_stream()
        stream.close()
        audio.terminate()

    def recognition_process(model_name, in_queue, output_queue):
        live_recognition = LiveInference(model_name, use_model=True)

        print("\nlistening to your voice\n")
        while True:
            audio_frames = in_queue.get()
            if audio_frames == "close":
                break

            float64_buffer = np.frombuffer(
                audio_frames, dtype=np.int16) / 32767
            start = time.perf_counter()
            text, confidence = live_recognition.buffer_to_text(float64_buffer)
            text = text.lower()
            inference_time = time.perf_counter()-start
            sample_length = len(float64_buffer) / 16000
            if text != "":
                output_queue.put([text,sample_length,inference_time,confidence])

    def get_input_device_id(device_name, microphones):
        for device in microphones:
            if device_name in device[1]:
                return device[0]

    def list_microphones(pyaudio_instance):
        info = pyaudio_instance.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')

        result = []
        for i in range(0, numdevices):
            if (pyaudio_instance.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                name = pyaudio_instance.get_device_info_by_host_api_device_index(
                    0, i).get('name')
                result += [[i, name]]
        return result

    def get_last_text(self):
        return self.recognition_output_queue.get()

if __name__ == "__main__":
    print("Live speech recognition")

    recognition = LiveRecognition("facebook/wav2vec2-large-robust-ft-swbd-300h")

    recognition.start()

    try:
        while True:
            text,sample_length,inference_time, confidence = recognition.get_last_text()
            print(f"{sample_length:.3f}s\t{inference_time:.3f}s\t{confidence}\t{text}")

    except KeyboardInterrupt:
        recognition.stop()
        exit()
