import numpy as np
import librosa
import soundfile as sf
import torch
from pyannote.audio import Pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pydub import AudioSegment, silence


def detect_silence(audio_file, min_silence_len=1000, silence_thresh_offset=-16):

    myaudio = AudioSegment.from_file(audio_file)
    dBFS = myaudio.dBFS
    silence_regions = silence.detect_silence(myaudio, min_silence_len=min_silence_len, silence_thresh=dBFS + silence_thresh_offset)
    silence_regions = [(start / 1000, stop / 1000) for start, stop in silence_regions]
    return silence_regions


def diarize_speakers(audio_file):
    hf_auth_token = "your_hugging_face_authetication_token_here"
    diarization = None
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_auth_token)
        diarization = pipeline(audio_file)
        return diarization

    except Exception as e:
        print(f"Error initializing pyannote pipeline: {e}")
        return None


def transcribe_audio(audio_file):
    model_name = "openai/whisper-large-v2"
    auth_token = "your_hugging_face_authetication_token_here"

    model = WhisperForConditionalGeneration.from_pretrained(model_name, token=auth_token)
    processor = WhisperProcessor.from_pretrained(model_name, token=auth_token)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    
    audio_data, sr = librosa.load(audio_file, sr=16000)
    input_features = processor(audio_data, sampling_rate=sr, return_tensors="pt").input_features

    attention_mask = torch.ones(input_features.shape[-1], dtype=torch.long).unsqueeze(0)
    input_features = input_features.to("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    generated_tokens = model.generate(input_features, forced_decoder_ids=forced_decoder_ids, attention_mask=attention_mask)

    transcription_with_timestamps = processor.batch_decode(generated_tokens, skip_special_tokens=True)

    return transcription_with_timestamps


def process_audio(audio_file):

    audio_data, sr = librosa.load(audio_file, sr=16000)
    
    temp_wav_file = "temp.wav"
    sf.write(temp_wav_file, audio_data, sr)
    
    silence_regions = detect_silence(temp_wav_file)
    
    diarization = diarize_speakers(temp_wav_file)
    if diarization is None:
        print("Diarization failed. Exiting.")
        return []
    
    transcription = transcribe_audio(temp_wav_file)
    
    if transcription:
        print("Transcription output:", transcription)
    else:
        print("No transcription detected.")
    
    output = []
    speaker_mapping = {}
    current_speaker = 0
    
    for i, text in enumerate(transcription):
        start_time = i
        end_time = start_time + 42
        

        speaker_label = 'speaker'
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if (turn.start <= start_time < turn.end) or (turn.start < end_time <= turn.end):
                if speaker not in speaker_mapping:
                    speaker_mapping[speaker] = f"SPEAKER_{current_speaker}"
                    current_speaker += 1
                speaker_label = speaker_mapping[speaker]
                break
        
        is_silence = any(start <= start_time < end for start, end in silence_regions)
        
        if is_silence:
            output.append(f"[SILENCE] {start_time:.2f} - {end_time:.2f}")
        elif speaker_label:
            output.append(f"[{speaker_label}] {start_time:.2f} - {end_time:.2f}: {text}")
        else:
            output.append(f"[UNKNOWN] {start_time:.2f} - {end_time:.2f}: {text}")
    
    return output, silence_regions, diarization


results = process_audio('C:/Users/USERNAME/Music/FILENAME' )
for line in results:
    print(line)

results = process_audio('C:/Users/USERNAME/Music/FILENAME')
for line in results:
    print(line)