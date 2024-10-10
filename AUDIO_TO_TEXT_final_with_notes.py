import numpy as np                           # Import NumPy for numerical operations
import librosa                               # Import Librosa for audio processing tasks
import soundfile as sf                       # Import SoundFile for reading and writing audio files
import torch                                 # Import PyTorch for deep learning computations
from pyannote.audio import Pipeline          # Import Pipeline from pyannote.audio for speaker diarization
from transformers import WhisperProcessor, WhisperForConditionalGeneration  # Import Whisper models from Hugging Face
from pydub import AudioSegment, silence      # Import AudioSegment and silence detection from pydub

def detect_silence(audio_file, min_silence_len=1000, silence_thresh_offset=-16):
    """
    Detects silent regions in an audio file.

    Parameters:
    - audio_file: Path to the audio file.
    - min_silence_len: Minimum length of silence to detect (in milliseconds).
    - silence_thresh_offset: Offset applied to the average dBFS to determine silence threshold.
    
    Returns:
    - List of tuples with start and end times (in seconds) of silent regions.
    """
    myaudio = AudioSegment.from_file(audio_file)       # Load the audio file into an AudioSegment object
    dBFS = myaudio.dBFS                                # Get the average loudness of the audio in dBFS
    silence_regions = silence.detect_silence(          # Detect silent regions in the audio
        myaudio,
        min_silence_len=min_silence_len,               # Minimum length of a silence to be considered (in ms)
        silence_thresh=dBFS + silence_thresh_offset    # Silence threshold in dBFS
    )
    # Convert silence regions from milliseconds to seconds
    silence_regions = [(start / 1000, stop / 1000) for start, stop in silence_regions]
    return silence_regions                             # Return the list of silent regions

def diarize_speakers(audio_file):
    """
    Performs speaker diarization on an audio file.

    Parameters:
    - audio_file: Path to the audio file.

    Returns:
    - Diarization object containing speaker segments.
    """
    hf_auth_token = "your_hugging_face_authentication_token_here"  # Hugging Face authentication token
    diarization = None                                             # Initialize diarization variable
    try:
        # Load the pre-trained speaker diarization pipeline from Hugging Face
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=hf_auth_token                           # Use your Hugging Face auth token
        )
        # Apply the pipeline to the audio file to get diarization results
        diarization = pipeline(audio_file)
        return diarization                                         # Return the diarization results
    except Exception as e:
        print(f"Error initializing pyannote pipeline: {e}")        # Print error message if initialization fails
        return None                                                # Return None if diarization fails

def transcribe_audio(audio_file):
    """
    Transcribes an audio file using the Whisper model.

    Parameters:
    - audio_file: Path to the audio file.

    Returns:
    - List of transcribed text segments.
    """
    model_name = "openai/whisper-large-v2"                         # Specify the Whisper model to use
    auth_token = "your_hugging_face_authentication_token_here"     # Hugging Face authentication token

    # Load the Whisper model and processor from Hugging Face
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, use_auth_token=auth_token
    )
    processor = WhisperProcessor.from_pretrained(
        model_name, use_auth_token=auth_token
    )

    # Get forced decoder IDs for English transcription tasks
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="en", task="transcribe"
    )
    
    # Load the audio data at a sampling rate of 16 kHz
    audio_data, sr = librosa.load(audio_file, sr=16000)
    # Process the audio data into input features for the model
    input_features = processor(
        audio_data, sampling_rate=sr, return_tensors="pt"
    ).input_features

    # Create an attention mask (optional for this model)
    attention_mask = torch.ones(
        input_features.shape[-1], dtype=torch.long
    ).unsqueeze(0)

    # Determine the device to use (GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_features = input_features.to(device)                     # Move input features to the device
    model = model.to(device)                                       # Move the model to the device

    # Generate transcription tokens using the model
    generated_tokens = model.generate(
        input_features,
        forced_decoder_ids=forced_decoder_ids,
        attention_mask=attention_mask
    )

    # Decode the tokens to get the transcribed text
    transcription_with_timestamps = processor.batch_decode(
        generated_tokens, skip_special_tokens=True
    )

    return transcription_with_timestamps                           # Return the transcription results

def process_audio(audio_file):
    """
    Processes an audio file to detect silence, perform speaker diarization,
    and transcribe the audio with speaker labels.

    Parameters:
    - audio_file: Path to the audio file.

    Returns:
    - List of formatted transcription strings with speaker labels and silence markers.
    """
    # Load the audio data at a sampling rate of 16 kHz
    audio_data, sr = librosa.load(audio_file, sr=16000)
    
    temp_wav_file = "temp.wav"                                     # Define a temporary WAV file name
    sf.write(temp_wav_file, audio_data, sr)                        # Write the audio data to the temporary WAV file
    
    silence_regions = detect_silence(temp_wav_file)                # Detect silence in the audio
    
    diarization = diarize_speakers(temp_wav_file)                  # Perform speaker diarization
    if diarization is None:
        print("Diarization failed. Exiting.")                      # Exit if diarization fails
        return []
    
    transcription = transcribe_audio(temp_wav_file)                # Transcribe the audio
    if transcription:
        print("Transcription output:", transcription)              # Print the transcription output
    else:
        print("No transcription detected.")                        # Notify if no transcription was detected
        return []
    
    output = []                                                    # Initialize list to hold the final output
    speaker_mapping = {}                                           # Dictionary to map speaker IDs to labels
    current_speaker = 0                                            # Counter for assigning speaker labels
    
    # Estimate time per segment based on transcription length
    total_duration = len(audio_data) / sr                          # Total duration of the audio in seconds
    time_per_segment = total_duration / len(transcription)         # Approximate duration per transcription segment
    
    # Iterate over each transcribed segment
    for i, text in enumerate(transcription):
        start_time = i * time_per_segment                          # Calculate start time of the segment
        end_time = start_time + time_per_segment                   # Calculate end time of the segment

        speaker_label = None                                       # Initialize speaker label
        # Iterate over diarization segments to find matching speaker
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Check if the transcription segment overlaps with the diarization segment
            if (turn.start <= start_time < turn.end) or (turn.start < end_time <= turn.end):
                if speaker not in speaker_mapping:
                    # Map the speaker ID to a speaker label like SPEAKER_0, SPEAKER_1, etc.
                    speaker_mapping[speaker] = f"SPEAKER_{current_speaker}"
                    current_speaker += 1
                speaker_label = speaker_mapping[speaker]           # Get the speaker label
                break                                              # Stop checking after finding the speaker
        
        # Check if the segment is within a silence region
        is_silence = any(
            start <= start_time < end for start, end in silence_regions
        )
        
        # Construct the output string based on the segment type
        if is_silence:
            output.append(f"[SILENCE] {start_time:.2f} - {end_time:.2f}")  # Mark as silence
        elif speaker_label:
            output.append(f"[{speaker_label}] {start_time:.2f} - {end_time:.2f}: {text}")  # Include speaker label
        else:
            output.append(f"[UNKNOWN] {start_time:.2f} - {end_time:.2f}: {text}")  # Mark as unknown speaker
    
    return output                                                  # Return the final output list

if __name__ == "__main__":
    # Define the path to your audio file (replace with your actual file path)
    audio_file_path = 'path_to_your_audio_file.wav'
    # Process the audio file and get the results
    results = process_audio(audio_file_path)
    # Print each line of the results
    for line in results:
        print(line)