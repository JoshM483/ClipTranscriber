
# Install CUDA https://developer.nvidia.com/cuda-downloads
# Install requirements.txt
# You will need to manually install torch to use CUDA. Ref: https://pytorch.org/get-started/locally/
import os
import time
import pandas as pd
from pydub import AudioSegment
import whisper
import torch

# Benchmark example
# https://github.com/openai/whisper/discussions/918

"""
Simple local transcription for a large number of small audio files
Requires installing CUDA and a using a compatible GPU
CPU may be possible my changing device = torch.device('cuda') to 'cpu'
but CPU may be slower than HuggingFace inference. 
"""

# print(torch.cuda.is_available())  # Should return True if the GPU is available and correctly set up



def transcribe_with_whisper(model, filename):
    """Sets up whisper transcription task"""
    print(f"Transcribing {filename}.")
    result = model.transcribe(filename, language='en', task='transcribe', fp16=False) # FP16=False should force FP32, potentially less hallucinated results
    return result["text"] if "text" in result else "No transcription found"


# Organize transcription requests
def transcribe_files(model, file_paths):
    """
    """
    transcriptions = []
    for file_path in file_paths:
        try:
            transcription = transcribe_with_whisper(model, file_path)
            transcriptions.append((os.path.basename(file_path), transcription))
        except Exception as e:
            print(f"Error transcribing {file_path}: {e}")
            transcriptions.append((os.path.basename(file_path), "Error during transcription"))
    return transcriptions

def update_csv(output_csv_path, transcriptions, mode='a'):
    df = pd.DataFrame(transcriptions, columns=['File Name', 'Transcription'])
    df.to_csv(output_csv_path, mode=mode, header=(not os.path.exists(output_csv_path)), index=False)

def get_audio_duration(file_path):
    """Info function to report total audio duration of all transcribed files."""
    try:
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000.0
    except Exception as e:
        print(f"Error getting duration of {file_path}: {e}")
        return 0
        
def main(folder_path, output_path, batch_size, model_size):
    start_time = time.time()

    # Load model, download if not available
    print("Loading Whisper model...")
    device = torch.device('cuda')   # Run on GPU
    model = whisper.load_model(model_size, device=device)

    # Set up loading files and output
    audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith((".mp3", ".wav"))]
    output_csv_path = os.path.join(output_path, 'transcriptions.csv')
    total_files_transcribed = 0
    total_duration = 0
    
    # Batch files, run transcription model and update output file at end of batch
    for i in range(0, len(audio_files), batch_size):
        batch_files = audio_files[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{len(audio_files)//batch_size + 1}...")
        
        transcriptions = []
        for file_path in batch_files:
            duration = get_audio_duration(file_path)
            total_duration += duration
            transcription_result = transcribe_files(model, [file_path])
            transcriptions.extend(transcription_result)
            total_files_transcribed += 1
        update_csv(output_csv_path, transcriptions, mode='a' if i > 0 else 'w')
        print(f"Batch {i//batch_size + 1} completed and saved.")

    # Quick Report
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Transcription completed. Total files transcribed: {total_files_transcribed}")
    print(f"Data saved to '{output_csv_path}'.")
    print(f"Total audio duration: {total_duration // 60:.0f}:{total_duration % 60:.0f} ")
    print(f"Total process time: {total_time // 60:.0f} minutes and {total_time % 60:.0f} seconds.")


folder_path = 'TestSounds'          # Replace with the path to your folder containing audio files
output_path = ''                    # Replace with the path where you want to save the csv
batch_size = 10                     # Adjust this value to control the number of audio files processed in each batch before appending to the CSV
model_size = 'medium'               # Whisper Model size 
main(folder_path, output_path, batch_size, model_size)