import tkinter as tk
from tkinter import Label, Text
from tkinter import filedialog
import pygame
import librosa
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

pygame.mixer.init()

# Loading the test dataset
test_ds_path = "./Dataset-2/test_full.csv"
test_ds = pd.read_csv(test_ds_path)

root = tk.Tk()
root.title("Speaker Identification System")
root.geometry("400x400")

audio_file = ""
input_text = ""
output = ""
playing = False

# To upload audio file
def upload_audio():
    global audio_file
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3;*.wav")])
    if file_path:
        audio_file = file_path
        # Enable play button once audio is selected
        play_button.config(state=tk.NORMAL)

# To play or pause audio
def toggle_audio():
    global playing
    if not playing:
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        playing = True
        play_button.config(text="Pause")
    else:
        pygame.mixer.music.pause()
        playing = False
        play_button.config(text="Play")

# For prediction
def predict_speaker():
    global input_text, output
    input_text = name_entry.get()
    
    if not audio_file:
        output_label.config(text="Please upload an audio file.")
        return
    
    if not input_text:
        output_label.config(text="Please enter the speaker's name.")
        return
    
    if input_text not in test_ds['speaker'].unique():
        output_label.config(text="Invalid speaker.")
        return

    # Load and preprocess the audio file
    try:
        audio, sr = librosa.load(audio_file, sr=16000, duration=3, mono=True)
        windowed_audio = audio
        fourier = np.fft.rfft(windowed_audio)
        fixed_num_freq_samples = 24001
        resampled_fourier = np.interp(np.linspace(0, len(fourier) - 1, fixed_num_freq_samples), np.arange(len(fourier)), fourier)
        norm_amplitude = np.abs(resampled_fourier) / (len(audio) / 2)
        X_test = np.array([norm_amplitude], dtype=float)
        
        # Perform prediction
        loaded_model = load_model("./1d_cnn_model_without_window_new.h5")
        y_pred = loaded_model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))
        predicted_label = np.argmax(y_pred, axis=1)
        

        labels=[]

        for label in test_ds['speaker']:
            labels.append(label)
        label_encoder = LabelEncoder()

        y_actual = label_encoder.fit_transform(labels)

        label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

        y_actual = label_mapping[input_text]
        if predicted_label == y_actual:
            output = "Prediction: Correct speaker"
        else:
            output = "Prediction: Incorrect speaker"
    except Exception as e:
        output = f"An error occurred: {str(e)}"
    
    output_label.config(text=output)

# UI elements
name_label = tk.Label(root, text="Enter Speaker's Name:")
name_label.pack()

name_entry = tk.Entry(root, width=50)
name_entry.pack()

upload_button = tk.Button(root, text="Upload Audio File", command=upload_audio)
upload_button.pack(pady=10)

play_button = tk.Button(root, text="Play", command=toggle_audio, state=tk.DISABLED)
play_button.pack(pady=10)

predict_button = tk.Button(root, text="Predict Speaker", command=predict_speaker)
predict_button.pack(pady=10)

output_label = tk.Label(root, text="")
output_label.pack()

root.mainloop()
