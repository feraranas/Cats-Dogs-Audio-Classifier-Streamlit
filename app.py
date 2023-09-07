import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy import stats

# ////////////////////////
# AUXILIARY FUNCTIONS
# ////////////////////////
def load_dataset(dataframe, name):
     df = dataframe
     arrays = [np.array(item) for item in df['audio_wav']]
     v = np.concatenate(arrays, axis=0).astype('float32')
     # v = np.concatenate([item for item in df], axis=0).astype('float32')
     df_std, df_mean = v.std(), v.mean()
     std, mean = (df_std, df_mean)
     v = (v - mean) / std
     print('loaded {} with {} sec of audio'.format(name, len(v) / 16000))

def get_amplitude_envelope(signal,frame_size,window_size):
  envelope = []
  for i in range(0,len(signal),window_size):
    frame_data = signal[i:i+frame_size]
    max_amp = max(frame_data)
    envelope.append(max_amp)
  return np.array(envelope)

def get_rms(signal,frame_size,window_size):
  new_signal = []
  for i in range(0,len(signal),window_size):
    frame_data = signal[i:i+frame_size]
    rmse_val = np.sqrt(1 / len(frame_data) * sum(i**2 for i in frame_data))
    new_signal.append(rmse_val)
  return np.array(new_signal)

# ////////////////////////
# TITLE & TEAM INFO
# ////////////////////////
st.title('Final Assessment')
st.header('Equipo 4')
team = pd.DataFrame({
     'Alumno': [
         'Mauricio Juárez Sánchez',
         'Alfredo Jeong Hyun Park',
         'Fernando Alfonso Arana Salas',
         'Miguel Ángel Bustamante Pérez'],
     'Matricula': [
      'A01660336',
      'A01658259',
      'A01272933',
      'A01781583']
     })
st.write(team)

# ////////////////////////
# READING DATA FROM URL
# ////////////////////////
cat_df_url = 'https://raw.githubusercontent.com/feraranas/ML_Assessments/master/data/cat_df.csv'
dog_df_url = 'https://raw.githubusercontent.com/feraranas/ML_Assessments/master/data/dog_df.csv'
cat_df = pd.read_csv(cat_df_url)
dog_df = pd.read_csv(dog_df_url)

# ////////////////////////
# IMPORTED LIBRARIES
# ////////////////////////
st.header('[1] Libraries used in the project')
libraries_col1, libraries_col2 = st.columns(2, gap='medium')
with libraries_col1:
     libraries = '''
          import numpy as np  # linear algebra
          import pandas as pd  # CSV file
          import matplotlib.pyplot as plt
          import random
          import os
          from tqdm import tqdm
          import librosa
          from IPython.display import Audio
     '''
     st.code(libraries, language="python", line_numbers=False)

with libraries_col2:
     st.caption('NUMPY was imported for its linear algebra capabilities.')
     st.caption('PANDAS was imported to read csv files.')
     st.caption('MATPLOTLIB was imported to plot graphs.')
     st.caption('RANDOM was imported to generate aleatory numbers.')
     st.caption('OS was imported to read files from local directories.')
     st.caption('TQDM was imported to unzip files from local directories.')
     st.caption('LIBROSA was imported to read audio files.')

# ////////////////////////
# DATA VISUALIZATION
# ////////////////////////
st.header('[2] Data set Visualization (20 pts)')
cat_audio, cat_sr = librosa.load('./cat_166.wav')
dog_audio, dog_sr = librosa.load('./dog_barking_74.wav')
data_vis_col1, data_vis_col2 = st.columns(2, gap='large')
with data_vis_col1:
   st.subheader("Cat Dataset")
   st.write(pd.DataFrame(dog_df[['file_name', 'audio_wav']].values, columns=['file_name', 'audio_wav']))
   st.caption('Loaded Cats dataset with {1323.899909297052} sec of audio.')
   st.markdown('Example: cat_166.wav')
   audio_file_cat = open('./cat_166.wav', 'rb')
   audio_bytes = audio_file_cat.read()
   st.audio(audio_bytes, format='audio/wav')
   fig, ax = plt.subplots()
   librosa.display.waveshow(cat_audio[:400], sr = cat_sr)
   st.pyplot(fig)

with data_vis_col2:
   st.subheader("Dog Dataset")
   st.write(pd.DataFrame(cat_df[['file_name', 'audio_wav']].values, columns=['file_name', 'audio_wav']))
   st.caption('Loaded Dogs dataset with {598.4407256235827} sec of audio.')
   st.markdown('Example: dog_barking_74.wav')
   audio_file_dog = open('./dog_barking_74.wav', 'rb')
   audio_bytes = audio_file_dog.read()
   st.audio(audio_bytes, format='audio/wav')
   fig, ax = plt.subplots()
   librosa.display.waveshow(dog_audio[:400], sr = dog_sr)
   st.pyplot(fig)

st.caption('''We can see that both plots are different, but in this case they are periodic waves.
           In terms of **Amplitude** the dog's wave has more amplitude than the cat's wave, this menas it has a higher air pressure disturbance, in this case it's something interesting because the dog is just barking and it also appears that is not that close to the microphone that is recording the audio, this means that the dog's bark is more loudly and powerful, in contrast the cat's audio has different changes in it's shape, this is because the animal is not only meowing, it's also purring, that creates a different sample with different tonalities.
           Another thing to consider is that the frequency of the cat's wave and the dog's wave is the same, both are equal to 22050 Hz.''')

# ////////////////////////
# FEATURE EXTRACTION
# ////////////////////////
st.header('Feature extraction data set (20 pts)')
# AMPLITUDE ENVELOPE
st.subheader('Amplitude Envelope')
amplitude_env_col1, amplitude_env_col2 = st.columns(2, gap='large')
with amplitude_env_col1:
     st.markdown('Example: cat_166.wav')
     amplitude_envelope=get_amplitude_envelope(cat_audio,1024,512)
     frames = range(0, len(amplitude_envelope))
     time = librosa.frames_to_time(frames, hop_length=512)
     fig, ax = plt.subplots()
     librosa.display.waveshow(cat_audio, alpha=0.5)
     plt.plot(time, amplitude_envelope, color="r")
     st.pyplot(fig)

with amplitude_env_col2:
     st.markdown('Example: dog_barking_74.wav')
     amplitude_envelope=get_amplitude_envelope(dog_audio,1024,512)
     frames = range(0, len(amplitude_envelope))
     time = librosa.frames_to_time(frames, hop_length=512)
     fig, ax = plt.subplots()
     librosa.display.waveshow(dog_audio, alpha=0.5)
     plt.plot(time, amplitude_envelope, color="r")
     st.pyplot(fig)

st.caption('''We can see that [...INSERT AN EXPLANATION FOR AMPLITUDE ENVELOPE]''')

# RMS
st.subheader('RMS')
rms_col1, rms_col2 = st.columns(2, gap='large')
with rms_col1:
     st.markdown('Example: cat_166.wav')
     rms=get_rms(cat_audio,1024,512)
     frames = range(0, len(rms))
     time = librosa.frames_to_time(frames, hop_length=512)
     fig, ax = plt.subplots()
     librosa.display.waveshow(cat_audio, alpha=0.5)
     plt.plot(time, rms, color="r")
     st.pyplot(fig)
     
with rms_col2:
     st.markdown('Example: dog_barking_74.wav')
     rms=get_rms(dog_audio,1024,512)
     frames = range(0, len(rms))
     time = librosa.frames_to_time(frames, hop_length=512)
     fig, ax = plt.subplots()
     librosa.display.waveshow(dog_audio, alpha=0.5)
     plt.plot(time, rms, color="b")
     st.pyplot(fig)

st.caption('''We can see that [...INSERT AN EXPLANATION FOR RMS]''')

fig, ax = plt.subplots()
librosa.display.waveshow(cat_audio, color='r',alpha=0.5)
librosa.display.waveshow(dog_audio, color='b',alpha=0.5)
st.pyplot(fig)

# FREQUENCY DOMAIN
st.subheader('Frequency Domain')
freq_col1, freq_col2 = st.columns(2, gap='large')
with freq_col1:
     st.markdown('Example: cat_166.wav')
     cat_f = np.abs(np.fft.fft(cat_audio))
     freq_steps = np.fft.fftfreq(cat_audio.size, d=1/cat_sr)
     fig, ax = plt.subplots()
     plt.plot(freq_steps, cat_f, color='r')
     plt.xlabel("Frequency [Hz]")
     plt.ylabel("Amplitude")
     st.pyplot(fig)

with freq_col2:
     st.markdown('Example: dog_barking_74.wav')
     dog_f = np.abs(np.fft.fft(dog_audio))
     freq_steps = np.fft.fftfreq(dog_audio.size, d=1/dog_sr)
     fig, ax = plt.subplots()
     plt.plot(freq_steps, dog_f, color='r')
     plt.xlabel("Frequency [Hz]")
     plt.ylabel("Amplitude")
     st.pyplot(fig)

st.caption('''We can see that [...INSERT AN EXPLANATION FOR FREQ DOMAIN GRAPHS ABOVE]''')

freq2_col1, freq2_col2 = st.columns(2, gap='large')
with freq2_col1:
     st.write("Nobs:", stats.describe(cat_audio)[0])
     st.write("minmax:", stats.describe(cat_audio)[1])
     st.write("mean:", stats.describe(cat_audio)[2])
     st.write("variance:", stats.describe(cat_audio)[3])
     st.write("skewness:", stats.describe(cat_audio)[4])
     st.write("kurtosis:", stats.describe(cat_audio)[5])
with freq2_col2:
     st.write("Nobs:", stats.describe(dog_audio)[0])
     st.write("minmax:", stats.describe(dog_audio)[1])
     st.write("mean:", stats.describe(dog_audio)[2])
     st.write("variance:", stats.describe(dog_audio)[3])
     st.write("skewness:", stats.describe(dog_audio)[4])
     st.write("kurtosis:", stats.describe(dog_audio)[5])

st.caption('''We can see that [...INSERT AN EXPLANATION FOR nobs, minmax, mean, skewness, etc]''')

# SPECTOGRAM
st.subheader('Spectogram')
spectogram_col1, spectogram_col2 = st.columns(2, gap='large')
with spectogram_col1:
     st.markdown('Example: cat_166.wav')
     D = librosa.stft(cat_audio)
     S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
     fig, ax = plt.subplots()
     img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
     fig.colorbar(img, ax=ax, format="%+2.f dB")
     st.pyplot(fig)

with spectogram_col2:
     st.markdown('Example: dog_barking_74.wav')
     D = librosa.stft(dog_audio)
     S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
     fig, ax = plt.subplots()
     img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
     fig.colorbar(img, ax=ax, format="%+2.f dB")
     st.pyplot(fig)

st.caption('''We can see that [...INSERT AN EXPLANATION FOR Spectograms & comparison]''')

# ////////////////////////
# DIMENSION ANALYSIS
# ////////////////////////
st.header('Dimension analysis (40 pts)')
st.caption('''Here, we add an explanation for FEATURE EXTRACTION [Time Domain Features]. 
           We use "from scipy import stats" to extract max_amplitude, min_amplitude, minmax, mean & variance. From each wav_file in the dataset. We already have rms and zcr. 
           So now we have the following features:''')
st.markdown('''min_ae: ''')
st.markdown('''max_ae: ''')
st.markdown('''mean_ae: ''')
st.markdown('''variance_ae: ''')
st.markdown('''min_rms: ''')
st.markdown('''max_rms: ''')
st.markdown('''mean_rms: ''')
st.markdown('''variance_rms: ''')
st.markdown('''min_zcr: ''')
st.markdown('''max_zcr: ''')
st.markdown('''mean_zcr: ''')
st.markdown('''variance_zcr: ''')

dimension_a_col1, dimension_a_col2 = st.columns(2, gap='large')
with dimension_a_col1:
     st.subheader("Cat Dataset")
     st.write(pd.DataFrame(cat_df.values, columns=[i for i in cat_df.columns]))

with dimension_a_col2:
    st.subheader("Dog Dataset")
    st.write(pd.DataFrame(dog_df.values, columns=[i for i in dog_df.columns]))



# ////////////////////////
# ML MODELS
# ////////////////////////
st.header('ML model (20 pts)')