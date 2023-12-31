import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import pickle
import joblib
# Models
from sklearn.linear_model import LogisticRegression






# ////////////////////////
# READING DATA FROM URL
# ////////////////////////
cat_df_url = 'https://raw.githubusercontent.com/feraranas/ML_Assessments/master/data/dataset_cat.csv'
dog_df_url = 'https://raw.githubusercontent.com/feraranas/ML_Assessments/master/data/dataset_dog.csv'
dataset_url = 'https://raw.githubusercontent.com/feraranas/ML_Assessments/master/data/dataset_full.csv'
cat_df = pd.read_csv(cat_df_url)
dog_df = pd.read_csv(dog_df_url)
dataset = pd.read_csv(dataset_url)






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

def amplitude_envelope(signal, frame_size, window_size):
  return np.array([max(signal[i:i+frame_size]) for i in range(0, signal.size, window_size)])

def get_rms(signal,frame_size,window_size):
  new_signal = []
  for i in range(0,len(signal),window_size):
    frame_data = signal[i:i+frame_size]
    rmse_val = np.sqrt(1 / len(frame_data) * sum(i**2 for i in frame_data))
    new_signal.append(rmse_val)
  return np.array(new_signal)

def extract_time_domain_features(animal_data):
    FRAME_SIZE = 1024
    WINDOW_SIZE = 128
    ae = amplitude_envelope(animal_data,FRAME_SIZE,WINDOW_SIZE)
    rms = librosa.feature.rms(y=animal_data, frame_length=FRAME_SIZE, hop_length=WINDOW_SIZE)[0]
    zcr = librosa.feature.zero_crossing_rate(animal_data, frame_length=FRAME_SIZE, hop_length=WINDOW_SIZE)[0]
    ae_stats = stats.describe(ae)
    rms_stats = stats.describe(rms)
    zcr_stats = stats.describe(zcr)
    fq = np.abs(np.fft.fft(animal_data))
    fq_stats = stats.describe(fq)
    data_features_vector=[ae_stats.minmax[0],ae_stats.minmax[1],ae_stats.mean,ae_stats.variance,
                         rms_stats.minmax[0],rms_stats.minmax[1],rms_stats.mean,rms_stats.variance,
                         zcr_stats.minmax[0],zcr_stats.minmax[1],zcr_stats.mean,zcr_stats.variance,
                         fq_stats.minmax[0], fq_stats.minmax[1], fq_stats.mean, fq_stats.variance]
    return data_features_vector



# ////////////////////////
# TITLE & TEAM INFO
# ////////////////////////
st.title('Cats & Dogs Classifier analysis')
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
# START -> CHOOSE A MODEL
# ////////////////////////
st.title('ML model (20 pts)')
st.caption('We trained the following models:')
log_reg_pickle = open('./models/logistic_regression_model.pkl', 'rb')
log_reg = pickle.load(log_reg_pickle)
knn_pickle = open('./models/knn_model.pkl', 'rb')
knn = pickle.load(knn_pickle)
naive_bayes_pickle = open('./models/naive_bayes_model.pkl', 'rb')
naive_bayes = pickle.load(naive_bayes_pickle)
model_col1, model_col2 = st.columns(2, gap='large')
with model_col1:
     st.write('Linear Regression Classifier')
     st.write(log_reg)
with model_col2:
     st.write('K Nearest Neighbors Classifier')
     st.write(knn)
st.write('Bayes Classifier')
st.write(naive_bayes)

st.title('See our results for yourself. Choose a model.')
st.subheader('Click on "Submit" button after selecting a model & choosing an audio file.')
with st.form('User_input'):
    selected_model = st.selectbox('Model', ['Logistic_Regression_Classifier', 'K_Nearest_Neighbors_Classifier', 'Naive_Bayes_Classifier'])
    wav_file = st.file_uploader('Select your own sound file')
    if wav_file is not None:
        uploaded_audio, _ = librosa.load(wav_file)
    st.form_submit_button()

if selected_model == "Logistic_Regression_Classifier":
    if not wav_file:
        st.subheader('No audio chosen.')
    else:
        user_audio = extract_time_domain_features(uploaded_audio)
        log_reg_prediction = log_reg.predict([user_audio])
        st.subheader('Result: {} species'.format(log_reg_prediction))
elif selected_model == "K_Nearest_Neighbors_Classifier":
    if not wav_file:
        st.subheader('No audio chosen.')
    else:
        user_audio = extract_time_domain_features(uploaded_audio)
        knn_prediction = knn.predict([user_audio])
        st.subheader('Result: {} species'.format(knn_prediction))
elif selected_model == "Naive_Bayes_Classifier":
    if not wav_file:
        st.subheader('No audio chosen.')
    else:
        user_audio = extract_time_domain_features(uploaded_audio)
        naive_bayes_prediction = naive_bayes.predict([user_audio])
        st.subheader('Result: {} species'.format(naive_bayes_prediction))












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
   st.write(cat_df.head())
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
   st.write(dog_df.head())
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
st.header('[3] Feature extraction data set (20 pts)')
# AMPLITUDE ENVELOPE
st.subheader('Amplitude Envelope')
st.caption('''
    - ADSR stands for Attack, Decay, Sustain, and Release, which are stages in sound synthesis.
    - In the context of time domain features, the amplitude envelope characterizes how a sound evolves over time.
    - Attack: The initial increase in amplitude when a sound is triggered.
    - Decay: The decrease in amplitude after the attack, leading to a sustained level.
    - Sustain: The constant amplitude during the main part of the sound.
    - Release: The gradual decrease in amplitude after the sound is released or stopped.
''')
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

# RMS
st.subheader('RMS')
st.caption('''
    - RMS measures the average power or energy of a signal over a period of time.
    - It calculates the square root of the mean of the squared amplitudes of the signal.
    - RMS is often used to estimate the loudness or energy of an audio segment.
''')
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

fig, ax = plt.subplots()
librosa.display.waveshow(cat_audio, color='r',alpha=0.5)
librosa.display.waveshow(dog_audio, color='b',alpha=0.5)
st.pyplot(fig)

# FREQUENCY DOMAIN
st.subheader('Frequency Domain')
st.caption('''
    - Frequency represents the number of cycles (oscillations) of a sound wave per second.
    - It is measured in Hertz (Hz) and determines the pitch of a sound.
    - High-frequency sounds have a higher pitch, while low-frequency sounds have a lower pitch.
''')
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

st.caption('''
The stats.describe function returns an object that contains several statistical properties of the dataset. We can analyze the following properties:''')
st.caption('''- nobs: The number of observations (data points) in the dataset.''')
st.caption('''- minmax: A tuple containing the minimum and maximum values in the dataset.''')
st.caption('''- mean: The arithmetic mean (average) of the dataset.''')
st.caption('''- variance: The variance, which measures the spread or dispersion of the data.''')
st.caption('''- skewness: A measure of the skewness (asymmetry) of the data distribution.''')
st.caption('''- kurtosis: A measure of the kurtosis (tailedness) of the data distribution.''')
st.caption('''- std_dev: The standard deviation, which is the square root of the variance.''')
st.caption('''- sem: The standard error of the mean, which estimates the standard deviation of the sample mean.''')

st.caption('''1. `nobs` (Number of Observations):
   - There are 28,522 observations or data points in your dataset. This tells you the total number of data instances you have for analysis.''')

st.caption('''
    2. `minmax` (Minimum and Maximum Values):
   - The minimum value in your dataset is approximately -0.996.
   - The maximum value in your dataset is approximately 0.982.
   - These values provide information about the range or spread of your data. In your case, your data varies between -0.996 and 0.982.''')

st.caption('''3. `mean` (Mean or Average):
   - The mean value of your dataset is approximately 0.0000186 (or 1.8618754e-05 in scientific notation).
   - The mean represents the central tendency of your data. In this case, it's close to zero, indicating that, on average, your data is centered around this value.''')

st.caption('''4. `variance` (Variance):
   - The variance of our dataset is approximately 0.0857.
   - Variance measures the spread or dispersion of your data. A higher variance suggests greater variability among data points.''')

st.caption('''5. `skewness` (Skewness):
   - The skewness of our dataset is approximately -0.156.
   - Skewness measures the asymmetry of the data distribution. A negative skewness indicates a slight leftward (negative) skew, meaning the data distribution has a longer tail on the left side.''')

st.caption('''6. `kurtosis` (Kurtosis):
   - The kurtosis of our dataset is approximately 2.063.
   - Kurtosis measures the tailedness or peakedness of the data distribution. A value greater than 3 suggests a distribution with heavier tails and a sharper peak (leptokurtic).''')

st.caption('''
- We have a moderate number of observations (28,522).
- The data values range between approximately -0.996 and 0.982.
- On average, the data is very close to zero.
- There is some variability in the data, as indicated by the variance.
- The data distribution is slightly negatively skewed, meaning it has a longer tail on the left.
- The kurtosis suggests that the distribution has heavier tails and a sharper peak compared to a normal distribution.''')

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

st.caption('''In the following spectrogram we used a visual representation of the frequency content of a signal over time. We can visualize a representation of the amplitude at a particular moment in time. Bright regions indicate strong or promiment frequencies and darker regions indicate weaker frequencies.
           Also, we can visualize horizontal lines that may correspond to continuous tones or steady-state sounds. But in the case of some dogs audio, vertical lines may represent abrupt changes or transient events.''')

st.subheader('Grand Features')
st.caption('''Here, we add an explanation for FEATURE EXTRACTION [Time Domain Features]. 
           We use "from scipy import stats" to extract max_amplitude, min_amplitude, minmax, mean & variance. From each wav_file in the dataset. We already have rms and zcr. 
           So now we have the following features:''')
st.markdown('''min_ae: Rrepresents the smallest amplitude envelope value observed in the audio.''')
st.markdown('''max_ae: Rrepresents the largest amplitude envelope value observed in the audio.''')
st.markdown('''mean_ae: Represents the average amplitude envelope value over the entire audio.''')
st.markdown('''variance_ae: Qquantifies the spread or dispersion of the amplitude envelope values.''')
st.markdown('''min_rms: Smallest RMS value observed in the audio.''')
st.markdown('''max_rms: Largest RMS value observed in the audio. It indicates the maximum energy or power over time.''')
st.markdown('''mean_rms: Average RMS value over the entire audio. It provides insight into the central tendency of the energy or power.''')
st.markdown('''variance_rms: Quantifies the spread or dispersion of the RMS values. A higher variance suggests greater variability in energy or power over time.''')
st.markdown('''min_zcr: Smallest ZCR value observed in the audio. It indicates the minimum rate of signal changes (crossings of zero) over time.''')
st.markdown('''max_zcr: Largest ZCR value observed in the audio. It indicates the maximum rate of signal changes over time.''')
st.markdown('''mean_zcr: Represents the average ZCR value over the entire audio. It provides insight into the central tendency of the rate of signal changes.''')
st.markdown('''variance_zcr: Quantifies the spread or dispersion of the ZCR values. A higher variance suggests greater variability in the rate of signal changes over time.''')
st.markdown('''min_fq: Lowest frequency component in the audio. Useful to know the lowest frequency at which a sound or signal is detected.''')
st.markdown('''max_fq: Maximum frequency component in the audio. Useful for detecting high-frequency noise or identifying the upper limit of frequency content in a signal.''')
st.markdown('''mean_fq: Average frequency content in the audio. A higher mean frequency may indicate a signal with a higher pitch or more high-frequency content.''')
st.markdown('''variance_fq: A high frequency variance suggests that the signal contains a wide range of frequencies, whereas a low variance indicates a more concentrated or narrowband frequency content.''')

features_col1, features_col2 = st.columns(2, gap='large')
with features_col1:
     st.subheader("Cat Dataset")
     cats_tmp = pd.DataFrame(cat_df.values, columns=[i for i in cat_df.columns])
     st.write(cats_tmp.head())

with features_col2:
    st.subheader("Dog Dataset")
    dogs_tmp = pd.DataFrame(dog_df.values, columns=[i for i in dog_df.columns])
    st.write(dogs_tmp.head())







# ////////////////////////
# DIMENSION ANALYSIS
# ////////////////////////
st.header('[4] Dimension analysis (40 pts)')
st.caption('''We are using PCA in the dataset to perform a dimension analysis in order to find the number of dimensions by which we might cover at least 85% of cumulative variance.
We are going to include 2D plots to visualize how the dimensions behave after being transformed. 
Then by using LDA we'll repeat the same analysis and transformation over the original data set. Again we'll repeat
the visualization process to see how it performs now the correlation.''')

dimension_a_col1, dimension_a_col2 = st.columns(2, gap='large')
X = dataset.drop(['Animal'], axis="columns")
y = pd.DataFrame(dataset['Animal'].values, columns=['Animal'])
with dimension_a_col1:
     st.caption('''The complete dataset is:''')
     complete_dataset = pd.DataFrame(X.values, columns=[i for i in X.columns])
     st.write(complete_dataset.head())

with dimension_a_col2:
     st.caption('''For this part the first thing is to select the label that we are trying to predict, in our case Dog/Cat.''')
     st.write(y)

st.subheader('Splitting the dataset for future validations')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
split_col1, split_col2, split_col3, split_col4 = st.columns(4, gap='large')
with split_col1:
    st.markdown('X trained data')
    st.write(X_train.head())
with split_col2:
    st.markdown('X test data')
    st.write(X_test.head())
with split_col3:
    st.markdown('y trained data')
    st.write(y_train.head())
with split_col4:
    st.markdown('y test data')
    st.write(y_test.head())

st.subheader('Standard Scaler')
st.caption('''We are going to apply an Standardization Scaler to the X data.''')
sc = StandardScaler()
st.write(sc)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)









# ////////////////////////////////////////////////
# /////////////////// PCA ////////////////////////
# ////////////////////////////////////////////////
st.subheader('PCA')
st.caption('''The Principal Component Analysis it's a powerful tool to analize data and reduce dimensionality. 
           This can aid with the data visualization and modeling.''')
st.caption('''By default we can create PCA for all the components (dimensions) but it is also posible to specify how many components we want to analyse / reduce, also we can send as parameter what is the minimum variance we are looking for to keep.''')
st.caption('''The code to do this is:''')

pca = PCA()
pca_X_train = pca.fit_transform(X_train)
pca_X_test = pca.transform(X_test)

pca_code = '''
     from sklearn.decomposition import PCA
     pca = PCA()
     pca_X_train = pca.fit_transform(X_train)
     pca_X_test = pca.transform(X_test)
'''
st.code(pca_code, language="python", line_numbers=False)
st.caption('''In this example we are using to methods (fit_transform) that performs the analys and transformation of the obervations all in once. the second method /(transform) will only transform the data based on the analysis maded in fir_transofrm method.''')
st.caption('''Let's see how the weights our new dimension is conformed, each component (sorted by variance) will transform each value of the dataset by the following dimension proportions.''')

pca_components = '''
     pd.DataFrame(
          data    = pca.components_,
          columns = X.columns,
          index   = ['PC1', 'PC2', 'PC3', 'PC4','PC5', 'PC6', 'PC7', 'PC8','PC9', 'PC10', 'PC11', 'PC12','PC13', 'PC14', 'PC15', 'PC16', 'PC17', 'PC18', 'PC19']
)'''
st.code(pca_components, language="python", line_numbers=False)
st.caption('''Result:''')
st.write(pd.DataFrame(
     data    = pca.components_,
     columns = X.columns,
     index   = ['PC1', 'PC2', 'PC3', 'PC4','PC5', 'PC6', 'PC7', 'PC8','PC9', 'PC10', 'PC11', 'PC12','PC13', 'PC14', 'PC15', 'PC16']
))
st.caption('''
Plotting the components in a heat map helps to understand how original dimensions influence in the new component. For example in the first component (the one with more variance) Zero Cross rate helps better to express data in terms of variance.
''')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
componentes = pca.components_
plt.imshow(componentes.T, cmap='viridis', aspect='auto')
plt.yticks(range(len(X.columns)), X.columns)
plt.xticks(range(len(X.columns)), np.arange(pca.n_components_) + 1)
plt.grid(False)
plt.colorbar()
st.pyplot(fig)

st.subheader('Variance Ratio (Per Component)')
st.caption('''Another important measure is the variance ratio, this means how much in percentage of variance each component represents, for example component 1 just by it self representes 58%.''')

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
ax.bar(
    x      = np.arange(pca.n_components_) + 1,
    height = pca.explained_variance_ratio_
)

for x, y in zip(np.arange(len(X.columns)) + 1, pca.explained_variance_ratio_):
    label = round(y, 2)
    ax.annotate(
        label,
        (x,y),
        textcoords="offset points",
        xytext=(0,10),
        ha='center'
    )

ax.set_xticks(np.arange(pca.n_components_) + 1)
ax.set_ylim(0, 1.1)
ax.set_title('Variance percentage per component')
ax.set_xlabel('Principal component')
ax.set_ylabel('Variance')
st.pyplot(fig)

st.subheader('Accumulative Variance')
st.caption('''Actually we can acumulate the percentage by including components, if we do it in order , then we might be able to perform an elbow test, this is, what is the minimum number of components needed to capture a minumum percentage of variance in my new dimension space (the elbow).''')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
ax.plot(
    np.arange(len(X.columns)) + 1,
    pca.explained_variance_ratio_.cumsum(),
    marker = 'o'
)

for x, y in zip(np.arange(len(X.columns)) + 1, pca.explained_variance_ratio_.cumsum()):
    label = round(y, 2)
    ax.annotate(
        label,
        (x,y),
        textcoords="offset points",
        xytext=(0,10),
        ha='center'
    )

ax.set_ylim(0, 1.1)
ax.set_xticks(np.arange(pca.n_components_) + 1)
ax.set_title('Acumulative variance')
ax.set_xlabel('Principal component')
ax.set_ylabel('Variance')
st.pyplot(fig)

st.caption('''We can stablish a number of components and apply the fit and transform methods too.''')
pca = PCA(3)
projected = pca.fit_transform(X_train)
st.write('X_train data shape (Rows, Cols):', X_train.data.shape)
st.write('Projected shape (Rows, Cols):', projected.shape)

st.subheader('3D Graph Representation')
Xax = projected[:, 0]
Yax = projected[:, 1]
Zax = projected[:, 2]
labels = y_train

cdict = {"Cat": 'red', "Dog": 'green'}
marker = {"Cat": '*', "Dog": 'o'}
alpha = {"Cat": 0.3, "Dog": 0.5}

# Create 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
fig.patch.set_facecolor('white')

for l in np.unique(labels):
    ix = np.where(labels == l)
    ax.scatter(Xax[[ix]], Yax[[ix]], Zax[[ix]], c=cdict[l], label=l, s=40, marker=marker[l], alpha=alpha[l])

plt.xlabel("First Principal Component", fontsize=14)
plt.ylabel("Second Principal Component", fontsize=14)

# Display the plot using Streamlit
st.pyplot(fig)







# ////////////////////////////////////////////////
# /////////////////// LDA ////////////////////////
# ////////////////////////////////////////////////
st.subheader('Linear Discriminant Analysis')
lda = LDA(n_components=1)
X_train = lda.fit_transform(X_train, y_train)
y_predict = lda.predict(X_test)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
fig.patch.set_facecolor('white')
for l in np.unique(labels):
    ix=np.where(labels==l)
    ax.scatter(Xax[[ix]],ys=0,c=cdict[l],label=l,s=40,marker=marker[l],alpha=alpha[l])
st.pyplot(fig)
# ////////////// CONFUSION MATRIX ////////////////
# ////////////////////////////////////////////////
st.subheader('Confusion Matrix of LDA')
confusion_matrix = metrics.confusion_matrix(y_test, y_predict)
accuracy = metrics.accuracy_score(y_test, y_predict)

# Display the accuracy score
st.write(f"Accuracy: {accuracy}")

# Display the confusion matrix using ConfusionMatrixDisplay
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
disp.plot(cmap='Blues', ax=ax)
plt.xlabel("Predicted Labels", fontsize=14)
plt.ylabel("True Labels", fontsize=14)
# Display the plot using Streamlit
st.pyplot(fig)




