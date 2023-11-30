import os
import glob
import annoy
import librosa
from tqdm import tqdm
import numpy as np
from python_speech_features import mfcc, fbank, logfbank
import pickle

data_dir = './nhaccuatoi'

def extract_features(y, sr=16000, nfilt=10, winsteps=0.02):
    try:
        feat = mfcc(y, sr, nfilt=nfilt, winstep=winsteps)
        return feat
    except:
        raise Exception("Extraction feature error")

def crop_feature(feat, i=0, nb_step=10, maxlen=100):
    crop_feat = np.array(feat[i : i + nb_step]).flatten()
    print(crop_feat.shape)
    crop_feat = np.pad(crop_feat, (0, maxlen - len(crop_feat)), mode='constant')
    return crop_feat

features = []
songs = []

for song in tqdm(os.listdir(data_dir)):
    song_path = os.path.join(data_dir, song)
    y, sr = librosa.load(song_path, sr=16000)
    feat = extract_features(y)
    for i in range(0, feat.shape[0] - 10, 5):
        crop_feat = crop_feature(feat, i, nb_step=10)
        features.append(crop_feat)
        songs.append(song_path)

pickle.dump(features, open('features.pk', 'wb'))
pickle.dump(songs, open('songs.pk', 'wb'))

f = 100
t = annoy.AnnoyIndex(f)

for i, v in enumerate(features):
    t.add_item(i, v)

t.build(100) # 100 trees
t.save('music.ann')

song_path = os.path.join('CUT/b1.mp3')
y, sr = librosa.load(song_path, sr=16000)
feat = extract_features(y)

results = []
for i in range(0, feat.shape[0], 10):
    crop_feat = crop_feature(feat, i, nb_step=10)
    result = t.get_nns_by_vector(crop_feat, n=5)
    result_songs = [songs[k] for k in result]
    results.append(result_songs)

results = np.array(results).flatten()

from collections import Counter

most_song = Counter(results)
most_common_songs = most_song.most_common()
print(most_common_songs)