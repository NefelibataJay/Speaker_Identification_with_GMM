import os
import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc
import pickle
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as GMM


class Model:
    def __init__(self):
        self.gmm = "./GMM/"
        self.train_file_txt = "./trainData.txt"
        self.test_file_txt = "./testData.txt"

    def init_data(self):
        file_list = os.listdir(self.train_path)
        k = 1
        fp = open(self.train_file_txt, mode='w', encoding='utf-8')
        for name in file_list:
            list_name = os.listdir(self.train_path + name)
            if(len(list_name)< self.size):
                raise ValueError("The train_size is smaller than the actual number of files")
            for file_name in list_name:
                n = name + '/' + file_name
                fp.write(n + '\n')
                k += 1
                if k > self.size:
                    k = 1
                    break
        fp.close()

    @staticmethod
    def calculate_delta(array):
        """Calculate and returns the delta of given feature vector matrix"""
        rows, cols = array.shape
        deltas = np.zeros((rows, 20))
        N = 2
        for i in range(rows):
            index = []
            j = 1
            while j <= N:
                if i - j < 0:
                    first = 0
                else:
                    first = i - j
                if i + j > rows - 1:
                    second = rows - 1
                else:
                    second = i + j
                index.append((second, first))
                j += 1
            deltas[i] = (array[index[0][0]] - array[index[0][1]] +
                         (2 * (array[index[1][0]] - array[index[1][1]]))) / 10
        return deltas

    def extract_features(self, audio, rate):
        """extract 20 dim mfcc features from an audio, performs CMS and combines
        delta to make it 40 dim feature vector"""
        mfcc_feature = mfcc.mfcc(
            audio, rate, 0.025, 0.01, 20, nfft=1200, appendEnergy=True)
        mfcc_feature = preprocessing.scale(mfcc_feature)
        delta = self.calculate_delta(mfcc_feature)
        combined = np.hstack((mfcc_feature, delta))
        return combined

    def training(self,train_path,size='5'):
        """
        params: train_path 
                size 
        """
        self.train_path = train_path
                
        self.label = os.listdir(self.train_path)
        
        self.size = int(size)
        
        self.init_data()
        
        count = 1
        # Extracting features for each speaker (5 files per speakers)
        file_paths = open(self.train_file_txt, 'r')
        features = np.asarray(())
        for path in file_paths:
            path = path.strip()

            # read the audio
            sr, audio = read(self.train_path + path)

            # extract 40 dimensional MFCC & delta MFCC features
            vector = self.extract_features(audio, sr)

            if features.size == 0:
                features = vector
            else:
                features = np.vstack((features, vector))
            
            if count == self.size:
                gmm = GMM(n_components=16, max_iter=200,
                          covariance_type='diag', n_init=3)
                gmm.fit(features)

                # dumping the trained gaussian model
                picklefile = path.split("/")[0] + ".gmm"

                pickle.dump(gmm, open(self.gmm + picklefile, 'wb+'))

                print('+ modeling completed for speaker:', picklefile,
                      " with data point = ", features.shape)
                features = np.asarray(())
                count = 0
            count = count + 1
    
    def predict_one(self,test_file):
        """对单个文件进行预测"""
        self.test_file = test_file
        name = test_file.split('/')
        name = name[len(name)-1]
        
        modelpath = self.gmm

        gmm_files = [os.path.join(modelpath, fname) for fname in
                   os.listdir(modelpath) if fname.endswith('.gmm')]

        # Load the Gaussian gender Models
        models = [pickle.load(open(fname, 'rb+')) for fname in gmm_files]
        speakers = [fname.split("/")[-1].split(".gmm")[0] for fname
                in gmm_files]

        # error = 0

        test_file_txt = "testData.txt"

        # init testData.txt
        # file_list = os.listdir(test_path)
        fp = open(test_file_txt, mode='w', encoding='utf-8')
        # for name in file_list:
        fp.write(name + '\n')
        fp.close()
        
        
        sr, audio = read(self.test_file)
        
        vector = self.extract_features(audio, sr)
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            gmm = models[i]  # checking with each model one by one
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()
        winner = np.argmax(log_likelihood)
        
        return speakers[winner]
            
    def predict(self, test_path):
        """
        对路径下所有文件预测
        """
        self.test_path = os.getcwd() + test_path
        
        # self.training()

        # path where training speakers will be saved
        modelpath = self.gmm

        gmm_files = [os.path.join(modelpath, fname) for fname in
                   os.listdir(modelpath) if fname.endswith('.gmm')]

        # Load the Gaussian gender Models
        models = [pickle.load(open(fname, 'rb+')) for fname in gmm_files]
        speakers = [fname.split("/")[-1].split(".gmm")[0] for fname
                in gmm_files]

        # error = 0
        total_sample = 0.0

        test_file_txt = "testData.txt"

        # init testData.txt
        file_list = os.listdir(test_path)
        fp = open(test_file_txt, mode='w', encoding='utf-8')
        for name in file_list:
            fp.write(name + '\n')
        fp.close()
        
        result = {}

        file_paths = open(test_file_txt, 'r')
        # Read the test directory and get the list of test audio files
        for path in file_paths:
            total_sample += 1.0
            path = path.strip()
            sr, audio = read(test_path + path)
            vector = self.extract_features(audio, sr)
            log_likelihood = np.zeros(len(models))
            for i in range(len(models)):
                gmm = models[i]  # checking with each model one by one
                scores = np.array(gmm.score(vector))
                log_likelihood[i] = scores.sum()
            winner = np.argmax(log_likelihood)
            result[path] = speakers[winner]
            
        return result
            
