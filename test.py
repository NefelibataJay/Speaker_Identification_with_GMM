from modeltraining import *

if __name__ == "__main__":
    test_file1="D:\\kust\\419\\Speaker_Identification_GMM\\testdata\\"
    test_file="D:\\kust\\419\\Speaker_Identification_GMM\\testdata\\dongT.wav"
    train_file="D:\\kust\\419\\Speaker_Identification_GMM\\trainData\\"
    
    model = Model()
    model.training(train_file,1)
    
    result = model.predict(test_file1)
    
    print(result)
    
