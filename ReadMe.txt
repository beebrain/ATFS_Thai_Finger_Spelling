<h> This is a Automatic Thai Finger language Spelling Transcription. </h>
1. Introduction
This is my implementation of ATFS in python. 
The ATFS has 3 part, e.g., Alphabet-separattion state(ALS), Sign recognition state(SR), and Sign Sequence classification state(SSC).
I will added each source code state later.


This state need to have the weight  trained of CNN from SR, the weight trained of LSTM from SSC, and sign frame marked from ALS.

2. Requirements
2.1 The weight of CNN push in folder Model_CNN
2.2 The weight of LSTM push in folder LSTM_CNN_****
Note that, **** mean the feature type input of LSTM. We have 4 althernatives. i.e., LSTM_CNN_Class, LSTM_CNN_Feature, LSTM_CNN_Onehot, and LSTM_CNN_Permultimate.
2.3 The sign frame marked data push in folder Distance_centroidData
Note that, we have 5 technique to marke sign frame
  1. Test_Distance folder is a D1. The D1 uses sigle threshold to mark a sign frame.
  2. Test_DistanceV2 folder is a D2. The D2 uses double threshold to mark a sign frame.
  3. Test_DistanceV2MAV folder is a D2M. The D2M uses double threshold with moving average to smooth the curve to mark a sign frame.
  4. Test_DistanceV2Skip folder is a D2S. The D2M uses double threshold to mark a sign frame and add skip frame technique to mark a curve is smoothed.
  5. Test_HeatMap folder is HM. The HM uses the Gaussian function to mark a sign frame.

tensorflow >= 1.8.0 (theoretically any version that supports tf.data is ok)
opencv-python
tqdm
3. Weights convertion
