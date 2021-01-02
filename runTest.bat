CHCP 65001



FOR /L %%A IN (1,1,5) DO (
	python LSTM_Test.py --hidden_size 2 --rep %%A --MethodCut "Test_Distance"
	python LSTM_Test.py --hidden_size 28 --rep %%A --MethodCut "Test_Distance"
	python LSTM_Test.py --hidden_size 56 --rep %%A --MethodCut "Test_Distance"
	python LSTM_Test.py --hidden_size 128 --rep %%A --MethodCut "Test_Distance"
	python LSTM_Test.py --hidden_size 256 --rep %%A --MethodCut "Test_Distance"


	python LSTM_Test.py --hidden_size 2 --rep %%A --MethodCut "Test_DistanceV2"
	python LSTM_Test.py --hidden_size 28 --rep %%A --MethodCut "Test_DistanceV2"
	python LSTM_Test.py --hidden_size 56 --rep %%A --MethodCut "Test_DistanceV2"
	python LSTM_Test.py --hidden_size 128 --rep %%A --MethodCut "Test_DistanceV2"
	python LSTM_Test.py --hidden_size 256 --rep %%A --MethodCut "Test_DistanceV2"
	
		python LSTM_Test.py --hidden_size 2 --rep %%A --MethodCut "Test_DistanceV2MAV"
	python LSTM_Test.py --hidden_size 28 --rep %%A --MethodCut "Test_DistanceV2MAV"
	python LSTM_Test.py --hidden_size 56 --rep %%A --MethodCut "Test_DistanceV2MAV"
	python LSTM_Test.py --hidden_size 128 --rep %%A --MethodCut "Test_DistanceV2MAV"
	python LSTM_Test.py --hidden_size 256 --rep %%A --MethodCut "Test_DistanceV2MAV"
	
		python LSTM_Test.py --hidden_size 2 --rep %%A --MethodCut "Test_DistanceV2MAV"
	python LSTM_Test.py --hidden_size 28 --rep %%A --MethodCut "Test_DistanceV2MAV"
	python LSTM_Test.py --hidden_size 56 --rep %%A --MethodCut "Test_DistanceV2MAV"
	python LSTM_Test.py --hidden_size 128 --rep %%A --MethodCut "Test_DistanceV2MAV"
	python LSTM_Test.py --hidden_size 256 --rep %%A --MethodCut "Test_DistanceV2MAV"
	
		python LSTM_Test.py --hidden_size 2 --rep %%A --MethodCut "Test_DistanceV2Skip"
	python LSTM_Test.py --hidden_size 28 --rep %%A --MethodCut "Test_DistanceV2Skip"
	python LSTM_Test.py --hidden_size 56 --rep %%A --MethodCut "Test_DistanceV2Skip"
	python LSTM_Test.py --hidden_size 128 --rep %%A --MethodCut "Test_DistanceV2Skip"
	python LSTM_Test.py --hidden_size 256 --rep %%A --MethodCut "Test_DistanceV2Skip"


		python LSTM_Test.py --hidden_size 2 --rep %%A --MethodCut "Test_HeatMap"
	python LSTM_Test.py --hidden_size 28 --rep %%A --MethodCut "Test_HeatMap"
	python LSTM_Test.py --hidden_size 56 --rep %%A --MethodCut "Test_HeatMap"
	python LSTM_Test.py --hidden_size 128 --rep %%A --MethodCut "Test_HeatMap"
	python LSTM_Test.py --hidden_size 256 --rep %%A --MethodCut "Test_HeatMap"
)


