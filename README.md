# LSTM-Neural-Networks
### Step by step instructions for setting up a conda development environment: ###
1) `Download this repository using git clone or similar method:`
   git clone https://github.com/EPantelaios/LSTM-Neural-Networks.git 
2) `Install Anaconda from official source based on your OS from here:`
   https://www.anaconda.com/products/individual 
3) `Run this command to setup the conda environment:`
   conda env create --file project3_env.yml
4) `Run programs in conda environment with:`
   a)   > python forecast.py -d nasdaq2007_17.csv -n 10
   b)   > python detect.py -d nasdaq2007_17.csv -n 10 -m 0.5
   c)   > python reduce.py -d nasdaq_input.csv -q nasdaq_query.csv

<br>
a) Using LSTM neural network for foracasting timeseries <br>
b) Using LSTM neural network for timeseries anomaly detection <br>
c) Using convolutional neural network autoencoder for dimensionality reduction of timeseries <br>

There are pre-trained models for each folder and are ready to be used <br>
In this example NASDAQ share prices are used as timeseries. <br>
Both .ipynb and .py files are included. <br>
