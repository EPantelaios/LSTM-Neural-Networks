# LSTM Neural Networks
#### Step by step instructions for setting up a conda development environment: ####
1) **Download this repository using git clone or similar method:**<br>
   `git clone https://github.com/EPantelaios/LSTM-Neural-Networks.git`
2) **Install Anaconda from official source based on your OS:**<br>
   [https://www.anaconda.com/products/individual](https://www.anaconda.com/products/individual){:target="_blank"}
3) **Run this command to setup the conda environment:**<br>
   `conda env create --file project3_env.yml`
4) **Run programs in conda environment with:**<br>
   - `python forecast.py -d nasdaq2007_17.csv -n 10`<br>
   - `python detect.py -d nasdaq2007_17.csv -n 10 -m 0.5`<br>
   - `python reduce.py -d nasdaq_input.csv -q nasdaq_query.csv`


#### In the respective folders there are the following implementations: ####
1) Using LSTM neural network for foracasting timeseries
2) Using LSTM neural network for timeseries anomaly detection
3) Using convolutional neural network autoencoder for dimensionality reduction of timeseries 


#### Quick Notes: ####
- There are pre-trained models for each folder and are ready to be used 
- In this example NASDAQ share prices are used as timeseries.
- Both .ipynb and .py files are included.
