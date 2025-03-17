# ECG5000 Time Series Prediction with Chronos

This repository contains a Google Colab notebook that demonstrates time series forecasting on the ECG5000 dataset using Amazon's Chronos T5-large model. The script predicts the next 140-timestep heartbeat sequence based on a context of 5 previous rows (700 timesteps).

## Overview
- **Dataset**: ECG5000 training data (`ECG5000_TRAIN.txt`), consisting of 500 rows of 140-timestep ECG heartbeats.
- **Model**: Chronos T5-large, a pretrained transformer-based time series forecasting model from Amazon, run on CUDA with bfloat16 precision.
- **Task**: Predict the next full row (140 timesteps) using 5 prior rows as context.
- **Preprocessing**: Normalizes data to [0, 1] range for model input.
- **Evaluation**: Computes Mean Absolute Error (MAE) on both normalized and original scales.
- **Visualization**: Plots the context, true next row, median forecast, and 80% prediction interval on the original scale using Matplotlib.

## Key Features
- Loads and parses ECG5000 data, excluding labels to focus on time series prediction.
- Flattens 5 rows (700 timesteps) as context to forecast the next 140 timesteps.
- Uses Chronos to generate probabilistic forecasts and extracts 10th, 50th (median), and 90th percentiles.
- Denormalizes predictions for comparison and visualization on the original ECG scale.
- Includes error metrics (MAE) and a detailed plot for visual inspection.

## Requirements
- Python 3.x
- Libraries: `matplotlib`, `numpy`, `torch`, `chronos` (via Hugging Face)
- Hardware: CUDA-enabled GPU recommended for Chronos model
- Data: `ECG5000_TRAIN.txt` (available from the UCR Time Series Archive)

## Usage
1. Upload `ECG5000_TRAIN.txt` to your Colab environment.
2. Run the notebook to load the Chronos pipeline, process the data, and generate predictions.
3. View the MAE results and visualization output.

## Results
The script outputs:
- MAE on normalized scale (e.g., 0.XXXX).
- MAE on original scale (e.g., X.XXXX).
- A plot showing the 700-timestep context, true next row, and forecasted row with confidence intervals.

## Notes
- The ECG5000 dataset is originally designed for classification, but this script repurposes it for forecasting by treating rows as sequential data.
- Adjust `n_context_rows` or `prediction_length` to experiment with different context sizes or forecast horizons.

Feel free to explore, modify, or extend this code for other time series datasets!
