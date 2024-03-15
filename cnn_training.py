# %% import packages

# helper libraries
import numpy as np

# import existing packages
import tensorflow as tf

# %% Inference functions

def predict_with_hidden(model, inputs, output_hidden=False):
    # WARNING: largely copied from ChatGPT (bug fixed)
    """
    Predicts using the given model and optionally returns hidden layer outputs.
    
    Args:
    - model: The Keras model to use for prediction.
    - inputs: Input data to the model.
    - output_hidden: If True, returns hidden layer outputs along with predictions.
    
    Returns:
    - If output_hidden is False, returns predictions only.
    - If output_hidden is True, returns a tuple containing predictions and hidden layer outputs.
    """
    if not output_hidden:
        return model.predict(inputs)
    
    # Get the outputs of all layers up to the last hidden layer
    hidden_outputs = []
    current_input = inputs
    for layer in model.layers:
        print(current_input.shape)
        current_output = layer(current_input)
        hidden_outputs.append(current_output)
        current_input = current_output
    
    # final output is the conventional prediction
    predictions = current_output
    
    return predictions, hidden_outputs