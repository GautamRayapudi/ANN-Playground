import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Dense
from keras.regularizers import L1, L2, L1L2
from mlxtend.plotting import plot_decision_regions
from keras.callbacks import Callback

std = StandardScaler()

base_url = "https://api.github.com/repos/GautamRayapudi/ANN-Playground/contents/data"
raw_base_url = "https://raw.githubusercontent.com/GautamRayapudi/ANN-Playground/main/data"

# Function to list files in the 'data' directory
@st.cache_data
def list_files():
    response = requests.get(base_url)
    if response.status_code == 200:
        files = response.json()
        return [file['name'] for file in files if file['type'] == 'file']
    else:
        st.error("Error fetching file list from GitHub")
        return []

# Function to load the selected CSV file
def load_data(file_name):
    file_url = f"{raw_base_url}/{file_name}"
    return pd.read_csv(file_url)

# Set up the Streamlit layout
st.title("ANN Playground")
# Load the dataset
st.sidebar.header("Dataset Selection")
available_datasets = list_files()
dataset_index = st.sidebar.selectbox("Select the type of dataset", range(len(available_datasets)), format_func=lambda x: available_datasets[x])
data_file = available_datasets[dataset_index]

df = load_data(data_file)
if df is None:
    st.stop()

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
n_classes = len(np.unique(y))

# Model parameters
st.sidebar.header("Model Parameters")
test_size = st.sidebar.slider("Test data ratio", 10, 90, 20) / 100.0
regularizer_choice = st.sidebar.selectbox("Regularizer", ['None', 'L1', 'L2', 'Elastic Net'])
batch = st.sidebar.slider("Batch size", 1, 30, 10)
epoch = st.sidebar.number_input("Number of epochs", 1, 100, 10)
hidden_layers = st.sidebar.number_input("Number of hidden layers", 1)

# Set regularizer
if regularizer_choice == 'L1':
    regularizer = L1(l1=0.01)
elif regularizer_choice == 'L2':
    regularizer = L2(l2=0.01)
elif regularizer_choice == 'Elastic Net':
    regularizer = L1L2(l1=0.01, l2=0.01)
else:
    regularizer = None

# Hidden layer sizes
hidden_sizes = []
activation_list = []
for i in range(hidden_layers):
    hidden_size = st.sidebar.number_input(f"Neurons in hidden layer {i + 1}", 1, 10, 1)
    active_hidden = st.sidebar.selectbox(f"Activation function for hidden layer {i+1}", ['sigmoid', 'tanh','relu'])
    hidden_sizes.append(hidden_size)
    activation_list.append(active_hidden)

active_output = st.sidebar.selectbox("Activation function for output layer", ['sigmoid', 'linear', 'softmax'])    

# Initialize session state for model and history
if 'model' not in st.session_state:
    st.session_state.model = None

if 'history' not in st.session_state:
    st.session_state.history = None

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=23)

input_shape = (X.shape[1],)

# Build the model using the functional API
def build_model():
    inputs = Input(shape=input_shape)
    x = inputs
    for size, active in zip(hidden_sizes, activation_list):
        x = Dense(units=size, activation=active, kernel_regularizer=regularizer)(x)
    
    if n_classes > 2:
        outputs = Dense(units=n_classes, activation=active_output, kernel_regularizer=regularizer)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
        y_train_encoded = std.fit_transform(y_train.values.reshape(-1, 1))
    else:
        outputs = Dense(units=1, activation=active_output, kernel_regularizer=regularizer)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        y_train_encoded = std.fit_transform(y_train.values.reshape(-1, 1))

    return model, y_train_encoded

# Custom callback to update progress bar
class ProgressBarCallback(Callback):
    def __init__(self, progress_bar, epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.progress((epoch + 1) / self.epochs)

# Button to train the model
if st.sidebar.button("Train Model"):
    progress_bar = st.progress(0)
    model, y_train_encoded = build_model()
    model.summary(print_fn=lambda x: st.text(x))

    start_time = time.time()
    history = model.fit(X_train, y_train_encoded, validation_split=0.2, epochs=epoch, batch_size=batch, callbacks=[ProgressBarCallback(progress_bar, epoch)])
    end_time = time.time()
    train_time = end_time - start_time

    st.session_state.model = model
    st.session_state.history = history.history

    st.write(f"Model training complete! Training time: {train_time:.2f} seconds")

# Plot training history
if st.session_state.history is not None and st.sidebar.button("Plot Training History"):
    st.subheader("Training History")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].plot(st.session_state.history['accuracy'], label='Train Accuracy')
    ax[0].plot(st.session_state.history['val_accuracy'], label='Validation Accuracy')
    ax[0].set_title('Model Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()

    ax[1].plot(st.session_state.history['loss'], label='Train Loss')
    ax[1].plot(st.session_state.history['val_loss'], label='Validation Loss')
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    st.pyplot(fig)

# Intermediate layer and neuron selection
if st.session_state.model is not None:
    st.sidebar.header("Intermediate Layer Analysis")
    intermediate_layer_index = st.sidebar.number_input("Intermediate layer index (0-based)", 0, len(st.session_state.model.layers) - 1, 0)
    neuron_index = st.sidebar.number_input("Neuron index in selected layer", 0, hidden_sizes[intermediate_layer_index] - 1, 0)

    # Plot decision regions
    if st.sidebar.button("Plot Decision Regions"):
        if n_classes <= 2:
            with st.spinner("Plotting decision regions..."):
                # Create a model for intermediate output
                intermediate_layer_model = Model(inputs=st.session_state.model.input, outputs=st.session_state.model.layers[intermediate_layer_index].output)

                # Define a wrapper class for the intermediate model
                class IntermediateClassifier:
                    def __init__(self, model, neuron_index):
                        self.model = model
                        self.neuron_index = neuron_index

                    def predict(self, X):
                        intermediate_output = self.model.predict(X)
                        return (intermediate_output[:, self.neuron_index] > 0.5).astype(int)

                intermediate_clf = IntermediateClassifier(intermediate_layer_model, neuron_index)

                st.subheader("Decision Regions")
                fig, ax = plt.subplots()
                plot_decision_regions(X_test.to_numpy(), y_test.to_numpy().astype(int), clf=intermediate_clf, legend=2)
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')
                plt.title('Decision Regions')
                st.pyplot(fig)
        else:
            with st.spinner("Plotting decision regions..."):
                # Create a model for intermediate output
                intermediate_layer_model = Model(inputs=st.session_state.model.input, outputs=st.session_state.model.layers[intermediate_layer_index].output)

                # Define a wrapper class for the intermediate model
                class IntermediateClassifier:
                    def __init__(self, model, neuron_index):
                        self.model = model
                        self.neuron_index = neuron_index

                    def predict(self, X):
                        intermediate_output = self.model.predict(X)
                        return (intermediate_output[:, self.neuron_index] > 0).astype(int)

                intermediate_clf = IntermediateClassifier(intermediate_layer_model, neuron_index)

                st.subheader("Decision Regions")
                fig, ax = plt.subplots()
                plot_decision_regions(X_test.to_numpy(), y_test.to_numpy().astype(int), clf=intermediate_clf, legend=2)
                plt.xlabel('Feature 1')
                plt.ylabel('Feature 2')
                plt.title('Decision Regions')
                st.pyplot(fig)   
