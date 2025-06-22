import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
from dotenv import load_dotenv
load_dotenv()
from constants import (
    activation,
    dropout_rate,
    optimizer,
    input_dim,
    epochs,
    batch_size,
    model_name
)
from logger import get_logger

mlflow.set_tracking_uri("https://dagshub.com/MavinashR88/AutoTaxAI-End-to-End-Tax-Prediction-with-MLflow-DVC-and-Docker.mlflow")
mlflow.set_experiment("Model")


class ANNPipeline:
    def __init__(self, data_dir="data/external"):
        self.logger = get_logger()
        self.data_dir = data_dir
        self.model = None

    def load_data(self):
        try:
            self.logger.info(f"Loading data from {self.data_dir}")
            self.X_train = pd.read_csv(os.path.join(self.data_dir, "X_train.csv"))
            self.X_test = pd.read_csv(os.path.join(self.data_dir, "X_test.csv"))
            self.y_train = pd.read_csv(os.path.join(self.data_dir, "y_train.csv"))
            self.y_test = pd.read_csv(os.path.join(self.data_dir, "y_test.csv"))
            self.logger.info("Data loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def build_model(self):
        try:
            self.logger.info("Building ANN model...")

            model = models.Sequential()
            model.add(layers.InputLayer(input_shape=(input_dim,)))
            model.add(layers.Dense(64, activation=activation))
            model.add(layers.Dropout(dropout_rate))
            model.add(layers.Dense(32, activation=activation))
            model.add(layers.Dropout(dropout_rate))
            model.add(layers.Dense(1))

            model.compile(
                optimizer=optimizer,
                loss='mean_squared_error',
                metrics=['mae']
            )

            self.model = model
            self.logger.info("Model built and compiled successfully.")
        except Exception as e:
            self.logger.error(f"Error building model: {e}")
            raise

    def train_model(self):
        try:
            self.logger.info("Starting training...")
            early_stop = EarlyStopping(
            monitor='val_loss',   # You can also monitor 'val_mae'
            patience=5,           # Stop after 5 epochs of no improvement
            restore_best_weights=True,
            verbose=1
        )
            self.history = self.model.fit(
                self.X_train,
                self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=1,
                callbacks=[early_stop]
            )
            self.logger.info("Model training complete.")
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise

    def evaluate_model(self):
        try:
            self.logger.info("Evaluating model on test data...")
            loss, mae = self.model.evaluate(self.X_test, self.y_test, verbose=0)
            self.logger.info(f"Test Loss: {loss:.4f} | Test MAE: {mae:.4f}")
            return loss, mae
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            raise

    def save_model(self, save_dir="models"):
        try:
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, model_name)
            self.model.save(model_path)
            self.logger.info(f"Model saved at {model_path}")
            return model_path
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise

    def plot_training_history(self):
        try:
            self.logger.info("Plotting training history...")

            plt.figure(figsize=(12, 5))

            # Plot Loss
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['loss'], label='Train Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.title('Loss Over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Loss (MSE)')
            plt.legend()

            # Plot MAE
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['mae'], label='Train MAE')
            plt.plot(self.history.history['val_mae'], label='Validation MAE')
            plt.title('MAE Over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Mean Absolute Error')
            plt.legend()

            plt.tight_layout()

            plots_dir = "plots"
            os.makedirs(plots_dir, exist_ok=True)
            plot_path = os.path.join(plots_dir, "training_history.png")
            plt.savefig(plot_path)
            plt.close()

            self.logger.info(f"Training history plot saved at {plot_path}")
            return plot_path

        except Exception as e:
            self.logger.error(f"Error plotting training history: {e}")
            raise

    def log_to_mlflow(self, loss, mae, model_path, plot_path):
        try:
            self.logger.info("Logging experiment to MLflow...")

            with mlflow.start_run(run_name="ANN Regression Pipeline"):
                # Log parameters
                mlflow.log_param("activation", activation)
                mlflow.log_param("dropout_rate", dropout_rate)
                mlflow.log_param("optimizer", optimizer)
                mlflow.log_param("input_dim", input_dim)
                mlflow.log_param("epochs", epochs)
                mlflow.log_param("batch_size", batch_size)

                # Log metrics
                mlflow.log_metric("loss", loss)
                mlflow.log_metric("mae", mae)

                # Log model
                # mlflow.tensorflow.log_model(
                #     tf_saved_model_dir=model_path,
                #     artifact_path="model"
                # )

                # Log plot artifact
                mlflow.log_artifact(plot_path, artifact_path="plots")

            self.logger.info("MLflow logging complete.")

        except Exception as e:
            self.logger.error(f"Error during MLflow logging: {e}")
            raise

    def run_pipeline(self):
        try:
            self.logger.info("Running ANN model pipeline with MLflow...")
            self.load_data()
            self.build_model()
            self.train_model()
            loss, mae = self.evaluate_model()
            model_path = self.save_model()
            plot_path = self.plot_training_history()
            self.log_to_mlflow(loss, mae, model_path, plot_path)
            self.logger.info("Full pipeline with MLflow completed successfully.")
        except Exception as e:
            self.logger.critical(f"Pipeline failed: {e}")
            raise