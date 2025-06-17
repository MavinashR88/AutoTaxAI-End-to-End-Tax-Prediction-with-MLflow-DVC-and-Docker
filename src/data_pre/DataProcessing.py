import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from constants import test_size, target, random_state
from logger import get_logger
import joblib

class Preprocessing:
    def __init__(self, path):
        self.logger = get_logger()
        self.path = path
        self.scaler = StandardScaler()
        
        # # Folder paths
        # self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        # self.raw_dir = os.path.join(self.base_dir, 'raw')
        # self.interim_dir = os.path.join(self.base_dir, 'interim')
        # self.processed_dir = os.path.join(self.base_dir, 'processed')

        # # Create folders if not exist
        # os.makedirs(self.interim_dir, exist_ok=True)
        # os.makedirs(self.processed_dir, exist_ok=True)

        try:
            self.data = pd.read_csv(self.path)
            self.logger.info(f"Data loaded from {self.path}")
        except Exception as e:
            self.logger.error(f"Error loading data from {self.path}: {e}")
            raise

    def preprocess(self):
        try:
            self.logger.info("Starting preprocessing...")
            self.data.dropna(inplace=True)

            if target not in self.data.columns:
                raise ValueError(f"Target column '{target}' not found in dataset.")

            X = self.data.drop(columns=[target])
            y = self.data[target]


            # Step 1: Ensure the directory exists
            processed_dir = os.path.join("data", "processed")
            os.makedirs(processed_dir, exist_ok=True)

            # Step 2: Define the file name
            processed_filename = "preprocessed_data.csv"

            # Step 3: Join to get full path
            processed_path = os.path.join(processed_dir, processed_filename)

            # Step 4: Save the DataFrame
            self.data.to_csv(processed_path, index=False)

            # Step 5: Log the info
            self.logger.info(f"Processed data saved to: {processed_path}")


            # Save X and y to interim
            interim_dir = os.path.join("data", "interim")
            os.makedirs(interim_dir, exist_ok=True)

            # Step 2: Create full file paths
            X_path = os.path.join(interim_dir, "X.csv")
            y_path = os.path.join(interim_dir, "y.csv")

            # Step 3: Save the data
            X.to_csv(X_path, index=False)
            y.to_frame().to_csv(y_path, index=False)

            # Step 4: Log the action
            self.logger.info(f"X saved to: {X_path}")
            self.logger.info(f"y saved to: {y_path}")

            # Step 5: Return values
            return X, y
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {e}")
            raise

    def split_data(self, X, y):
        try:
            self.logger.info("Splitting data into train and test sets...")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            self.logger.info("Data splitting complete.")
        except Exception as e:
            self.logger.error(f"Error during train-test split: {e}")
            raise

    def scale_data(self):
        try:
            self.logger.info("Scaling data...")
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)

            # Save scaler to models
            models_dir = os.path.join("models")
            os.makedirs(models_dir, exist_ok=True)

            # Step 2: Define the full path to save the scaler
            scaler_path = os.path.join(models_dir, "scaler.pkl")

            # Step 3: Save the scaler using joblib
            joblib.dump(self.scaler, scaler_path)

            # Step 4: Log the success
            self.logger.info(f"Scaler saved to: {scaler_path}")
        except Exception as e:
            self.logger.error(f"Error scaling data: {e}")
            raise

    def run_pipeline(self):
        try:
            self.logger.info("Running full preprocessing pipeline...")
            X, y = self.preprocess()
            self.split_data(X, y)
            self.scale_data()
            self.logger.info("Pipeline executed successfully.")

            return {
                'X_train': self.X_train_scaled,
                'X_test': self.X_test_scaled,
                'y_train': self.y_train,
                'y_test': self.y_test
            }
        except Exception as e:
            self.logger.critical(f"Pipeline failed: {e}")
            raise
