import os
import sys
import pandas as pd
from dataclasses import dataclass
from logger import logger
from exception import CustomException

@dataclass
class DataIngestionConfig:
    cleaned_data_path: str = os.path.join('artifacts', 'music_cleaned_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Starting data ingestion using pre-cleaned file...")

        try:
            # Read the already cleaned dataset
            df = pd.read_csv('notebook/cleaned_data.csv')
            logger.info(f"Data loaded successfully. Initial shape: {df.shape}")

            # --- Null handling as final cleanup step ---
            text_cols = ['text', 'Genre_str', 'Artist(s)', 'song']
            numeric_cols = [
                'Tempo', 'Loudness (db)', 'Energy', 'Danceability', 'Positiveness',
                'Speechiness', 'Liveness', 'Acousticness', 'Instrumentalness'
            ]
            boolean_cols = [col for col in df.columns if col.startswith('Good for')]

            for col in text_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].mode()[0])

            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())

            for col in boolean_cols:
                df[col] = df[col].fillna(df[col].mode()[0])

            logger.info(f"Final shape after null handling: {df.shape}")

            # Save to artifacts
            os.makedirs(os.path.dirname(self.ingestion_config.cleaned_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.cleaned_data_path, index=False)
            logger.info(f"Cleaned data saved to: {self.ingestion_config.cleaned_data_path}")

            return self.ingestion_config.cleaned_data_path

        except Exception as e:
            raise CustomException(e, sys)
