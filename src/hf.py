import os
import dotenv
from typing import Optional
from pandas import DataFrame
from functools import lru_cache
from huggingface_hub import login
from datasets import load_dataset, Dataset


@lru_cache(maxsize=1)
def _load_env() -> None:
    """
    Ensure .env variables are loaded exactly once.
    """

    dotenv.load_dotenv(override=True)
    
    
def hf_login() -> None:
    """
    Authenticate with Hugging Face using a token stored in environment variables.
    """

    # Ensure environment variables are loaded
    _load_env()

    # Extract the hugging face token from the user data
    HF_TOKEN = os.getenv('HF_TOKEN')
    
    # Check if the HF token has been provided
    if not HF_TOKEN:
        # Raise an exception if the HF token was not provided
        raise Exception("Token is not set. Please save the token first.")
    
    # Authenticate with hugging face
    login(HF_TOKEN)
    
    # Login successful
    print("Successfully logged in to Hugging Face!")
    
    
def load_hf_dataset(dataset_name: str, split: str = 'train', config_name: Optional[str] = None) -> Dataset:
    """
    Load a dataset from Hugging Face Hub.

    Args:
        dataset_name (str): The name of the dataset to load.
        split (str): The split of the dataset to load (default is 'train').

    Returns:
        Dataset: The loaded dataset.
    """
    
    # Ensure environment variables are loaded
    _load_env()
        
    # Load the dataset from Hugging Face Hub
    dataset = load_dataset(dataset_name, split=split, name=config_name)
    
    # Verify that the dataset is of type Dataset
    if not isinstance(dataset, Dataset):
        raise ValueError(f"The loaded dataset is not of type Dataset. Got {type(dataset)} instead.")
    
    # Return the loaded dataset
    return dataset


def dataset_to_pandas(dataset: Dataset) -> DataFrame:
    """
    Convert a Hugging Face Dataset to a Pandas DataFrame.

    Args:
        dataset (Dataset): The Hugging Face Dataset to convert.

    Returns:
        DataFrame: The converted Pandas DataFrame.
    """
    
    # Convert the dataset to a Pandas DataFrame
    df = dataset.to_pandas()
    
    # Verify that the result is a DataFrame
    if not isinstance(df, DataFrame):
        raise ValueError(f"The converted object is not a DataFrame. Got {type(df)} instead.")
    
    # Return the DataFrame
    return df
