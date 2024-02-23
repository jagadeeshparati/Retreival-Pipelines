from datasets import Dataset
import datasets
from typing import Dict, List, Any
import pandas as pd
from fastrag.stores import PLAIDDocumentStore
import fastrag, torch

def preprocess_function(examples: Dict[str, List]) -> Dict[str, List]:
    """
    Preprocesses the given dataset examples by combining ingredients into a single string,
    ensuring directions are properly formatted, and concatenating title and directions into a single content field.
    
    Parameters:
    examples (Dict[str, List]): A dictionary where keys are column names and values are lists of column values.
    
    Returns:
    Dict[str, List]: The modified dictionary with updated 'content' and 'title' based on preprocessing steps.
    """
    # Combine ingredients into a single string, separating each ingredient by a comma
    examples["content"] = [",".join(ingredients) for ingredients in examples["ingredients"]]
    
    # Ensure directions are a single string; join if list, else leave as is
    examples["directions"] = [" ".join(directions) if isinstance(directions, list) else directions for directions in examples["directions"]]
    
    # Concatenate title and directions into a single string with formatting
    examples["title"] = [title + ". Preparation steps: " + directions for title, directions in zip(examples["title"], examples["directions"])]
    
    return examples


def filter_none_values_in_columns(example: Dict[str, str]) -> bool:
    """
    Filters out examples where specified columns contain 'None', ['None'], or 'N/A'.
    
    Parameters:
    example (Dict[str, str]): A dictionary representing a single row from the dataset.
    
    Returns:
    bool: True if the example does not contain 'None' values in specified columns, False otherwise.
    """
    # Columns to check for 'None' or 'N/A' values
    columns_to_check = ['title', 'content']
    
    # Iterate through specified columns and return False if 'None' or 'N/A' values are found
    for column in columns_to_check:
        if example[column] in ("None", ["None"], "N/A"):
            return False  # Indicates row should be removed
    return True  # Indicates row should be kept


def initialize_document_store(index_folder_name: str, model_checkpoint: str, passages_path: str, create_index: bool, num_gpus: int) -> Any:
    """
    Initializes and returns a document store using the PLAIDDocumentStore class.
    
    This function configures the document store with the specified index folder, model checkpoint,
    passages path, and the number of GPUs. It also determines whether to create a new index based on the 'create_index' flag.
    
    Parameters:
    - index_folder_name (str): The folder name where the index is stored or will be created.
    - model_checkpoint (str): The path to the model checkpoint for document embedding.
    - passages_path (str): The path to the file or directory containing the passages to be indexed.
    - create_index (bool): Flag indicating whether to create a new index.
    - num_gpus (int): The number of GPUs to use for document processing and indexing.
    
    Returns:
    - Any: An instance of the PLAIDDocumentStore class configured with the specified parameters.
    """
    
    # Initialize the PLAIDDocumentStore with provided parameters
    store = PLAIDDocumentStore(index_path=index_folder_name,
                               checkpoint_path=model_checkpoint,
                               collection_path=passages_path, 
                               create=create_index, 
                               gpus=num_gpus)
    
    # Return the initialized document store
    return store


def index_data():

    # Load the receipts dataset from higgingface
    dataset=datasets.load_dataset("recipe_nlg", data_dir="dataset 2")

    # Apply the preprocessing function to the dataset with batch processing
    preprocessed_dataset = dataset['train'].map(preprocess_function, batched=True)

    # Filter the preprocessed dataset to remove rows with 'None' values in specified columns
    # Utilizing multiple processing cores (num_proc=4) for efficiency
    filtered_dataset = preprocessed_dataset.filter(lambda example: filter_none_values_in_columns(example), num_proc=4)

    # Remove unwanted columns
    columns_to_remove = ['directions','ingredients', 'link','source','ner']  # Update this list as needed
    filtered_dataset = filtered_dataset.remove_columns(columns_to_remove)


    new_dataset=filtered_dataset.to_pandas()

    # Clean the data to store as a tsv file required for PLAID document store.
    new_dataset['content'] = new_dataset['content'].apply(lambda x: x.replace('\n', '').replace('\r', '') if pd.notnull(x) else x)
    new_dataset['title'] = new_dataset['title'].apply(lambda x: x.replace('\n', '').replace('\r', '') if pd.notnull(x) else x)

    # As all titile content are imoportant fields to get the receipe outputs dropping nulls.
    new_dataset=new_dataset.dropna()
    new_dataset['id']=[i for i in range(new_dataset.shape[0])]

    # Sample the first 100,000 rows from the input dataset
    new_dataset[['id', 'content', 'title']][0:100000].to_csv('new_dataset1.tsv', sep='\t', index=False, encoding='utf-8', header=False)

    # Index the sample data to PLAID document store by using the COLBERTv2 model trained on the natural questions dataset
    document_store=initialize_document_store(index_path="plaid",
                                checkpoint_path="Intel/ColBERT-NQ",
                                collection_path="sample_dataset.tsv", create=True, gpus=1) 
    return "done indexing"