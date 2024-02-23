
Input data is downloaded from huggingface manually due to privacy restrictions. It is not added to the repo as the file occupies around 2GB of space. 

The input data is sampled by taking the first 100,000 rows and applied preprocessing steps like removing the rows with null values in ingredients, description and title columns and formatting the title and content columns by removing the white spaces inorder to store it in tsv format requierd for the PLAID docuemnt store which is added to the repo with name sample_dataset.tsv.

While indexing, documet store expects the input data as id, content, title format where the ingredients from the input are labelled as content which is used by the retriever to embed and search and directions appended to the tile and used as title in the meta section. 


The sampled data is already being indexed into the PLAID document store and created an index folder plaid which contains the clusters, residuals and other files that PLAID document store requires to load the documents when needed.

Steps to run the gradio app and see the output:

1) Clone this project 

2) Create a virtual environmet and activate using the below comands:

    ```bash
    conda create -n venv python=3.10
    cond activate venv
    ```

3) Install the requirements from the requirements.txt file using the below command:

    ```bash
    pip install -r requirements.txt
    ```

4) To run the Gradio app using the following command:

    The gradio app accepts the ingredients and tools as inputs and retrieves the relevant receipies, ingredients required to prepare the receipe, preperation steps along with the confidence score in a table format from the indexed documents.
    There is a button added to index the data but disabled for now.

    ```bash
    cd recepies
    python app.py
```