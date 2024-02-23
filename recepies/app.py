from fastrag.retrievers.colbert import ColBERTRetriever
from fastrag.stores import PLAIDDocumentStore
from haystack import Pipeline
import pandas as pd
import gradio as gr
from typing import List, Any

from index_docs import index_data

store = PLAIDDocumentStore(index_path="plaid",
                           checkpoint_path="Intel/ColBERT-NQ",
                           collection_path="sample_dataset.tsv", create=False, gpus=0, query_maxlen=120) 

retriever = ColBERTRetriever(document_store=store)

haystack_pipeline = Pipeline()
haystack_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])

def process_string(receipe_prep_steps):
    """
    Processes a given string to extract the recipe title and reformat content related to preparation steps.

    The function first identifies the recipe title as the text before the first period. It then looks for
    "Preparation steps:" in the string. If found, it reorganizes the content to follow these preparation steps
    directly, ensuring any additional text is concatenated appropriately.

    Parameters:
    - s (str): The input string containing the recipe title and preparation steps.

    Returns:
    - tuple: A tuple containing two strings; the first is the recipe title, and the second is the reformatted
             content string with preparation steps.
    """
    # Extract the recipe title as the text before the first period
    recipe_title = receipe_prep_steps.split('.')[0]
    
    # Attempt to rejoin the string without the recipe title, to focus on preparation steps
    prep_steps = ' '.join(receipe_prep_steps.split(recipe_title)[1:])
    
    # Split the remaining string at "Preparation steps:"
    parts = prep_steps.split('Preparation steps:')

    # Initialize content to be empty as a default case
    content = ''

    # Check if "Preparation steps:" is found within the string
    if len(parts) > 1:
        # The first part is considered as additional text, and the second part as the main content
        # Strip leading and trailing spaces and dots from the additional text (now reassigned as title for clarity)
        title = parts[0].strip('. ')
        # Also strip leading and trailing spaces from the content part
        content = parts[1].strip()
        # Construct the new string without explicitly adding "Preparation steps:" at the beginning
        output_string = title + " " + content
    else:
        # If "Preparation steps:" is not found, the entire string after the recipe title is considered as content
        output_string = parts[0].strip('. ')

    # Return the recipe title and the reformatted content
    return recipe_title, output_string


def get_suggestions(ingredients: List[str], tools: List[str]) -> pd.DataFrame:
    """
    Generates suggestions for recipes based on provided ingredients and tools,
    querying an unspecified model and processing the output into a DataFrame.
    
    Parameters:
    - ingredients (List[str]): A list of ingredients available for use.
    - tools (List[str]): A list of kitchen tools available for use.
    
    Returns:
    - pd.DataFrame: A DataFrame containing suggested recipes with titles,
                    ingredients, and preparation steps.
    """
    # Constructing the query from ingredients and tools
    query = f"What are the recipes that I can prepare using ingredients {ingredients} and {tools}?"
    # Assuming 'p.run' is a method call to some model's API. Replace with actual implementation.
    output = haystack_pipeline.run(query, params={"Retriever": {"top_k": 5}})
    
    # Initializing empty lists to store recipe information
    recipe_titles = []
    recipe_ingredients = []
    recipe_steps = []
    scores=[]
    
    # Iterating through the documents in the output to extract and process recipe information
    for out in output['documents']:
        receipe_title, receipe_prep_steps = process_string(out.meta['title'])
        # Append processed information to the lists
        recipe_ingredients.append(out.content.split(','))
        recipe_titles.append(receipe_title)
        recipe_steps.append(receipe_prep_steps)
        scores.append(round(out.score, 2))
        print(out.score)
    
    # Creating a DataFrame from the gathered recipe information
    output_df = pd.DataFrame({
        'Recipe Name': recipe_titles,
        'Total Ingredients Required': recipe_ingredients,
        'Preparation Steps': recipe_steps,
        'Confidence Score':scores
    })
    
    return output_df

with gr.Blocks() as demo:
    gr.Markdown("Provide the ingredients and kitchen tools available for interesting receipies.")
    with gr.Row():
      with gr.Column():
          input1 = gr.Textbox(placeholder="ingredients", label="Ingredients")
          input2 = gr.Textbox(placeholder="kitchen tools", label="Kitchen tools")
          btn = gr.Button("Extract Receipies")
          index_button = gr.Button("Index documents")

      out = gr.Dataframe()

    btn.click(fn=get_suggestions, inputs=[input1, input2], outputs=out)
    # index_button.click(index_data, outputs=gr.Textbox())

demo.launch(share=True, debug=True)

