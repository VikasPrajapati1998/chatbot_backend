from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from model import *  # Import the fine-tuning function

app = FastAPI()

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; modify for production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods; modify as needed
    allow_headers=["*"],  # Allows all headers; modify as needed
)

# Initialize the question-answering pipeline with a specific model
model_name = "deepset/roberta-base-squad2"  # Using a more capable QA model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

class URLRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    question: str

# In-memory storage for scraped data
scraped_data_storage = {}

@app.post("/scrape/")
async def scrape_website(request: URLRequest):
    try:
        response = requests.get(request.url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36'
        })

        # Check if the request was successful
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            scraped_text = " ".join([tag.get_text(strip=True) for tag in soup.find_all([all])])

            # Store the scraped text in memory
            scraped_data_storage[request.url] = scraped_text

            # Return the scraped text
            return {"error": False, "scraped_text": scraped_text}
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to retrieve data from the provided URL.")

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

# # Initialize the text generation model
# llm_pipeline = pipeline("text-generation", model="gpt2")  # or "EleutherAI/gpt-neo-125M"

def get_llm_answer(scraped_text, question):
    prompt = f"Here is some information about a website:\n\n{scraped_text}\n\nGiven this information, answer the following question:\n{question}\n\nAnswer:"
    print("B4: ", type(prompt), len(prompt))
    # Limit the length of the prompt to avoid index errors
    if len(prompt) > 1024:
        prompt = prompt[:1024]
        print("After", type(prompt), len(prompt))
        print(prompt)  # Truncate the prompt to the first 1024 characters

    # Use the local model for generation with adjusted parameters
    # response = llm_pipeline(prompt, max_new_tokens=50, num_return_sequences=1)  # Adjust max_new_tokens as needed

    fine_tune_gpt2_on_dataset(prompt)
    text_generator = GPT2TextGenerator(model_path='fine_tuned_gpt2.pth')
    response = text_generator.generate_text(prompt, length=200)

    return response[0]['generated_text'].strip()

@app.post("/query/")
async def query_scraped_data(request: QueryRequest):
    if not scraped_data_storage:
        raise HTTPException(status_code=404, detail="No scraped data available.")

    last_url = list(scraped_data_storage.keys())[-1]
    scraped_text = scraped_data_storage[last_url]

    answer = get_llm_answer(scraped_text, request.question)

    return {
        "error": False,
        "answer": answer,
        "context": scraped_text[:100]  # Optionally, return a snippet of the context
    }

# To run the application, use:
# uvicorn app:app --reload
