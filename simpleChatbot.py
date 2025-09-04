# =========================================
# OpenAI Chat API Demo
# Securely load API key, test the API, 
# generate a question, and answer it.
# =========================================

from dotenv import load_dotenv
import os
from openai import OpenAI
from IPython.display import Markdown, display

# Step 1 — Load .env file
load_dotenv(override=True)

# Step 2 — Get the API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key:
    print(f"✅ OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("❌ OpenAI API Key not set - please check your .env file")

# Step 3 — Initialize the OpenAI client
openai = OpenAI()

# Step 4 — Send a simple test prompt
messages = [{"role": "user", "content": "What is 2+2?"}]
response = openai.chat.completions.create(
    model="gpt-4.1-nano",
    messages=messages
)
print("\nTest response:", response.choices[0].message.content)

# Step 5 — Ask the model to invent a hard question
question_prompt = "Please propose a hard, challenging question to assess someone's IQ. Respond only with the question."
messages = [{"role": "user", "content": question_prompt}]
response = openai.chat.completions.create(
    model="gpt-4.1-mini",
    messages=messages
)

question = response.choices[0].message.content
print("\nGenerated Question:", question)

# Step 6 — Ask the model to answer its own question
messages = [{"role": "user", "content": question}]
response = openai.chat.completions.create(
    model="gpt-4.1-mini",
    messages=messages
)

answer = response.choices[0].message.content
print("\nAnswer:", answer)

# Step 7 — Display nicely in Markdown (Jupyter only)
display(Markdown(answer))
