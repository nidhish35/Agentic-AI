# --- Imports ---
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from IPython.display import Markdown, display

# --- Load API keys from .env ---
load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

# --- Debugging API keys ---
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set (and this is optional)")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:2]}")
else:
    print("Google API Key not set (and this is optional)")

if deepseek_api_key:
    print(f"DeepSeek API Key exists and begins {deepseek_api_key[:3]}")
else:
    print("DeepSeek API Key not set (and this is optional)")

if groq_api_key:
    print(f"Groq API Key exists and begins {groq_api_key[:4]}")
else:
    print("Groq API Key not set (and this is optional)")

# --- Generate a nuanced test question using GPT-4o-mini ---
request = "Please come up with a challenging, nuanced question that I can ask a number of LLMs to evaluate their intelligence. Answer only with the question, no explanation."
messages = [{"role": "user", "content": request}]

openai = OpenAI()
response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)
question = response.choices[0].message.content
print("\nGenerated Question:", question)

# --- Prepare for competition ---
competitors = []
answers = []
messages = [{"role": "user", "content": question}]

# --- Ask GPT-4o-mini ---
model_name = "gpt-4o-mini"
response = openai.chat.completions.create(model=model_name, messages=messages)
answer = response.choices[0].message.content
display(Markdown(answer))
competitors.append(model_name)
answers.append(answer)

# --- Ask Claude 3.7 Sonnet ---
model_name = "claude-3-7-sonnet-latest"
claude = Anthropic()
response = claude.messages.create(model=model_name, messages=messages, max_tokens=1000)
answer = response.content[0].text
display(Markdown(answer))
competitors.append(model_name)
answers.append(answer)

# --- Ask Gemini 2.0 Flash ---
gemini = OpenAI(api_key=google_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
model_name = "gemini-2.0-flash"
response = gemini.chat.completions.create(model=model_name, messages=messages)
answer = response.choices[0].message.content
display(Markdown(answer))
competitors.append(model_name)
answers.append(answer)

# --- Ask DeepSeek ---
deepseek = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com/v1")
model_name = "deepseek-chat"
response = deepseek.chat.completions.create(model=model_name, messages=messages)
answer = response.choices[0].message.content
display(Markdown(answer))
competitors.append(model_name)
answers.append(answer)

# --- Ask Groq (Llama 3.3) ---
groq = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
model_name = "llama-3.3-70b-versatile"
response = groq.chat.completions.create(model=model_name, messages=messages)
answer = response.choices[0].message.content
display(Markdown(answer))
competitors.append(model_name)
answers.append(answer)

# --- Show collected answers ---
print("\n--- Competitors and Answers ---")
for competitor, answer in zip(competitors, answers):
    print(f"\nCompetitor: {competitor}\nAnswer: {answer}\n")

# --- Prepare responses for judging ---
together = ""
for index, answer in enumerate(answers):
    together += f"# Response from competitor {index+1}\n\n"
    together += answer + "\n\n"

judge = f"""You are judging a competition between {len(competitors)} competitors.
Each model has been given this question:

{question}

Your job is to evaluate each response for clarity and strength of argument, and rank them in order of best to worst.
Respond with JSON, and only JSON, with the following format:
{{"results": ["best competitor number", "second best competitor number", "third best competitor number", ...]}}

Here are the responses from each competitor:

{together}

Now respond with the JSON with the ranked order of the competitors, nothing else. Do not include markdown formatting or code blocks."""

judge_messages = [{"role": "user", "content": judge}]

# --- Judge with OpenAI o3-mini ---
response = openai.chat.completions.create(
    model="o3-mini",
    messages=judge_messages,
)
results = response.choices[0].message.content
print("\nRaw Judge Results:", results)

# --- Parse and print leaderboard ---
results_dict = json.loads(results)
ranks = results_dict["results"]
print("\n--- Final Leaderboard ---")
for index, result in enumerate(ranks):
    competitor = competitors[int(result)-1]
    print(f"Rank {index+1}: {competitor}")
