import os
from together import Together
from dotenv import load_dotenv

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY is not set in the environment variables.")

client = Together(api_key=TOGETHER_API_KEY)

def generate_answer(question, contexts):
    prompt = f"""You are an expert assistant. Use the following context to answer the question.

Context:
{''.join(['- ' + ctx + '\n' for ctx in contexts])}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="togethercomputer/llama-2-70b-chat",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    if not response.choices:
        raise ValueError("Failed to generate a response from the model.")

    return response.choices[0].message.content.strip()
