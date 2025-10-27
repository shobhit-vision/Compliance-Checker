import os
from dotenv import load_dotenv
from groq import Groq
load_dotenv()

api_key=os.environ.get("groq_api_key")

client = Groq(
    api_key=api_key,
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of Compliance Checker",
        }
    ],
    model="llama-3.3-70b-versatile",
    max_tokens=500,
    temperature=0.3,
    top_p=1,
)

print(chat_completion.choices[0].message.content)
