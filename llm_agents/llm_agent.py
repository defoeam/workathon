import os
import ai21
from ai21.models.chat import ChatMessage
from dotenv import load_dotenv
load_dotenv()

def run_agent(sysinsruct, text):
    client = ai21.AI21Client(api_key=os.getenv("AI21_API_KEY"))
    response = client.chat.completions.create(
        model="jamba-instruct-preview", 
        messages=[
            ChatMessage(   
                role="system",
                content=sysinsruct
            ),
            ChatMessage(
                role="user",
                content=text
            )
        ],
        temperature=0.8,
    )
    
    return response.choices[0].message.content