#can make ai sad,funny or anything we want
#assigning 3roles human,ai,system msgses
# human and ai bar bar and system ek bar 
#here we r giving system plus user prompts
import os 
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation", 
    huggingfacehub_api_token=hf_token,
    max_new_tokens=100
)

chat_model = ChatHuggingFace(llm=llm)

print("choose you AI mode")
print("1 for angry,2 for funny,3 for sad mode")

choice=int(input("enter you response"))

if choice == 1:
    mode = "Your r a very angry ai agent, u respond aggressively and impatiently"
elif choice ==2 :
    mode =" u r a funny ai agent, just  joke in every msg"
elif choice ==3:
    mode=" u r a sad ai agent, "    
messages =[
    SystemMessage(content=mode)
]

print("-----------Welcome Type 0 To Exit The Convo")

while True:
    prompt = input("You : ")
    messages.append(HumanMessage(content=prompt))

    if prompt == "0":
        print("Goodbye!")
        break

    response = chat_model.invoke(messages)
    messages.append(AIMessage(content=response.content))

    print("Bot :", response.content)
print(messages)
