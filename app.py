import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
import json
from flask import Flask,  request
from flask_cors import CORS, cross_origin


api_key=os.getenv("GEMINI_API_KEY") 

data = SimpleDirectoryReader(input_dir="data").load_data()

Settings.embed_model = GeminiEmbedding(
    model_name="models/embedding-001", api_key=api_key
)


Settings.llm = Gemini(api_key=api_key,temperature=0)
index = VectorStoreIndex.from_documents(data)

from llama_index.core.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=4000)

chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=(
        """You are a friendly chatbot, able to have normal interactions.
        You help the users with their questions. Return answers from the document only. 
        Do not make up your own answers.
        
        """
        
    ),
)



app = Flask(__name__)
cors = CORS(app=app) 
app.config['CORS_HEADERS'] = 'Content-Type' 

app.static_folder = 'static'
@app.route("/")
def home():
    return "KB Search API "
@app.route("/kb")
@cross_origin()
def get_bot_response():
    prompt = request.args.get('prompt')
    response = chat_engine.chat(prompt)
    output_dict={"response":response.response}
    output_json = json.dumps(output_dict)
    return output_json

# driver code 
if __name__ == "__main__":
    app.run()