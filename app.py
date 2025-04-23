from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import Settings, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import StorageContext
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from ollama import chat as ch

app = Flask(__name__)

# -------------------- RAG Chatbot Code --------------------
class RAGChatbot:
    def __init__(self, model_name="mistral", collection_name="documents"):
        self.chat_history = [
            {"role": "system", "content": "You are a plant disease expert providing concise, relevant information on diagnosis and treatment"}
        ]
        self.setup_components(model_name, collection_name)
        
    def setup_components(self, model_name, collection_name):
        self.system_prompt = """
                System Role:
        You are a plant disease expert providing concise, relevant information on diagnosis and treatment only when asked.

        Response Rules:
        1. Disease Definition: For "What is [disease]?" → brief symptoms and cause only.
        2. Treatment: For "How to treat [disease]?" → treatment options only.
        3. Both: If both requested → separate sections for each.
        4. Understand informal queries.
        5. Fertilizer: For "[crop] fertilizer" → suggest specific options.
        6. Crops: For "crops for [condition]" → suggest suitable varieties.
        7. Always: Short, clear, structured responses without background/history.

        Examples:
        ✅ "what is apple scab?" → "Fungal disease (Venturia inaequalis) causing dark spots on leaves/fruit, leading to defoliation."
        ✅ "how fix late blight?" → "Apply copper-based fungicides, remove infected plants, improve airflow."
        """

        self.memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        Settings.llm = Ollama(model=model_name, request_timeout=300.0)
        Settings.embed_model = OllamaEmbedding(model_name="mxbai-embed-large")

        self.storage_context = StorageContext.from_defaults(persist_dir="data_storage")
        self.index = load_index_from_storage(self.storage_context)
        
        self.chat_engine = self.index.as_chat_engine(
            query_engine=self.index.as_query_engine(),
            chat_mode="context",
            memory=self.memory,
            system_prompt=self.system_prompt,
            llm=Ollama(model=model_name, request_timeout=300.0)
        )

    def chat_with_ollama(self, user_input):
        self.chat_history.append({"role": "user", "content": user_input})
        response = ch(
            model="mistral",
            messages=self.chat_history,
            stream=False
        )
        reply = response["message"]["content"]
        
       
        self.chat_history.append({"role": "assistant", "content": reply})
        
        return reply

    def query(self, user_input):
        if not hasattr(self, 'chat_engine'):
            raise ValueError("Documents haven't been loaded yet.")
        response = self.chat_engine.chat(user_input)
        return response.response

chatbot = RAGChatbot()

# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat")
def chat():
    return render_template("chat.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_query = request.json.get("query", "")
    try:
        response = chatbot.query(user_query)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": str(e)}), 400

# -------------------- Plant Disease Classification --------------------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1000)
model.load_state_dict(torch.load('plant_disease_classification.pth'))
model.eval()

with open("class_names.json", "r") as f:
    class_names = json.load(f)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/disease_prediction', methods=['GET', 'POST'])
def disease_prediction():
    if request.method == 'GET':
        return render_template('disease_prediction.html')
        
    try:
        if 'image' not in request.files:
            return render_template('disease_prediction.html', error="No image file uploaded.")
        
        file = request.files['image']
        if file.filename == '':
            return render_template('disease_prediction.html', error="No image selected.")
        
        file.save('ollama_image.jpg')
        image = Image.open('ollama_image.jpg')
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_batch)
            _, predicted_class = output.max(1)
        
        predicted_class_name = class_names[predicted_class.item()]
        
        return render_template('disease_prediction_result.html', prediction=predicted_class_name)
    
    except Exception as e:
        return render_template('disease_prediction.html', error=str(e))


# Stream response for real-time typing effect
@app.route('/ask_question_stream', methods=['POST'])
def ask_question_stream():
    data = request.json
    question = data.get("question", "")
    predicted_class_name = data.get("predicted_class_name", "")
    
    def generate():
        context = f"The user is asking about {predicted_class_name}. "
        full_question = context + question
        
        # Add to chat history
        chatbot.chat_history.append({"role": "user", "content": full_question})
        
        # Get streaming response
        response = ch(
            model="mistral",
            messages=chatbot.chat_history[-1],
            stream=False
        )
        
        # Accumulated response for chat history
        reply = response["message"]["content"]
        
        for chunk in response:
            if "message" in chunk and "content" in chunk["message"]:
                piece = chunk["message"]["content"]
                full_reply += piece
                yield f"data: {json.dumps({'chunk': piece})}\n\n"
        
        # Add the complete response to chat history after streaming
        chatbot.chat_history.append({"role": "assistant", "content": full_reply})
        yield f"data: {json.dumps({'end': True})}\n\n"
        
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

# Keep the old endpoint for backward compatibility
@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get("question", "")
    predicted_class_name = data.get("predicted_class_name", "")
    
    try:
        context = f"The user is asking about {predicted_class_name}. "
        full_question = context + question
        
        response = chatbot.chat_with_ollama(full_question)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": str(e)}), 400

# Keep the old endpoint for backward compatibility
@app.route('/ask_followup_question', methods=['POST'])
def ask_followup_question():
    data = request.json
    question = data.get("question", "")
    
    try:
        chatbot_response = chatbot.chat_with_ollama(question)
        return jsonify({"chatbot_response": chatbot_response})
    except Exception as e:
        return jsonify({"chatbot_response": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
