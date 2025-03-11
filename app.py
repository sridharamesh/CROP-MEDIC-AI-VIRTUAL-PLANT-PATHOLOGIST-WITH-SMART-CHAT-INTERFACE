from flask import Flask, render_template, request, jsonify
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
import os

app = Flask(__name__)

# -------------------- RAG Chatbot Code --------------------
class RAGChatbot:
    def __init__(self, model_name="mistral", collection_name="documents"):
        self.setup_components(model_name, collection_name)
        
    def setup_components(self, model_name, collection_name):
        self.system_prompt = """" 
        System Role:
        "You are an intelligent plant disease expert that provides only relevant, concise, and precise answers. Your role is to diagnose diseases, explain symptoms, and suggest treatments only when explicitly requested. Your responses should adapt to different question formats, including grammatical and ungrammatical queries, while maintaining clarity and brevity."

        Response Guidelines:

        If the user asks "What is [disease]?" (or similar questions like "Tell about [disease]" / "Explain [disease]") →

        Provide only the definition and why it occurs (symptoms and causes).

        Do not include treatment or management unless specifically asked.

        If the user asks "How to overcome [disease]?" (or similar like "Cure for [disease]?" / "How to treat [disease]?" / "Fix [disease]?") →

        Provide only management and treatment (chemical, organic, cultural).

        If both symptoms and management are requested in any way →

        Provide both but keep each section concise and relevant.

        Handle ungrammatical and informal questions naturally.

        Example: "Y tomato leaf blight happen?" → Answer with causes only.

        Example: "fix apple scab?" → Answer with treatment only.

        Example: "what is late blight n how to stop it?" → Answer with both definition and treatment.

        No Extra Details:

        Avoid unnecessary background, history, or excessive explanations unless explicitly asked.

        Keep responses short, clear, and structured for efficiency.

        Example Responses for Various Question Styles:
        ✅ User: "What is apple scab?"
        💡 Chatbot: "Apple scab is a fungal disease (Venturia inaequalis) causing olive-green or dark spots on leaves and fruit, leading to defoliation and poor fruit quality."

        ✅ User: "why tomato leaf blight happen?"
        💡 Chatbot: "Tomato leaf blight occurs due to fungal pathogens like Alternaria solani in warm, humid conditions, leading to brown leaf spots and defoliation."

        ✅ User: "how fix late blight?"
        💡 Chatbot: "Use fungicides like copper spray, remove infected plants, and ensure good air circulation to prevent moisture buildup."

        ✅ User: "late blight info and cure?"
        💡 Chatbot:
        "Late blight (Phytophthora infestans) causes dark lesions on leaves and spreads in wet conditions.
        Cure: Apply fungicides like chlorothalonil, avoid overhead watering, and remove infected plants."
        """
       

        self.memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        Settings.llm = Ollama(model="mistral", request_timeout=300.0)
        embedding = OllamaEmbedding(model_name="mxbai-embed-large")
        Settings.embed_model = embedding
        
        self.storage_context = StorageContext.from_defaults(persist_dir="data_storafe")
        self.index = load_index_from_storage(self.storage_context)
        
        self.chat_engine = self.index.as_chat_engine(
            query_engine=self.index.as_query_engine(),
            chat_mode="context",
            memory=self.memory,
            system_prompt=self.system_prompt,
            llm=Ollama(model="mistral", request_timeout=300.0)
        )
    
    def query(self, user_input):
        if not hasattr(self, 'chat_engine'):
            raise ValueError("Documents haven't been loaded yet.")
        response = self.chat_engine.chat(user_input)
        return response.response

    def append_message(self, message):
        self.memory.add_message(role="user", content=message)

# Initialize chatbot only when needed (not during import)
chatbot = None

# -------------------- Added Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat")
def chat():
    global chatbot
    if chatbot is None:
        chatbot = RAGChatbot()
    return render_template("chat.html")

# -------------------- API Routes --------------------
@app.route("/get_response", methods=["POST"])
def get_response():
    global chatbot
    if chatbot is None:
        chatbot = RAGChatbot()
    
    user_query = request.json.get("query", "")
    try:
        response = chatbot.query(user_query)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": str(e)}), 400

# -------------------- Plant Disease Classification --------------------
# Load model on first request
model = None
class_names = None
preprocess = None
device = None

def setup_model():
    global model, class_names, preprocess, device
    
    # Check if model is already loaded
    if model is not None:
        return
    
    # Initialize model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1000)
    
    # Add device detection to handle both CPU and GPU environments
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define model path
    model_path = os.path.join('models', 'plant_disease_classification.pth')
    
    # Load the model with map_location parameter to handle CPU-only environments
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load class names
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    
    # Setup preprocessing
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
        # Setup model on first request
        setup_model()
        
        if 'image' not in request.files:
            return render_template('disease_prediction.html', error="No image file uploaded.")
        
        file = request.files['image']
        if file.filename == '':
            return render_template('disease_prediction.html', error="No image selected.")
        
        image = Image.open(file.stream)
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(device)  # Move input to the same device as model
        
        with torch.no_grad():
            output = model(input_batch)
            _, predicted_class = output.max(1)
        
        predicted_class_name = class_names[predicted_class.item()]
        
        return render_template('disease_prediction_result.html', prediction=predicted_class_name)
    
    except Exception as e:
        return render_template('disease_prediction.html', error=str(e))
