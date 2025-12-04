# ==============================================================================
# --- 1. IMPORTS
# ==============================================================================
#
import os
from dotenv import load_dotenv
import requests
import traceback
from flask import Flask, request, jsonify
from collections import defaultdict
import threading
import json
from fpdf import FPDF
from pydub import AudioSegment
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
import re
from datetime import datetime, timedelta

# --- AI & Machine Learning ---
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ==============================================================================
# --- 2. CONFIGURATION
# ==============================================================================
#
load_dotenv()

# --- WhatsApp & Meta Config ---
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")

# --- Google AI & API Config ---
GENAI_API_KEY = os.getenv("GENAI_API_KEY")

# --- RAG Config ---
RAG_DATA_PATH = "data"
VECTOR_DB_PATH = "vector_index/os_docs"

# --- Flask Config ---
FLASK_PORT = 5000

# ==============================================================================
# --- 3. GLOBAL STATE & AGENT LOGIC
# ==============================================================================

# A structured quiz data store with topics and questions
QUIZ_DATA = {
    "Operating Systems": [
        {
            "question": "What is the primary function of a kernel?",
            "options": ["Manage hardware resources", "Run applications", "Display graphics"],
            "answer": "Manage hardware resources"
        },
        {
            "question": "Which of these is a type of process scheduling algorithm?",
            "options": ["FIFO", "LIFO", "FILO"],
            "answer": "FIFO"
        },
        {
            "question": "What does 'mutex' stand for in OS?",
            "options": ["Mutual Exclusion", "Memory Unification", "Multithread Execution"],
            "answer": "Mutual Exclusion"
        }
    ],
    "Data Structures": [
        {
            "question": "Which data structure is a Last-In, First-Out (LIFO) collection?",
            "options": ["Queue", "Stack", "Linked List"],
            "answer": "Stack"
        },
        {
            "question": "What is the time complexity of searching for an element in a hash table (on average)?",
            "options": ["O(1)", "O(log n)", "O(n)"],
            "answer": "O(1)"
        },
        {
            "question": "In a binary search tree, are elements in the right subtree smaller or larger than the root?",
            "options": ["Smaller", "Larger", "Random"],
            "answer": "Larger"
        }
    ]
}

# Stores user-specific quiz data and active agent state
quiz_state = defaultdict(lambda: {"active": False, "topic": None, "question_index": 0, "score": 0})
agent_state = defaultdict(str)

def send_main_menu(to):
    """Sends the main menu with agent buttons."""
    if not ACCESS_TOKEN or not PHONE_NUMBER_ID:
        print("WhatsApp API credentials missing.")
        return False
        
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {
                "text": "Hello! I'm your AI OS Assistant. How can I help you today? Please choose an option:"
            },
            "action": {
                "buttons": [
                    {"type": "reply", "reply": {"id": "doubt_solving_agent", "title": "Doubt Solving"}},
                    {"type": "reply", "reply": {"id": "motivation_agent", "title": "Motivation"}},
                    {"type": "reply", "reply": {"id": "assessment_agent", "title": "Assessment"}}
                ]
            }
        }
    }
    
    try:
        res = requests.post(url, headers=headers, json=payload)
        res.raise_for_status()
        print(f"Interactive message sent to {to}.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to send interactive message: {e}")
        print(f"Response: {res.text if 'res' in locals() else 'No response'}")
        return False

def send_whatsapp_text_message(to, message):
    """Sends a standard text message."""
    if not ACCESS_TOKEN or not PHONE_NUMBER_ID:
        print("WhatsApp API credentials missing.")
        return False
        
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "text": {"body": message}
    }
    try:
        res = requests.post(url, headers=headers, json=payload)
        res.raise_for_status()
        print(f"WhatsApp text message sent to {to}.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to send WhatsApp message: {e}")
        print(f"Response: {res.text if 'res' in locals() else 'No response'}")
        return False

def send_quiz_topic_options(to):
    """Sends a list of quiz topics as interactive buttons."""
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    sections = [
        {
            "title": "Available Topics",
            "rows": [
                {"id": f"quiz_topic_{topic.replace(' ', '_').lower()}", "title": topic} for topic in QUIZ_DATA.keys()
            ]
        }
    ]
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "list",
            "header": {"type": "text", "text": "Choose a Quiz Topic"},
            "body": {"text": "Select the topic you'd like to be tested on:"},
            "action": {
                "button": "View Topics",
                "sections": sections
            }
        }
    }
    try:
        res = requests.post(url, headers=headers, json=payload)
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to send topic options: {e}")

def send_quiz_question(to, topic, index):
    """Sends a single quiz question with interactive radio buttons."""
    quiz_topic = QUIZ_DATA.get(topic)
    if not quiz_topic or index >= len(quiz_topic):
        send_whatsapp_text_message(to, "An error occurred with the quiz topic.")
        agent_state[to] = ''
        send_main_menu(to)
        return

    question_data = quiz_topic[index]
    options_list = [
        {"type": "reply", "reply": {"id": f"quiz_q_{index}_opt_{i}", "title": opt}}
        for i, opt in enumerate(question_data["options"])
    ]
    
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {
                "text": f"Question {index + 1}: {question_data['question']}"
            },
            "action": {
                "buttons": options_list
            }
        }
    }
    try:
        res = requests.post(url, headers=headers, json=payload)
        res.raise_for_status()
        print(f"Quiz question sent to {to}.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send quiz question: {e}")

def handle_assessment(to, user_text, button_id):
    """Manages the quiz flow."""
    state = quiz_state[to]
    current_agent_mode = agent_state[to]
    
    # Handle initial assessment button click from main menu
    if button_id == 'assessment_agent':
        agent_state[to] = 'quiz_topic_selection'
        send_whatsapp_text_message(to, "Let's test your knowledge! First, please choose a topic.")
        send_quiz_topic_options(to)
        return

    # Handle topic selection from the list menu
    if current_agent_mode == 'quiz_topic_selection' and button_id and button_id.startswith('quiz_topic_'):
        topic = button_id.replace('quiz_topic_', '').replace('_', ' ').title()
        if topic in QUIZ_DATA:
            state["active"] = True
            state["topic"] = topic
            state["question_index"] = 0
            state["score"] = 0
            agent_state[to] = 'quiz_in_progress'
            send_whatsapp_text_message(to, f"Starting the '{topic}' quiz. Good luck!")
            send_quiz_question(to, state["topic"], state["question_index"])
        else:
            send_whatsapp_text_message(to, "Invalid topic selected. Please try again.")
            send_quiz_topic_options(to)
        return
    
    # Quiz is active and user provided an answer
    elif current_agent_mode == 'quiz_in_progress' and button_id and button_id.startswith('quiz_q_'):
        current_quiz = QUIZ_DATA.get(state["topic"])
        if not current_quiz:
            send_whatsapp_text_message(to, "An error occurred with the quiz. Returning to main menu.")
            state["active"] = False
            agent_state[to] = ''
            send_main_menu(to)
            return

        current_question = current_quiz[state["question_index"]]
        try:
            option_index = int(button_id.split('_')[-1])
            user_answer = current_question["options"][option_index]
        except (ValueError, IndexError):
            send_whatsapp_text_message(to, "Invalid answer received. Please select an option using the buttons.")
            return

        if user_answer == current_question["answer"]:
            state["score"] += 1
            send_whatsapp_text_message(to, "Correct! 🎉")
        else:
            send_whatsapp_text_message(to, f"Not quite. The correct answer was: {current_question['answer']}")
        
        state["question_index"] += 1
        if state["question_index"] < len(current_quiz):
            send_whatsapp_text_message(to, "Next question coming up...")
            send_quiz_question(to, state["topic"], state["question_index"])
        else:
            send_whatsapp_text_message(to, f"Quiz complete! Your final score is {state['score']}/{len(current_quiz)}.")
            state["active"] = False
            agent_state[to] = ''
            send_main_menu(to)


# ==============================================================================
# --- 4. RAG SYSTEM
# ==============================================================================

if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)
else:
    print("Warning: GENAI_API_KEY not found. RAG functionality will be disabled.")

class RAGSystem:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = None
        self._load_vector_index()

    def _load_vector_index(self):
        if os.path.exists(VECTOR_DB_PATH):
            try:
                self.db = FAISS.load_local(VECTOR_DB_PATH, self.embeddings, allow_dangerous_deserialization=True)
                print("FAISS vector index loaded successfully.")
            except Exception as e:
                print(f"Error loading FAISS index: {e}")
                self.db = None
        else:
            print(f"No FAISS index found at {VECTOR_DB_PATH}.")
            self.db = None

    def build_vector_index(self, documents):
        print("Building new vector index...")
        if not documents:
            print("No documents to build index from.")
            return

        self.db = FAISS.from_documents(documents, self.embeddings)
        os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
        self.db.save_local(VECTOR_DB_PATH)
        print("Vector index built and saved successfully.")

    def query_knowledge_base(self, question):
        if not self.db:
            return "Knowledge base not available. Please contact the administrator."

        try:
            retriever = self.db.as_retriever(search_kwargs={"k": 2})
            docs = retriever.invoke(question)

            if not docs:
                return "No relevant information found in the knowledge base."

            context = "\n\n".join([doc.page_content for doc in docs])
            
            prompt = f"""
            Based on the following context, answer the question accurately and concisely.
            If the context does not contain enough information, state that directly and do not guess.

            Context:
            {context}

            Question: {question}

            Answer:
            """
            
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            return response.text
        
        except Exception as e:
            print(f"Error during RAG query: {e}")
            return "An error occurred while retrieving information."

    @staticmethod
    def load_documents_from_path(path):
        loaders = []
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if filename.endswith(".pdf") or filename.endswith(".pptx"):
                loaders.append(UnstructuredFileLoader(file_path))
        
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        
        for loader in loaders:
            try:
                raw_docs = loader.load()
                split_docs = text_splitter.split_documents(raw_docs)
                documents.extend(split_docs)
                print(f"Loaded {len(raw_docs)} documents from {loader.file_path}")
            except Exception as e:
                print(f"Error loading {loader.file_path}: {e}")
        return documents

def run_rag_setup(rag_system):
    if not os.path.exists(VECTOR_DB_PATH):
        documents = rag_system.load_documents_from_path(RAG_DATA_PATH)
        rag_system.build_vector_index(documents)

# ==============================================================================
# --- 5. FLASK APPLICATION
# ==============================================================================
#
app = Flask(__name__)
rag_system = RAGSystem()

def process_doubt_solving_async(sender, message):
    """Worker function for the doubt-solving agent."""
    try:
        rag_response_text = rag_system.query_knowledge_base(message)
        if rag_response_text:
            send_whatsapp_text_message(sender, rag_response_text)
        else:
            send_whatsapp_text_message(sender, "Sorry, I couldn't find an answer to that question.")
        send_whatsapp_text_message(sender, "Is there anything else I can help you with? You can always type 'menu' to go back.")
    except Exception as e:
        print(f"Error in async processing: {e}")
        send_whatsapp_text_message(sender, "An internal error occurred. Please try again later.")

def process_motivation_async(sender, message):
    """Worker function for the motivation agent."""
    try:
        prompt = f"""
        The user is asking for motivation for a specific reason or mood. Their input is: "{message}".
        Craft a short, encouraging, and uplifting message that is relevant to their request.
        Do not use emojis.
        """
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        
        motivational_text = response.text
        send_whatsapp_text_message(sender, motivational_text)
        send_whatsapp_text_message(sender, "Is there anything else I can help you with? You can always type 'menu' to go back.")
    except Exception as e:
        print(f"Error in motivation processing: {e}")
        send_whatsapp_text_message(sender, "An error occurred while getting your motivation. Please try again later.")
    finally:
        pass

@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    """Handles webhook verification and incoming messages from WhatsApp."""
    if request.method == 'GET':
        mode = request.args.get('hub.mode')
        token = request.args.get('hub.verify_token')
        challenge = request.args.get('hub.challenge')
        
        if mode == 'subscribe' and token == VERIFY_TOKEN:
            print("WEBHOOK_VERIFIED")
            return challenge, 200
        else:
            return "Verification token mismatch", 403

    if request.method == 'POST':
        data = request.json
        print("Received webhook data:", data)
        try:
            for entry in data.get('entry', []):
                for change in entry.get('changes', []):
                    value = change.get('value', {})
                    messages = value.get('messages', [])
                    for msg in messages:
                        sender = msg.get('from')
                        msg_type = msg.get('type')
                        
                        # Handle text messages for specific agent modes or main menu return
                        if msg_type == 'text':
                            user_text = msg['text']['body']
                            print(f"Received message from {sender}: {user_text}")
                            
                            if user_text.lower() in ['menu', 'main menu', 'start', 'end', 'stop']:
                                agent_state[sender] = ''
                                quiz_state[sender]["active"] = False
                                send_whatsapp_text_message(sender, "Returning to the main menu...")
                                send_main_menu(sender)
                                continue

                            current_agent = agent_state[sender]
                            if current_agent == 'doubt_solving':
                                threading.Thread(target=process_doubt_solving_async, args=(sender, user_text)).start()
                                agent_state[sender] = '' # Reset state after handling the doubt
                            elif current_agent == 'motivation':
                                threading.Thread(target=process_motivation_async, args=(sender, user_text)).start()
                                agent_state[sender] = '' # Reset state after handling the motivation
                            else:
                                send_main_menu(sender)
                                
                        # Handle interactive messages (button or list replies)
                        elif msg_type == 'interactive':
                            interactive_data = msg['interactive']
                            button_id = None
                            if interactive_data.get('button_reply'):
                                button_id = interactive_data['button_reply']['id']
                            elif interactive_data.get('list_reply'):
                                button_id = interactive_data['list_reply']['id']
                            
                            print(f"Received interactive message from {sender} with ID: {button_id}")

                            if not button_id:
                                continue
                            
                            # Route to the appropriate agent or state based on button ID
                            if button_id in ['doubt_solving_agent', 'motivation_agent']:
                                if button_id == 'doubt_solving_agent':
                                    agent_state[sender] = 'doubt_solving'
                                    send_whatsapp_text_message(sender, "What is your doubt? Please ask your question.")
                                elif button_id == 'motivation_agent':
                                    agent_state[sender] = 'motivation'
                                    send_whatsapp_text_message(sender, "What kind of motivation do you need today? Tell me about your mood or what you're trying to achieve.")
                            # All assessment-related logic is now handled by this one block
                            elif button_id.startswith('assessment_') or button_id.startswith('quiz_'):
                                handle_assessment(sender, None, button_id)
            
            return jsonify({"status": "ok"}), 200
            
        except Exception as e:
            print(f"Error processing message: {e}")
            traceback.print_exc()
            return jsonify({"status": "error"}), 500

# ==============================================================================
# --- 6. APPLICATION STARTUP
# ==============================================================================
#
if __name__ == "__main__":
    run_rag_setup(rag_system)
    
    print(f"Starting Flask app on port {FLASK_PORT}...")
    app.run(port=5000, debug=False)
