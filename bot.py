import os
import logging
from flask import Flask, request
from telegram import Update, Bot
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, Filters
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --------------------
# Logging setup
# --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------
# Flask app
# --------------------
app = Flask(__name__)

# --------------------
# Telegram Bot setup
# --------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
bot = Bot(token=TELEGRAM_TOKEN)

# Dispatcher to handle updates
dispatcher = Dispatcher(bot, None, workers=0)

# --------------------
# Load LLM + FAISS
# --------------------
FAISS_PATH = "rag_assets/faiss_index"

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

def init_llm():
    groq_key = os.getenv("GROQ_API_KEY")
    return ChatGroq(groq_api_key=groq_key, model="llama-3.3-70b-versatile")

vectorstore = load_vectorstore()
llm = init_llm()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# --------------------
# Command Handlers
# --------------------
def start(update: Update, context):
    update.message.reply_text("🐔 Hello! I’m Chikka, your broiler farming assistant. Ask me anything about poultry health, feeding, or management!")

def handle_message(update: Update, context):
    query = update.message.text
    response = qa_chain.run(query)
    update.message.reply_text(response)

dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

# --------------------
# Flask Routes
# --------------------
@app.route("/")
def home():
    return "Chikka Telegram Bot is running!", 200

@app.route("/webhook", methods=["POST"])
def webhook():
    if request.method == "POST":
        update = Update.de_json(request.get_json(force=True), bot)
        dispatcher.process_update(update)
        return "ok", 200

# --------------------
# Run Locally
# --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
