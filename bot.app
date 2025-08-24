import os
import logging
import re
import datetime
from typing import List, Dict

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# =======================
# Logging
# =======================
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# =======================
# Load FAISS
# =======================
FAISS_PATH = "rag_assets/faiss_index"

def load_vectorstore(faiss_path: str):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    return db

def init_llm_from_groq(model_name: str = "llama-3.3-70b-versatile"):
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY not found in environment variables.")
    llm = ChatGroq(groq_api_key=groq_key, model=model_name)
    return llm

def make_qa_chain(llm, vectorstore):
    prompt_template = """You are Chikka, a friendly expert AI assistant specialized in backyard broiler farming.
Provide helpful, conversational answers that are clear and focused. Be natural, avoid fluff.

Context: {context}
Question: {question}

Answer in a friendly, expert tone."""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# =======================
# Global
# =======================
vectorstore = load_vectorstore(FAISS_PATH)
llm = init_llm_from_groq()
qa_chain = make_qa_chain(llm, vectorstore)

# =======================
# Handlers
# =======================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send welcome message when user starts the bot"""
    await update.message.reply_text(
        "👋 Hi! I’m 🐔 ChikkaBot, your friendly backyard broiler assistant.\n"
        "Ask me anything about broiler care, feeding, housing, or diseases!"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command"""
    await update.message.reply_text("Just type your question, and I’ll do my best to help you with broiler farming 🐔.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user messages"""
    user_text = update.message.text
    logger.info(f"User said: {user_text}")

    await update.message.reply_text("🤔 Thinking...")

    try:
        out = qa_chain.invoke({"query": user_text})
        result = out["result"] if isinstance(out, dict) else str(out)
    except Exception as e:
        result = f"⚠️ Sorry, something went wrong: {e}"

    await update.message.reply_text(result)

# =======================
# Main
# =======================
def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not found in environment variables.")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("🤖 Bot started polling...")
    app.run_polling()

if __name__ == "__main__":
    main()
