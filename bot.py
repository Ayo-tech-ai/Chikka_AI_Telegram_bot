import os
import logging
import threading
import json
from flask import Flask
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters,
)
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import groq
import asyncio

# === Logging ===
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# === Flask Server ===
flask_app = Flask(__name__)

@flask_app.route("/")
def home():
    return "✅ Chikka AI Broiler Farming Assistant is alive."

def run_flask():
    flask_app.run(host="0.0.0.0", port=8080)

# === Load AI Components ===
FAISS_PATH = "rag_assets/faiss_index"
INDEX_METADATA_PATH = "rag_assets/index_metadata.json"

class SimpleRAGBot:
    def __init__(self):
        self.embedder = None
        self.index = None
        self.documents = []
        self.groq_client = None
        self.load_components()
    
    def load_components(self):
        try:
            # Load embeddings model
            self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("✅ Embeddings model loaded")
            
            # Load FAISS index
            if os.path.exists(FAISS_PATH):
                self.index = faiss.read_index(FAISS_PATH)
                logger.info("✅ FAISS index loaded")
            else:
                logger.warning("⚠️ FAISS index not found")
                return False
            
            # Load document metadata
            if os.path.exists(INDEX_METADATA_PATH):
                with open(INDEX_METADATA_PATH, 'r') as f:
                    metadata = json.load(f)
                    self.documents = metadata.get('documents', [])
                logger.info(f"✅ Loaded {len(self.documents)} documents")
            else:
                logger.warning("⚠️ Index metadata not found")
            
            # Initialize Groq client
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                self.groq_client = groq.Groq(api_key=groq_key)
                logger.info("✅ Groq client initialized")
            else:
                logger.warning("⚠️ GROQ_API_KEY not set")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading components: {e}")
            return False
    
    def search_similar_documents(self, query, k=3):
        """Search for similar documents using FAISS"""
        if not self.index or not self.embedder:
            return []
        
        # Embed the query
        query_embedding = self.embedder.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        # Get relevant documents
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    'content': self.documents[idx],
                    'distance': distances[0][i]
                })
        
        return results
    
    async def generate_response(self, query):
        """Generate response using Groq API with RAG context"""
        if not self.groq_client:
            return "❌ AI service is currently unavailable. Please check API configuration."
        
        # Get relevant context
        relevant_docs = self.search_similar_documents(query)
        context = "\n".join([doc['content'] for doc in relevant_docs[:2]])  # Use top 2 docs
        
        # Create prompt with context
        prompt = f"""You are Chikka, an expert broiler farming assistant. Use the following knowledge to answer the user's question.

Relevant knowledge:
{context}

User's question: {query}

Provide a helpful, expert response based on the knowledge above. If the knowledge doesn't contain the answer, say you don't have that specific information but offer related advice."""

        try:
            # Call Groq API directly
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are Chikka, a friendly expert in backyard broiler farming."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            return "❌ Sorry, I encountered an error processing your request. Please try again."

# Initialize the bot
rag_bot = SimpleRAGBot()

# === Telegram Commands ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    intro_message = (
        "🐔 *Welcome to Chikka AI - Your Broiler Farming Assistant!*\n\n"
        "🌱 *How I can help you:*\n"
        "• Ask about broiler health & diseases\n"
        "• Get advice on feeding & nutrition\n"
        "• Learn about housing & management\n"
        "• Understand vaccination & prevention\n"
        "• Get breed recommendations\n\n"
        "💡 *Just ask me anything about broiler farming*.\n\n"
        "⚠️ *Note:* I provide expert guidance, but always consult local experts for specific cases."
    )
    await update.message.reply_text(intro_message, parse_mode="Markdown")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_message = (
        "🐔 *Chikka AI Help*\n\n"
        "*Available Commands:*\n"
        "/start - Welcome message and introduction\n"
        "/help - Show this help message\n\n"
        "*Example Questions:*\n"
        "• What's the best feed for broilers?\n"
        "• How to prevent Newcastle disease?\n"
        "• What temperature should broilers be kept at?\n"
        "• Symptoms of Coccidiosis?\n"
        "• Best broiler breeds for small farms?\n\n"
        "💡 Tip: Ask specific questions for more accurate answers!"
    )
    await update.message.reply_text(help_message, parse_mode="Markdown")

# === Message Handler ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not rag_bot.groq_client:
        await update.message.reply_text(
            "❌ AI service is currently unavailable.\n\n"
            "Please check if GROQ_API_KEY is properly configured.",
            parse_mode="Markdown",
        )
        return

    user_query = update.message.text
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        # Generate response using our simple RAG system
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: rag_bot.generate_response(user_query))
        
        if not result:
            result = "⚠️ Sorry, I couldn't generate an answer for that question."

        # Telegram has 4096 character limit
        if len(result) > 4000:
            result = result[:4000] + "\n\n📝 Response truncated due to length limits."

        await update.message.reply_text(result, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        await update.message.reply_text(
            "❌ Error processing your question.\n\n"
            "Please try again in a moment.",
            parse_mode="Markdown",
        )

# === Main Entry ===
if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        raise ValueError("❌ TELEGRAM_BOT_TOKEN environment variable not found!")

    app = ApplicationBuilder().token(bot_token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("✅ Chikka AI Telegram bot starting...")
    app.run_polling()
