import os
import logging
import threading
from flask import Flask
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import asyncio

# === Logging ===
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# === Flask Server (for uptime monitoring) ===
flask_app = Flask(__name__)

@flask_app.route("/")
def home():
    return "✅ Chikka AI Broiler Farming Assistant is alive."

def run_flask():
    flask_app.run(host="0.0.0.0", port=8080)

# === Load AI Components ===
FAISS_PATH = "rag_assets/faiss_index"

def load_ai_components():
    """Load vectorstore + Groq LLM and return QA chain"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        if not os.path.exists(FAISS_PATH):
            logger.warning(f"⚠️ FAISS index not found at {FAISS_PATH}")
            return None

        vectorstore = FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )

        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            logger.warning("⚠️ GROQ_API_KEY not set - AI functionality disabled")
            return None

        llm = ChatGroq(
            groq_api_key=groq_key,
            model="llama-3.3-70b-versatile",
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=False,
        )

        logger.info("✅ AI components loaded successfully")
        return qa_chain

    except Exception as e:
        logger.error(f"❌ Error loading AI components: {e}")
        return None

qa_chain = load_ai_components()

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
    if qa_chain is None:
        await update.message.reply_text(
            "❌ Knowledge base unavailable.\n\n"
            "Possible reasons:\n"
            "• FAISS index missing\n"
            "• GROQ_API_KEY not set\n"
            "• Technical maintenance",
            parse_mode="Markdown",
        )
        return

    user_query = update.message.text
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        # Run AI call in a separate thread (non-blocking)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, qa_chain.run, user_query)

        if not result:
            result = "⚠️ Sorry, I couldn't generate an answer for that."

        if len(result) > 4000:
            result = result[:4000] + "\n\n📝 Response truncated due to Telegram limits."

        await update.message.reply_text(result, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        await update.message.reply_text(
            "❌ Error processing your question.\n\n"
            "• AI service issue\n"
            "• Query too complex\n"
            "• Try again later",
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
