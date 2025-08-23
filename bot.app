import os
from telegram.ext import Application, CommandHandler, MessageHandler, filters

# Load your bot token
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "API")

# Start command
async def start(update, context):
    await update.message.reply_text("👋 Hello farmer! I'm Chikka 🐔. Ask me about broiler farming.")

# Echo all messages for now
async def handle_message(update, context):
    user_text = update.message.text
    await update.message.reply_text(f"🐔 You said: {user_text}")

def main():
    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
