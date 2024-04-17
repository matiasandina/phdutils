from telegram import Update, ForceReply
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackContext
import pandas as pd
from datetime import datetime, timedelta
import yaml
import logging

# Load configuration from YAML file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

bot_token = config['bot_token']
csv_file_path = config['csv_file_path']
threshold_minutes = config['threshold_minutes']

# Set up logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def load_subscribers():
    try:
        with open("subscribers.yaml", "r") as file:
            return set(yaml.safe_load(file))
    except FileNotFoundError:
        return set()

def save_subscribers(subscribers):
    with open("subscribers.yaml", "w") as file:
        yaml.safe_dump(list(subscribers), file)

subscribers = load_subscribers()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! Use /subscribe to get updates on Bonsai status.")

async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    if chat_id not in subscribers:
        subscribers.add(chat_id)
        save_subscribers(subscribers)
        await update.message.reply_text('You are now subscribed to alerts.')
    else:
        await update.message.reply_text('You are already subscribed.')

async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    if chat_id in subscribers:
        subscribers.remove(chat_id)
        save_subscribers(subscribers)
        await update.message.reply_text('You are no longer subscribed to alerts.')


alert_enabled = True

async def stop_bonsai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global alert_enabled
    alert_enabled = False
    await update.message.reply_text("Alerts are now disabled. Use `/restart_bonsai` to restart alerts")

async def restart_bonsai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global alert_enabled
    alert_enabled = True
    await update.message.reply_text("Alerts are now re-enabled.")

async def check_bonsai_status(context: CallbackContext):
    if not alert_enabled:
        return
    else:
        alert_data = await fetch_bonsai_status()
    if alert_data:
        await send_alerts(context, alert_data)

async def fetch_bonsai_status():
    """Fetch the latest status from the Bonsai CSV and determine if an alert is needed."""
    try:
        logger.info(f"Checking Bonsai Status at {csv_file_path}")
        data = pd.read_csv(csv_file_path)
        data['local_dt'] = pd.to_datetime(data['local_dt'], utc=True)
        latest_timestamp = data['local_dt'].iloc[-1].to_pydatetime()
        current_time_utc = datetime.utcnow().replace(tzinfo=latest_timestamp.tzinfo)

        if current_time_utc - latest_timestamp > timedelta(minutes=threshold_minutes):
            return {
                'working_dir': data['working_dir'].iloc[-1],
                'workflow_name': data['workflow_name'].iloc[-1],
                'last_update': latest_timestamp.astimezone()
            }
    except Exception as e:
        logger.error(f"Error checking Bonsai status: {e}")
    return None

async def send_alerts(context: CallbackContext, alert_data):
    """Send alert messages to all subscribed users."""
    message = (f"⚠️ *Bonsai System Alert* ⚠️\n\n"
               f"The Bonsai system at `{alert_data['working_dir']}` "
               f"running workflow `{alert_data['workflow_name']}` "
               f"has not updated since `{alert_data['last_update'].strftime('%Y-%m-%d %H:%M:%S %Z')}`. "
               f"Please check the system!")
    for chat_id in subscribers:
        await context.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')


def main():
    """Start the bot and set up job queue for monitoring."""
    application = Application.builder().token(bot_token).build()

    # Register command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("subscribe", subscribe))
    application.add_handler(CommandHandler("unsubscribe", unsubscribe))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, lambda update, context: update.message.reply_text(update.message.text)))

    application.add_handler(CommandHandler("stop_bonsai", stop_bonsai))
    application.add_handler(CommandHandler("restart_bonsai", restart_bonsai))

    # Set up and start the job queue
    job_queue = application.job_queue
    job_queue.run_repeating(check_bonsai_status, interval=60, first=0)

    # Run the bot until the user presses Ctrl-C
    application.run_polling()

if __name__ == "__main__":
    main()
