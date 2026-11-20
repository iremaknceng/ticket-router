"""Central configuration: paths, model names, intent -> team routing table."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # downloaded CLINC150 (git-ignored)
MODEL_DIR = ROOT / "models"           # trained weights (git-ignored, hosted on HF Hub)
RESULTS_DIR = ROOT / "results"        # metrics produced by evaluate.py

# Hugging Face Hub repo that hosts the trained weights (set yours here or via env var)
HF_WEIGHTS_REPO = os.environ.get("ROUTER_HF_REPO", "iremaknceng/ticket-router-weights")

# CLINC150 "oos_plus" split (has 250 out-of-scope training examples)
CLINC_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_oos_plus.json"

# Transformer ladder
TEACHER_NAME = "bert-base-multilingual-cased"   # multilingual -> Turkish + English
STUDENT_LAYERS = 6                              # distilled student keeps every 2nd layer
MAX_LEN = 64
SEED = 42

OOS = "oos"
HUMAN_AGENT = "Human agent"
AUTO_REPLY = "Auto-reply (rule-based bot)"

# Model keys used everywhere (results table rows, app columns)
MODEL_KEYS = ["tfidf_lr", "w2v_ffnn", "bert_teacher", "student_fp32", "student_int8"]
MODEL_TITLES = {
    "tfidf_lr": "TF-IDF + LogReg (baseline)",
    "w2v_ffnn": "word2vec + FFNN (own, SGD)",
    "bert_teacher": "mBERT fine-tuned (FP32)",
    "student_fp32": "Distilled student (FP32)",
    "student_int8": "Distilled student (INT8)",
}

# --- CLINC150: 150 intents grouped in 10 domains (from the CLINC150 paper) ---
CLINC_DOMAINS = {
    "banking": ["transfer", "transactions", "balance", "freeze_account", "pay_bill", "bill_balance",
                "bill_due", "interest_rate", "routing", "min_payment", "order_checks", "pin_change",
                "report_fraud", "account_blocked", "spending_history"],
    "credit_cards": ["credit_score", "report_lost_card", "credit_limit", "rewards_balance", "new_card",
                     "application_status", "card_declined", "international_fees", "apr",
                     "redeem_rewards", "credit_limit_change", "damaged_card",
                     "replacement_card_duration", "improve_credit_score", "expiration_date"],
    "kitchen_dining": ["recipe", "restaurant_reviews", "calories", "nutrition_info",
                       "restaurant_suggestion", "ingredients_list", "ingredient_substitution",
                       "cook_time", "food_last", "meal_suggestion", "restaurant_reservation",
                       "confirm_reservation", "how_busy", "cancel_reservation", "accept_reservations"],
    "home": ["shopping_list", "shopping_list_update", "next_song", "play_music", "update_playlist",
             "todo_list", "todo_list_update", "calendar", "calendar_update", "what_song", "order",
             "order_status", "reminder", "reminder_update", "smart_home"],
    "auto_commute": ["traffic", "directions", "gas", "gas_type", "distance", "current_location", "mpg",
                     "oil_change_when", "oil_change_how", "jump_start", "uber", "schedule_maintenance",
                     "last_maintenance", "tire_pressure", "tire_change"],
    "travel": ["book_flight", "book_hotel", "car_rental", "travel_suggestion", "travel_alert",
               "travel_notification", "carry_on", "timezone", "vaccines", "translate", "flight_status",
               "international_visa", "lost_luggage", "plug_type", "exchange_rate"],
    "utility": ["time", "alarm", "share_location", "find_phone", "weather", "text", "spelling",
                "make_call", "timer", "date", "calculator", "measurement_conversion", "flip_coin",
                "roll_dice", "definition"],
    "work": ["direct_deposit", "pto_request", "taxes", "payday", "w2", "pto_balance",
             "pto_request_status", "next_holiday", "insurance", "insurance_change", "schedule_meeting",
             "pto_used", "meeting_schedule", "rollover_401k", "income"],
    "small_talk": ["greeting", "goodbye", "tell_joke", "where_are_you_from", "how_old_are_you",
                   "what_is_your_name", "who_made_you", "thank_you", "what_can_i_ask_you",
                   "what_are_your_hobbies", "do_you_have_pets", "are_you_a_bot", "meaning_of_life",
                   "who_do_you_work_for", "fun_fact"],
    "meta": ["change_ai_name", "change_user_name", "cancel", "user_name", "reset_settings",
             "whisper_mode", "repeat", "no", "yes", "maybe", "change_language", "change_accent",
             "change_volume", "change_speed", "sync_device"],
}
INTENT_TO_DOMAIN = {i: d for d, intents in CLINC_DOMAINS.items() for i in intents}

# Hand-written home-appliance intents (data/appliance_intents.csv), TR + EN
APPLIANCE_INTENT_TO_TEAM = {
    "appl_malfunction": "Technical Service",
    "appl_service_appointment": "Field Service Scheduling",
    "appl_spare_parts": "Spare Parts",
    "appl_installation": "Installation",
    "appl_warranty": "Warranty",
    "appl_billing_refund": "Billing & Refunds",
}

# CLINC domains -> support teams of the (simulated) contact center
DOMAIN_TO_TEAM = {
    "banking": "Payments & Banking",
    "credit_cards": "Payments & Banking",
    "kitchen_dining": "Kitchen & Cooking Advice",
    "home": "Smart Home & Orders",
    "auto_commute": "General Support",
    "travel": "General Support",
    "utility": "Self-service Assistant",
    "work": "HR & Employee Services",
    "small_talk": AUTO_REPLY,
    "meta": "Self-service Assistant",
}


def domain_for(intent: str) -> str:
    if intent.startswith("appl_"):
        return "appliance"
    return INTENT_TO_DOMAIN.get(intent, "unknown")


def team_for(intent: str) -> str:
    if intent == OOS:
        return HUMAN_AGENT
    if intent in APPLIANCE_INTENT_TO_TEAM:
        return APPLIANCE_INTENT_TO_TEAM[intent]
    return DOMAIN_TO_TEAM.get(domain_for(intent), "General Support")
