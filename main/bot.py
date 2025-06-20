# ./app/bot.py

import random
import pickle
import os
import json
import logging
import traceback
from enum import Enum
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from dotenv import load_dotenv
from data.config import CONFIG
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz

from main.utils import (
    clear_phrase, is_meaningful_text,
    extract_shoe_name, extract_shoe_category, extract_price, extract_brand,
    lemmatize_phrase, analyze_sentiment, Stats
)

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Загрузка токена ---
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')

# --- Загрузка каталога обуви ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PRODUCTS_PATH = os.path.join(DATA_DIR, 'products.json')
with open(PRODUCTS_PATH, encoding='utf-8') as f:
    PRODUCTS = json.load(f)
PRODUCTS_MAP = {item['name']: item for item in PRODUCTS}

# --- Состояния бота ---
class BotState(Enum):
    NONE = "NONE"
    WAITING_FOR_ITEM = "WAITING_FOR_ITEM"
    WAITING_FOR_INTENT = "WAITING_FOR_INTENT"

# --- Намерения ---
class Intent(Enum):
    HELLO = "hello"
    BYE = "bye"
    YES = "yes"
    NO = "no"
    SHOE_TYPES = "shoe_types"
    SHOE_PRICE = "shoe_price"
    SHOE_AVAILABILITY = "shoe_availability"
    SHOE_RECOMMENDATION = "shoe_recommendation"
    FILTER_SHOES = "filter_shoes"
    SHOE_INFO = "shoe_info"
    BOOK_FITTING = "book_fitting"
    COMPARE_SHOES = "compare_shoes"

# --- Типы ответов ---
class ResponseType(Enum):
    INTENT = "intent"
    GENERATE = "generate"
    FAILURE = "failure"

# --- Класс бота ---
class Bot:
    def __init__(self):
        # загрузка моделей
        try:
            base = os.path.join(os.path.dirname(__file__), '..', 'models')
            with open(os.path.join(base, 'intent_model.pkl'), 'rb') as f:
                self.clf = pickle.load(f)
            with open(os.path.join(base, 'intent_vectorizer.pkl'), 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(os.path.join(base, 'dialogues_vectorizer.pkl'), 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            with open(os.path.join(base, 'dialogues_matrix.pkl'), 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
            with open(os.path.join(base, 'dialogues_answers.pkl'), 'rb') as f:
                self.answers = pickle.load(f)
        except FileNotFoundError as e:
            logger.error(f"Модель не найдена: {e}\n{traceback.format_exc()}")
            raise

    def _update_context(self, context, replica, answer, intent=None):
        """Обновляет контекст пользователя."""
        context.user_data.setdefault('state', BotState.NONE.value)
        context.user_data.setdefault('current_shoe', None)
        context.user_data.setdefault('last_bot_response', None)
        context.user_data.setdefault('last_intent', None)
        context.user_data.setdefault('history', [])

        context.user_data['history'].append(replica)
        context.user_data['history'] = context.user_data['history'][-CONFIG['history_limit']:]
        context.user_data['last_bot_response'] = answer
        if intent:
            context.user_data['last_intent'] = intent

    def classify_intent(self, replica):
        """Классифицирует намерение пользователя."""
        text = replica.lower()
        if 'сравн' in text or 'чем отличается' in text:
            return Intent.COMPARE_SHOES.value
        replica_lemmatized = lemmatize_phrase(replica)
        if not replica_lemmatized:
            return None
        vectorized = self.vectorizer.transform([replica_lemmatized])
        intent_pred = self.clf.predict(vectorized)[0]
        best_score = 0
        best_intent = None
        for intent_key, data in CONFIG['intents'].items():
            examples = [lemmatize_phrase(ex) for ex in data.get('examples', []) if lemmatize_phrase(ex)]
            if not examples:
                continue
            match = process.extractOne(replica_lemmatized, examples, scorer=fuzz.ratio)
            score = (match[1] / 100) if match else 0
            if score > best_score and score >= CONFIG['thresholds']['intent_score']:
                best_score = score
                best_intent = intent_key
        logger.info(
            f"Classify intent: text='{replica_lemmatized}', pred='{intent_pred}', best='{best_intent}', score={best_score}")
        return best_intent or intent_pred if best_score >= CONFIG['thresholds']['intent_score'] else None

    def _get_shoe_response(self, intent, shoe_name, replica, context):
        """Обрабатывает запросы, связанные с конкретной моделью обуви."""
        prod = PRODUCTS_MAP.get(shoe_name)
        if not prod:
            return "Извините, такой модели обуви нет в каталоге."
        template = random.choice(CONFIG['intents'][intent]['responses'])
        # Замена заполнителей
        answer = template.replace('{shoe_name}', shoe_name).replace('{price}', str(prod['price']))
        answer = answer.replace('{description}', prod.get('description', 'отличная модель'))
        # Добавляем эмоциональную приписку
        sentiment = analyze_sentiment(replica)
        if sentiment == 'positive':
            answer += " Рад, что вам нравится! 😊"
        elif sentiment == 'negative':
            answer += " Понимаю сомнения. Давайте примерим? 😊"
        return f"{answer} Что ещё интересует?"

    def _find_shoe_by_context(self, replica, context):
        """Ищет модель обуви по контексту."""
        last = context.user_data.get('last_bot_response', '')
        last_intent = context.user_data.get('last_intent', '')
        category = extract_shoe_category(replica)

        # Если мы недавно рекламировали модель
        if last and 'Кстати, у нас есть' in last:
            return extract_shoe_name(last)
        # Если пользователь упомянул категорию
        if category:
            candidates = [name for name, p in PRODUCTS_MAP.items() if category in p.get('categories', [])]
            return random.choice(candidates) if candidates else None
        # Если предыдущий интент был показ типов
        if last_intent == Intent.SHOE_TYPES.value:
            for hist in reversed(context.user_data.get('history', [])):
                name = extract_shoe_name(hist)
                if name:
                    return name
        return None

    def _handle_filter_shoes(self, price, category, context):
        """Фильтрует каталог по цене и/или категории."""
        candidates = [
            name for name, p in PRODUCTS_MAP.items()
            if (not price or p['price'] <= price)
               and (not category or category in p.get('categories', []))
        ]
        used = [extract_shoe_name(h) for h in context.user_data.get('history', [])]
        candidates = [n for n in candidates if n not in used]

        if not candidates:
            conds = []
            if price:
                conds.append(f"до {price}₽")
            if category:
                conds.append(f"в категории {category}")
            return f"Извините, не нашлось моделей для {' и '.join(conds)}."
        # Если нет конкретных критериев — даём совет
        if not price and not category:
            choice_name = random.choice(candidates)
            context.user_data['current_shoe'] = choice_name
            context.user_data['state'] = BotState.WAITING_FOR_INTENT.value
            return f"Рекомендую {choice_name}! Хотите узнать цену или детали?"
        # Иначе — список моделей
        return "Нашлись такие модели: " + ", ".join(candidates)

    def get_answer_by_intent(self, intent, replica, context):
        """Генерирует ответ на основе распознанного интента."""
        shoe_name = context.user_data.get('current_shoe')
        last_intent = context.user_data.get('last_intent', '')
        category = extract_shoe_category(replica)
        price = extract_price(replica)

        if intent not in CONFIG['intents']:
            return None
        responses = CONFIG['intents'][intent]['responses']
        if not responses:
            return None
        answer = random.choice(responses)
        # Эмоциональная приписка
        sentiment = analyze_sentiment(replica)
        suffix = ""
        if sentiment == 'positive':
            suffix = " Рад, что вы в хорошем настроении! 😊"
        elif sentiment == 'negative':
            suffix = " Кажется, вы не в духе. Давайте подберём что-то доброе! 😊"

        # Обработка разных интентов
        if intent in [Intent.SHOE_PRICE.value, Intent.SHOE_AVAILABILITY.value, Intent.SHOE_INFO.value]:
            if not shoe_name:
                # пытаемся найти по контексту
                shoe_name = self._find_shoe_by_context(replica, context)
                if shoe_name:
                    context.user_data['current_shoe'] = shoe_name
                    context.user_data['state'] = BotState.WAITING_FOR_INTENT.value
                    return f"Из {category or 'каталога'} доступны {shoe_name}. Что хотите узнать?{suffix}"
                context.user_data['state'] = BotState.WAITING_FOR_ITEM.value
                return f"Уточните, пожалуйста, модель или категорию.{suffix}"
            return self._get_shoe_response(intent, shoe_name, replica, context)

        elif intent == Intent.SHOE_RECOMMENDATION.value:
            answer = self._handle_filter_shoes(None, category, context)

        elif intent == Intent.FILTER_SHOES.value:
            if price or category:
                answer = self._handle_filter_shoes(price, category, context)
            else:
                return f"Укажите цену или категорию для фильтрации.{suffix}"

        elif intent == Intent.SHOE_TYPES.value:
            # Если пользователь прямо назвал категорию — показываем все модели в ней
            category = extract_shoe_category(replica)
            if category:
                items = [name for name, p in PRODUCTS_MAP.items() if category in p.get('categories', [])]
                if items:
                    answer = f"В категории «{category}» есть: {', '.join(items)}."
                else:
                    answer = f"В категории «{category}» пока нет товаров."
                # Сбрасываем текущую модель, чтобы фото потом не дублировалось
                context.user_data['current_shoe'] = None
                return answer
            # Иначе — общий ответ со списком категорий и примерами моделей
            cats = list({c for p in PRODUCTS_MAP.values() for c in p.get('categories', [])})
            some = random.sample(cats, min(3, len(cats)))
            example = random.sample(list(PRODUCTS_MAP.keys()), 2)
            answer = f"У нас есть категории {', '.join(some)} и модели вроде {', '.join(example)}. Что интересно?{suffix}"
            context.user_data['current_shoe'] = None

        elif intent == Intent.COMPARE_SHOES.value:
            text_lem = lemmatize_phrase(replica)
            found = []
            for name, data in PRODUCTS_MAP.items():
                name_lem = lemmatize_phrase(name)
                if name_lem in text_lem:
                    found.append(name)
                else:
                    for syn in data.get('synonyms', []):
                        if lemmatize_phrase(syn) in text_lem:
                            found.append(name)
                            break
            if len(found) >= 2:
                s1, s2 = found[0], found[1]
            else:
                s1, s2 = random.sample(list(PRODUCTS_MAP.keys()), 2)
            answer = answer.replace('[shoe1]', s1).replace('[shoe2]', s2) + suffix
            
        elif intent == Intent.BOOK_FITTING.value:
            if not shoe_name:
                context.user_data['state'] = BotState.WAITING_FOR_ITEM.value
                return f"Какую модель вы хотите примерить?{suffix}"
            template = random.choice(responses)
            answer = template.replace('{shoe_name}', shoe_name) + suffix
            return answer

        elif intent == Intent.YES.value:
            # аналогично деталям по YES/NO для обуви...
            answer = f"Что ещё вас интересует?{suffix}"

        elif intent == Intent.NO.value:
            context.user_data['current_shoe'] = None
            context.user_data['state'] = BotState.NONE.value
            answer = f"Хорошо, какую модель обсудим дальше?{suffix}"

        context.user_data['last_intent'] = intent
        return answer

    def generate_answer(self, replica, context):
        """Генерирует ответ на основе диалогов."""
        replica_lemmatized = lemmatize_phrase(replica)
        if not replica_lemmatized or not self.answers:
            return None
        if not is_meaningful_text(replica):
            return None
        replica_vector = self.tfidf_vectorizer.transform([replica_lemmatized])
        similarities = cosine_similarity(replica_vector, self.tfidf_matrix).flatten()
        best_idx = similarities.argmax()
        if similarities[best_idx] > CONFIG['thresholds']['dialogues_similarity']:
            answer = self.answers[best_idx]
            logger.info(
                f"Found in dialogues.txt: replica='{replica_lemmatized}', answer='{answer}', similarity={similarities[best_idx]}")
            sentiment = analyze_sentiment(replica)
            if sentiment == 'positive':
                answer += " Рад, что ты в хорошем настроении! 😊"
            elif sentiment == 'negative':
                answer += " Кажется, ты не в духе. Может, новый авто поднимет настроение? 😊"
            if random.random() < 0.3:
                promo = random.choice(list(PRODUCTS_MAP.keys()))
                answer += f" Кстати, у нас есть {promo} — отличный выбор!"
            context.user_data['last_intent'] = 'offtopic'
            return answer
        logger.info(f"No match in dialogues.txt for replica='{replica_lemmatized}'")
        return None

    def get_failure_phrase(self, replica):
        """Фраза при неудаче с учётом тональности."""
        promo = random.choice(list(PRODUCTS_MAP.keys()))
        answer = random.choice(CONFIG['failure_phrases']).replace('[shoe_name]', promo)
        sentiment = analyze_sentiment(replica)
        if sentiment == 'positive':
            answer += " Вы в отличном настроении — давайте выберем! 😊"
        elif sentiment == 'negative':
            answer += " Не волнуйтесь, найдем лучшее! 😊"
        return answer
    def _process_none_state(self, replica, context):
        shoe = extract_shoe_name(replica)
        if shoe:
            context.user_data['current_shoe']=shoe
            context.user_data['state']=BotState.WAITING_FOR_INTENT.value
            suffix = " Рад! 😊" if analyze_sentiment(replica)=='positive' else " Давайте найдем. 😊"
            return f"Вы имеете в виду {shoe}? Что хотите узнать?{suffix}"
        category = extract_shoe_category(replica)
        if category:
            cand = [n for n,p in PRODUCTS_MAP.items() if category in p.get('categories',[])]
            if cand:
                shoe = random.choice(cand)
                context.user_data['current_shoe']=shoe
                context.user_data['state']=BotState.WAITING_FOR_INTENT.value
                suffix = " Отлично! 😊"
                return f"Из {category} есть {shoe}. Что узнать?{suffix}"
            return f"Нет моделей в категории {category}, попробуйте другую."
        intent = self.classify_intent(replica)
        if intent:
            return self.get_answer_by_intent(intent, replica, context)
        return self.generate_answer(replica, context) or self.get_failure_phrase(replica)

    def _process_waiting_for_item(self, replica, context):
        shoe = extract_shoe_name(replica)
        if shoe and shoe in PRODUCTS_MAP:
            context.user_data['current_shoe']=shoe
            context.user_data['state']=BotState.WAITING_FOR_INTENT.value
            return f"Вы имеете в виду {shoe}? Что хотите узнать?"
        category = extract_shoe_category(replica)
        if category:
            cand = [n for n,p in PRODUCTS_MAP.items() if category in p.get('categories',[])]
            if cand:
                shoe = random.choice(cand)
                context.user_data['current_shoe']=shoe
                context.user_data['state']=BotState.WAITING_FOR_INTENT.value
                return f"Из {category} есть {shoe}. Что хотите узнать?"
        return "Уточните модель обуви или категорию, пожалуйста."

    def _process_waiting_for_intent(self, replica, context):
        shoe = extract_shoe_name(replica) or context.user_data.get('current_shoe')
        intent = self.classify_intent(replica)
        if intent in [Intent.SHOE_PRICE.value, Intent.SHOE_AVAILABILITY.value, Intent.SHOE_INFO.value]:
            context.user_data['state']=BotState.NONE.value
            return self._get_shoe_response(intent, shoe, replica, context)
        return self.get_answer_by_intent(intent, replica, context) or "Что ещё?"

    def process(self, replica, context):
        stats = Stats(context)
        # 1) Проверка на осмысленность
        if not is_meaningful_text(replica):
            answer = self.get_failure_phrase(replica)
            stats.add(ResponseType.FAILURE.value, replica, answer, context)
            return answer
        
        # 2) Пытаемся понять явный интент
        intent = self.classify_intent(replica)
        if intent:
            answer = self.get_answer_by_intent(intent, replica, context)
            stats.add(ResponseType.INTENT.value, replica, answer, context)
            return answer

        # 3) Fallback: если нет интента — пробуем corpus (dialogues.txt)
        answer = self.generate_answer(replica, context)
        if answer:
            stats.add(ResponseType.GENERATE.value, replica, answer, context)
            return answer

        # 4) Фильтрация по цене — только если в запросе нет явного названия модели
        if not extract_shoe_name(replica):
            price = extract_price(replica)
            category = extract_shoe_category(replica)
            if price:
                answer = self._handle_filter_shoes(price, category, context)
                stats.add(ResponseType.INTENT.value, replica, answer, context)
                return answer

        state = context.user_data.get('state', BotState.NONE.value)
        if state == BotState.WAITING_FOR_ITEM.value:
            answer = self._process_waiting_for_item(replica, context)
        elif state == BotState.WAITING_FOR_INTENT.value:
            answer = self._process_waiting_for_intent(replica, context)
        else:
            answer = self._process_none_state(replica, context)

        stats.add(ResponseType.INTENT.value, replica, answer, context)
        return answer

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    answer = CONFIG['start_message']
    context.user_data['last_bot_response'] = answer
    context.user_data['last_intent'] = Intent.HELLO.value
    await update.message.reply_text(answer)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    answer = CONFIG['help_message']
    context.user_data['last_bot_response'] = answer
    context.user_data['last_intent'] = 'help'
    await update.message.reply_text(answer)

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats = context.user_data.get('stats', {ResponseType.INTENT.value: 0, ResponseType.GENERATE.value: 0,
                                            ResponseType.FAILURE.value: 0})
    answer = (
        f"Статистика:\n"
        f"Обработано намерений: {stats[ResponseType.INTENT.value]}\n"
        f"Ответов из диалогов: {stats[ResponseType.GENERATE.value]}\n"
        f"Неудачных запросов: {stats[ResponseType.FAILURE.value]}"
    )
    await update.message.reply_text(answer)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    if not user_text:
        answer = "Пожалуйста, отправьте текст."
        context.user_data['last_bot_response'] = answer
        await update.message.reply_text(answer)
        return
    bot = context.bot_data.setdefault('bot', Bot())
    answer = bot.process(user_text, context)
    shoe = context.user_data.get('current_shoe')
    last_photo = context.user_data.get('last_photo_shown_for')
    await update.message.reply_text(answer)
    if shoe and shoe in PRODUCTS_MAP and shoe != last_photo:
        url = PRODUCTS_MAP[shoe].get('image_url')
        if url:
            await update.message.reply_photo(photo=url)
            context.user_data['last_photo_shown_for'] = shoe

# Текст в голос
def text_to_voice(text):
    if not text:
        return None
    try:
        tts = gTTS(text=text, lang='ru')
        voice_file = 'response.mp3'
        tts.save(voice_file)
        return voice_file
    except Exception as e:
        logger.error(f"Ошибка синтеза речи: {e}\n{traceback.format_exc()}")
        return None

def voice_to_text(voice_file):
    recognizer = sr.Recognizer()
    try:
        import signal
        def signal_handler(signum, frame):
            raise TimeoutError("Speech recognition timed out")

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(5)  # Таймаут 5 секунд
        audio = AudioSegment.from_ogg(voice_file)
        audio.export('voice.wav', format='wav')
        with sr.AudioFile('voice.wav') as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language='ru-RU')
        return text
    except (sr.UnknownValueError, sr.RequestError, TimeoutError, Exception) as e:
        logger.error(f"Ошибка распознавания голоса: {e}\n{traceback.format_exc()}")
        return None
    finally:
        signal.alarm(0)
        if os.path.exists('voice.wav'):
            os.remove('voice.wav')

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice
    bot = context.bot_data.setdefault('bot', Bot())
    try:
        voice_file = await context.bot.get_file(voice.file_id)
        await voice_file.download_to_drive('voice.ogg')
        text = voice_to_text('voice.ogg')
        if text:
            answer = bot.process(text, context)
            voice_response = text_to_voice(answer)
            if voice_response:
                with open(voice_response, 'rb') as audio:
                    await update.message.reply_voice(audio)
                os.remove(voice_response)
            else:
                await update.message.reply_text(answer)
        else:
            answer = "Не удалось распознать голос. Попробуйте ещё раз."
            context.user_data['last_bot_response'] = answer
            await update.message.reply_text(answer)
    except Exception as e:
        logger.error(f"Ошибка обработки голосового сообщения: {e}\n{traceback.format_exc()}")
        answer = "Произошла ошибка. Попробуйте снова."
        context.user_data['last_bot_response'] = answer
        await update.message.reply_text(answer)
    finally:
        if os.path.exists('voice.ogg'):
            os.remove('voice.ogg')

# --- Запуск бота ---
def run_bot():
    if not TOKEN:
        raise ValueError('TELEGRAM_TOKEN не найден')
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('stats', stats_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    logger.info('Бот запускается...')
    app.run_polling()

if __name__ == '__main__':
    run_bot()