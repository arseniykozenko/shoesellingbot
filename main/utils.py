# ./app/utils.py

import logging
import os
import json
from rapidfuzz import process, fuzz
from data.config import CONFIG
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Инициализация Natasha
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

# Загрузка тонального словаря
def load_tonal_dict():
    tonal_dict = {}
    try:
        with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'tonal_dict.txt'), encoding='utf-8') as f:
            for line in f:
                word, score = line.strip().split('\t')
                tonal_dict[word] = float(score)
    except FileNotFoundError:
        logger.error("Файл tonal_dict.txt не найден")
    return tonal_dict

TONAL_DICT = load_tonal_dict()

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PRODUCTS_PATH = os.path.join(DATA_DIR, 'products.json')
with open(PRODUCTS_PATH, encoding='utf-8') as f:
    PRODUCTS = json.load(f)
PRODUCTS_MAP = {item['name']: item for item in PRODUCTS}

# Очистка фразы
def clear_phrase(phrase):
    if not phrase:
        return ""
    phrase = phrase.lower()
    alphabet = '1234567890qwertyuiopasdfghjklzxcvbnmабвгдеёжзийклмнопрстуфхцчшщъыьэюя- '
    return ''.join(symbol for symbol in phrase if symbol in alphabet).strip()

# Лемматизация и морфологический анализ
def lemmatize_phrase(phrase):
    cleaned = clear_phrase(phrase)
    if not cleaned:
        return ""
    doc = Doc(cleaned)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    lemmas = []
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        lemmas.append(token.lemma or token.text)
    return ' '.join(lemmas)

# Анализ тональности
def analyze_sentiment(phrase):
    if not phrase:
        return 'neutral'
    words = lemmatize_phrase(phrase).split()
    score_sum = 0.0
    count = 0
    for w in words:
        if w in TONAL_DICT:
            score_sum += TONAL_DICT[w]
            count += 1
    if count == 0:
        return 'neutral'
    avg = score_sum / count
    if avg > 0.3:
        return 'positive'
    if avg < -0.3:
        return 'negative'
    return 'neutral'

# Проверка на осмысленность текста
def is_meaningful_text(text):
    txt = clear_phrase(text)
    return any(len(w) > 2 and all(c in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя' for c in w) for w in txt.split())

# Извлечение цены
def extract_price(replica):
    text = clear_phrase(replica)
    logger.info(f"Extracting price from: '{text}'")
    for token in text.split():
        if token.isdigit():
            return int(token)
    return None

# Извлечение названия модели обуви
def extract_shoe_name(replica):
    txt = lemmatize_phrase(replica)
    if not txt:
        return None
    # Точное совпадение названия
    for name in PRODUCTS_MAP:
        if lemmatize_phrase(name) in txt:
            return name
    # Синонимы и fuzzy match
    for name, data in PRODUCTS_MAP.items():
        # По синонимам
        for syn in data.get('synonyms', []):
            if lemmatize_phrase(syn) in txt:
                return name
        # fuzzy matching
        candidates = [name] + data.get('synonyms', [])
        best = process.extractOne(txt, candidates, scorer=fuzz.partial_ratio)
        if best and best[1] >= CONFIG['thresholds']['fuzzy_match']:
            return name
    return None

# Извлечение категории обуви
def extract_shoe_category(replica):
    text = lemmatize_phrase(replica)
    if not text:
        return None

    # Собираем все уникальные категории из каталога
    all_categories = set()
    for item in PRODUCTS_MAP.values():
        for cat in item.get('categories', []):
            all_categories.add(cat)

    # 1) Ищем прямое вхождение названия категории
    for cat in all_categories:
        cat_lem = lemmatize_phrase(cat)
        if cat_lem and cat_lem in text:
            return cat

    # 2) Если прямого нет — проверяем синонимы категорий
    for item in PRODUCTS_MAP.values():
        for cat, syns in item.get('category_synonyms', {}).items():
            for syn in syns:
                syn_lem = lemmatize_phrase(syn)
                if syn_lem and syn_lem in text:
                    return cat

    return None

# (Опционально) Извлечение бренда
def extract_brand(replica):
    txt = lemmatize_phrase(replica)
    for data in PRODUCTS_MAP.values():
        brand = data.get('brand')
        if brand and lemmatize_phrase(brand) in txt:
            return brand
    return None

# Класс для статистики
class Stats:
    def __init__(self, context):
        self.context = context
        if 'stats' not in context.user_data:
            context.user_data['stats'] = {'intent': 0, 'generate': 0, 'failure': 0}
        self.stats = context.user_data['stats']

    def add(self, type, replica, answer, context):
        """Обновляет статистику, сохраняет её в context и логирует."""
        if type in self.stats:
            self.stats[type] += 1
        else:
            self.stats[type] = 1
        self.context.user_data['stats'] = self.stats
        logger.info(f"Stats: {self.stats} | Вопрос: {replica} | Ответ: {answer}")