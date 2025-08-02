import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import speech_recognition as sr
from googletrans import Translator
import random

words_by_level = {
    "A1": {
        "мама": "mother", "папа": "father", "кот": "cat", "собака": "dog", "яблоко": "apple",
        "молоко": "milk", "вода": "water", "дом": "house", "друг": "friend", "книга": "book",
        "стол": "table", "окно": "window", "день": "day", "ночь": "night", "еда": "food",
        "ребёнок": "child", "мяч": "ball", "машина": "car", "школа": "school", "учитель": "teacher"
    },
    "A2": {
        "велосипед": "bicycle", "письмо": "letter", "город": "city", "музыка": "music", "магазин": "shop",
        "работа": "job", "семья": "family", "погода": "weather", "сестра": "sister", "брат": "brother",
        "солнце": "sun", "комната": "room", "поезд": "train", "река": "river", "здание": "building",
        "подарок": "gift", "отпуск": "vacation", "картина": "picture", "телефон": "phone", "деньги": "money"
    },
    "B1": {
        "путешествие": "journey", "здоровье": "health", "разговор": "conversation", "решение": "decision", "успех": "success",
        "проблема": "problem", "история": "story", "опыт": "experience", "мнение": "opinion", "удивление": "surprise",
        "безопасность": "safety", "влияние": "influence", "выбор": "choice", "ошибка": "mistake", "правда": "truth",
        "тренировка": "training", "эмоция": "emotion", "качество": "quality", "отношения": "relationship", "внимание": "attention"
    },
    "B2": {
        "ответственность": "responsibility", "достижение": "achievement", "обсуждение": "discussion", "вдохновение": "inspiration", "сотрудничество": "cooperation",
        "мотивация": "motivation", "доверие": "trust", "планирование": "planning", "разработка": "development", "перспектива": "perspective",
        "реализация": "implementation", "реформы": "reforms", "анализ": "analysis", "стратегия": "strategy", "влияние": "impact",
        "потенциал": "potential", "модель": "model", "организация": "organization", "приоритет": "priority", "решимость": "determination"
    },
    "C1": {
        "восприятие": "perception", "устойчивость": "resilience", "предпринимательство": "entrepreneurship", "осведомленность": "awareness", "взаимозависимость": "interdependence",
        "последовательность": "consistency", "предположение": "assumption", "убеждение": "conviction", "соображение": "consideration", "рациональность": "rationality",
        "ограничение": "limitation", "двусмысленность": "ambiguity", "структура": "framework", "контекст": "context", "совместимость": "compatibility",
        "внедрение": "integration", "антипатия": "antipathy", "анализ": "evaluation", "сознательность": "mindfulness", "эксплуатация": "utilization"
    },
    "C2": {
        "самоактуализация": "self-actualization", "мировоззрение": "worldview", "интуиция": "intuition", "осмысление": "comprehension", "совершенствование": "refinement",
        "антропоцентризм": "anthropocentrism", "когнитивность": "cognition", "онтология": "ontology", "экзистенция": "existence", "детерминизм": "determinism",
        "интерпретация": "interpretation", "менталитет": "mentality", "трансцендентность": "transcendence", "дискурсивность": "discursiveness", "парадигма": "paradigm",
        "синергия": "synergy", "эмпатия": "empathy", "самоидентичность": "self-identity", "интеллектуальность": "intellectuality", "доказательство": "justification"
    }
}

sentences_by_level = {
    "A1": {
        "Я люблю свою маму.": "i love my mother",
        "У меня есть собака.": "i have a dog",
        "Это мой дом.": "this is my house",
        "Я читаю книгу.": "i am reading a book",
        "Он пьёт воду.": "he drinks water",
        "Мы играем в мяч.": "we are playing ball",
        "Она идёт в школу.": "she is going to school",
        "Я кушаю яблоко.": "i am eating an apple",
        "Это мой друг.": "this is my friend",
        "Кошка на окне.": "the cat is on the window"
    },
    "A2": {
        "Сегодня хорошая погода.": "the weather is good today",
        "Я поехал на велосипеде.": "i rode a bicycle",
        "Мой брат работает в магазине.": "my brother works at a shop",
        "У нас большая семья.": "we have a big family",
        "Мы едем на поезде.": "we are going by train",
        "Я отправил письмо.": "i sent a letter",
        "Моя сестра рисует картину.": "my sister is drawing a picture",
        "Он слушает музыку.": "he is listening to music",
        "У нас нет денег.": "we don't have money",
        "Это красивое здание.": "this is a beautiful building"
    },
    "B1": {
        "Я принял важное решение.": "i made an important decision",
        "Путешествие помогает расширить кругозор.": "traveling helps to broaden the mind",
        "Здоровье — это богатство.": "health is wealth",
        "У нас был интересный разговор.": "we had an interesting conversation",
        "Она выразила своё мнение.": "she expressed her opinion",
        "Эта ошибка была неожиданной.": "this mistake was unexpected",
        "Тренировки помогают улучшить форму.": "training helps improve fitness",
        "Безопасность важна для всех.": "safety is important for everyone",
        "Я получил новый опыт.": "i gained new experience",
        "Он привлёк внимание аудитории.": "he attracted the audience's attention"
    },
    "B2": {
        "Ответственность — важное качество.": "responsibility is an important quality",
        "Обсуждение было продуктивным.": "the discussion was productive",
        "Мы достигли большого успеха.": "we achieved great success",
        "Эта идея вдохновила меня.": "this idea inspired me",
        "Сотрудничество помогает развиваться.": "cooperation helps development",
        "Мотивация приходит изнутри.": "motivation comes from within",
        "Он доверяет своей команде.": "he trusts his team",
        "Планирование помогает избежать ошибок.": "planning helps to avoid mistakes",
        "Реформа изменила ситуацию.": "the reform changed the situation",
        "Они разработали новую стратегию.": "they developed a new strategy"
    },
    "C1": {
        "Восприятие зависит от контекста.": "perception depends on context",
        "Устойчивость помогает преодолевать трудности.": "resilience helps overcome difficulties",
        "Предпринимательство требует смелости.": "entrepreneurship requires courage",
        "Осведомлённость улучшает принятие решений.": "awareness improves decision-making",
        "Социальная взаимозависимость усиливает прогресс.": "social interdependence strengthens progress",
        "Предположения нужно проверять.": "assumptions should be verified",
        "Он выразил своё убеждение.": "he expressed his conviction",
        "Ограничения формируют рамки.": "limitations form boundaries",
        "Рамки мышления влияют на результат.": "frameworks influence outcomes",
        "Контекст имеет значение.": "context matters"
    },
    "C2": {
        "Самоактуализация — высшая человеческая потребность.": "self-actualization is the highest human need",
        "Мировоззрение влияет на мышление.": "worldview influences thinking",
        "Интуиция дополняет логическое мышление.": "intuition complements logical thinking",
        "Осмысление требует времени и усилий.": "comprehension requires time and effort",
        "Совершенствование требует терпения.": "refinement requires patience",
        "Трансцендентность выходит за пределы опыта.": "transcendence goes beyond experience",
        "Парадигмы меняются со временем.": "paradigms change over time",
        "Дискурс формирует реальность.": "discourse shapes reality",
        "Интеллектуальность не гарантирует мудрость.": "intellectuality does not guarantee wisdom",
        "Эмпатия — основа человеческой связи.": "empathy is the basis of human connection"
    }
}

def record_and_recognize(duration=10, sample_rate=44100):
    print("🎙 Запись началась. Говори, у тебя есть 10 секунд!")
    recording = sd.rec(
        int(duration * sample_rate), 
        samplerate=sample_rate, 
        channels=1, 
        dtype='int16')
    sd.wait()
    wav.write('output.wav', sample_rate, recording)
    print("⏳ Распознаём речь...")
    recognizer = sr.Recognizer()
    with sr.AudioFile('output.wav') as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language="en")  # ожидаем, что игрок говорит по-английски
            return text.lower().strip()
        except sr.UnknownValueError:
            return "[непонятно]"
        except sr.RequestError as e:
            print(f"❌ Ошибка сервиса распознавания: {e}")
            return "[ошибка]"

def play_game():
    print("👋 Привет! Это игра на произношение английских переводов.")
    print("Ты выбираешь, что хочешь переводить: слова или предложения, и уровень сложности.")
    print("У тебя 3 жизни. За каждую ошибку теряешь одну. 🍀 Удачи!\n")

    main = input("Что хочешь тренировать — слова или предложения? ").lower().strip()
    level = input("Выбери уровень (A1, A2, B1, B2, C1, C2): ").upper()
    chose = input("Проверка произношения или перевод? (Подсказка: напиши проверка или перевод) ").lower().strip()

    if main not in ["слова", "предложения"] or level not in words_by_level or chose not in ["перевод", "проверка"]:
        print("⚠️ Неправильный ввод. Перезапусти игру и выбери корректно.")
        return

    data = words_by_level[level] if main == "слова" else sentences_by_level[level]
    items = list(data.items())

    lives = 3
    score = 0

    while lives > 0:
        russian, correct_english = random.choice(items)
        if chose=="проверкаслова":
            text = f"\n🔤 Произнеси: **{correct_english}**"
        else:
            text = f"\n🔤 Переведи на английский: **{russian}**"
        print(text)

        result = record_and_recognize()
        print(f"📢 Ты сказал: {result}")
        print(f"✅ Правильный перевод: {correct_english}")

        if result == correct_english.lower().strip():
            print("🎉 Верно!")
            score += 1
        else:
            print("❌ Неверно!")
            lives -= 1
            print(f"❤️ Осталось жизней: {lives}")

    print("\n💀 Game Over!")
    print(f"🏆 Твой финальный счёт: {score}")

play_game()