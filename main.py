from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
from random import randint

import requests
import torch


# Переменные
box_width = 2
box_color = []
fill_color = "red"
text_color = "white"
font_size = 20

for i in range(20):
    box_color.append('#%06X' % randint(0, 0xFFFFFF))


# Создаем модель данных которые будем принимать
class Item(BaseModel):
    url: str

# Запускаем FastApi
app = FastAPI()

# Первый роут 
@app.get("/")
def root():
    return {"message": "Hello World"}

# Роут по которому происходит отправка  и получение информации от модели машинного обучения
@app.post("/process/")
def predict(Item: Item):
    """Анализ изрбражения"""

    # Загружаем изображение
    url = Item.url
    image = Image.open(url)

    # Подключаем модель Обноружения предметов (Object Detection)
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    # Берем вхождения > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # Рисуем на изображении 
    draw = ImageDraw.Draw(image)
    # Выбираем шрифт
    font = ImageFont.truetype("./Arial.ttf", size=font_size)
    
    # перебираем все вхождения
    for score, label, box, colors in zip(results["scores"], results["labels"], results["boxes"], box_color):
        box = [round(i, 2) for i in box.tolist()]
        # Пишем лог что нашли
        #print(
        #        f"Нашли {model.config.id2label[label.item()]} с уверенностью "
        #       f"{round(score.item(), 3)} в квадрате {box}"
        #)
        # вычисляем квадрат с найденым объектом
        x, y, x_max, y_max = box
        # рисуем квадрат на найденом объекте
        draw.rectangle([x, y, x_max, y_max], outline=colors, width=box_width)

        # высчитываем размер текста
        text_width, text_height = font.getsize(model.config.id2label[label.item()])
        # рисуем подложку для текста
        draw.rectangle([x, y, x + text_width + 6, y + text_height], fill=fill_color)
        # рисуем текст
        draw.text((x, y), model.config.id2label[label.item()], fill=text_color, font=font)

    # сохроняем изображение с найдеными объектами
    image.save("./output.jpg")
    # Показываем изображение

    return {'image': 'output.jpg'}
