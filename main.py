import os
import easyocr
import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

app = FastAPI(title="Конспектиум OCR API")

# Загружаем модель EasyOCR (только русский язык)
# Первый запуск может быть долгим — скачивается модель
reader = easyocr.Reader(['ru'], gpu=False)

@app.get("/")
async def root():
    return {"message": "OCR сервер для Конспектиума работает!"}

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    try:
        # Читаем загруженный файл
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Конвертируем PIL Image в numpy array для EasyOCR
        image_np = np.array(image)
        
        # Распознаём текст
        result = reader.readtext(image_np, detail=0, paragraph=True)
        
        # Объединяем все строки в один текст
        full_text = "\n".join(result)
        
        return JSONResponse(content={
            "success": True,
            "text": full_text,
            "lines": len(result)
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)