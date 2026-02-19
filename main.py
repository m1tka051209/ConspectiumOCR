import os
import easyocr
import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

app = FastAPI(title="Конспектиум OCR API")

reader = easyocr.Reader(['ru'], gpu=False)

@app.get("/")
async def root():
    return {"message": "OCR сервер для Конспектиума работает!"}

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)
        
        result = reader.readtext(image_np, detail=0, paragraph=True)
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