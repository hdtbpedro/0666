from fastapi import FastAPI, UploadFile
import pytesseract
import cv2
import numpy as np

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/process")
async def process(file: UploadFile):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)

    return {"text": text}
