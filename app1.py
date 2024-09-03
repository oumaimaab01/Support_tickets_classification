from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = FastAPI()

# Load the model and tokenizer
model_save_path = "./saved_model"
tokenizer = BertTokenizer.from_pretrained(model_save_path)
model = BertForSequenceClassification.from_pretrained(model_save_path)

class TicketRequest(BaseModel):
    body: str

@app.post("/predict/")
async def predict(request: TicketRequest):
    text = request.body
    try:
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)
            prediction = predictions.item()

        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
