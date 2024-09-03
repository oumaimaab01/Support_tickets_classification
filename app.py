from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pickle
from starlette.requests import Request

app = FastAPI()

# Load the model
with open('outputs/ticket_type.model', 'rb') as model_file:
    model = pickle.load(model_file)

class TicketRequest(BaseModel):
    body: str

@app.post("/predict/")
async def predict(request: TicketRequest):
    text = request.body
    try:
        prediction = model.predict([text])
        return {"prediction": prediction[0]}
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
