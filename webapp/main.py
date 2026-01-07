from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException
import base64
import os

from wine_pairing import wine_pairing  

app = FastAPI(
    title="Wine Pairing API",
    description="요리 이미지 URL을 기반으로 와인을 추천해주는 API 서비스입니다.",
    version="1.0.0"
)

# 템플릿 & 정적 파일 설정
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# 응답 모델 정의
# class WinePairingResponse(BaseModel):
#     recommend_wine: str
#     recommend_reason: str

# https://sitem.ssgcdn.com/95/55/96/item/1000346965595_i1_750.jpg
# ✅ 메인 페이지 (프론트엔드)
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# ✅ 이미지 업로드 처리
@app.post("/wine-pairing")
async def wine_pairing_api(file: UploadFile = File(...)):
    try:
        # 1. 이미지 파일 읽기
        image_bytes = await file.read()

        # 2. base64 인코딩
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # 3. 핵심 로직 호출 (경로 ❌, base64 ✅)
        result = wine_pairing(image_base64)

        return {
            "filename": file.filename,
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

