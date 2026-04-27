import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

# 初始化大模型客户端
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_CHAT_MODEL = "qwen-plus"

api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise RuntimeError("没有读取到环境变量 DASHSCOPE_API_KEY")

client = OpenAI(api_key=api_key, base_url=QWEN_BASE_URL)

# 定义请求体格式
class ChatRequest(BaseModel):
    message: str

# 聊天接口
@app.post("/chat")
async def chat(request: ChatRequest):
    completion = client.chat.completions.create(
        model=QWEN_CHAT_MODEL,
        messages=[
            {"role": "system", "content": "你是自然语言处理课程助教，回答要准确、简洁。"},
            {"role": "user", "content": request.message}
        ],
        temperature=0.3,
    )
    return {"reply": completion.choices[0].message.content}

# 根路由，返回聊天页面
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)