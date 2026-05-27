"""
FastAPI 后端服务

功能说明：
    提供翻译 API 服务，包括：
    - 同步翻译接口
    - 流式翻译接口（WebSocket）
    - 批量翻译接口
    - 模型信息接口
    - 健康检查接口

接口列表：
    POST /translate      - 同步翻译
    WS   /stream         - 流式翻译
    POST /batch          - 批量翻译
    GET  /models         - 模型信息
    GET  /health         - 健康检查

依赖：
    - fastapi: Web 框架
    - uvicorn: ASGI 服务器
    - pydantic: 数据验证

作者：NMT Project
版本：1.0.0
"""

import os
import sys
import time
import json
import logging
import asyncio

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager

# FastAPI 相关
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent  # web/backend/main.py -> Graduation-Project
sys.path.insert(0, str(project_root))

# 导入推理引擎
try:
    from nmt.inference import PromptCache, CacheConfig
    from nmt.model.config import NMTModelManager
    HAS_LOCAL_MODULES = True
except ImportError:
    logging.warning("无法导入本地模块，部分功能受限")
    HAS_LOCAL_MODULES = False


# 导入 transformers
from transformers import MarianMTModel, MarianTokenizer
import torch


# ====================================
# 配置
# ====================================

# 默认模型路径（按优先级排序）
MODEL_SEARCH_PATHS = {
    "zh2en": [
        "outputs/quantized/zh2en/quantized",  # FP16量化模型（GPU加速）
        "outputs/compressed/zh2en/best",       # 剪枝+量化模型
        "outputs/models/zh2en/best",           # 训练最佳模型
        "models/Helsinki-NLP-opus-mt-zh-en",   # 预训练模型
    ],
    "en2zh": [
        "outputs/quantized/en2zh/quantized",  # FP16量化模型（GPU加速）
        "outputs/compressed/en2zh/best",       # 剪枝+量化模型
        "outputs/models/en2zh/best",           # 训练最佳模型
        "models/Helsinki-NLP-opus-mt-en-zh",   # 预训练模型
    ]
}

# 默认模型路径（向后兼容）
DEFAULT_ZH_EN_MODEL = "outputs/quantized/zh2en/quantized"
DEFAULT_EN_ZH_MODEL = "outputs/quantized/en2zh/quantized"

# 服务配置
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
MAX_BATCH_SIZE = 32
MAX_TEXT_LENGTH = 2000

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ====================================
# 数据模型
# ====================================

class TranslationRequest(BaseModel):
    """翻译请求"""
    text: str = Field(..., max_length=MAX_TEXT_LENGTH, description="待翻译文本")
    direction: str = Field(default="zh2en", description="翻译方向 (zh2en 或 en2zh)")
    use_cache: bool = Field(default=True, description="是否使用缓存")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "你好世界",
                "direction": "zh2en",
                "use_cache": True
            }
        }


class TranslationResponse(BaseModel):
    """翻译响应"""
    translation: str = Field(..., description="翻译结果")
    direction: str = Field(..., description="翻译方向")
    input_length: int = Field(default=0, description="输入文本长度")
    output_length: int = Field(default=0, description="输出文本长度")
    time_cost: float = Field(..., description="翻译耗时（秒）")
    from_cache: bool = Field(default=False, description="是否来自缓存")


class BatchTranslationRequest(BaseModel):
    """批量翻译请求"""
    texts: List[str] = Field(..., max_items=MAX_BATCH_SIZE, description="待翻译文本列表")
    direction: str = Field(default="zh2en", description="翻译方向")


class BatchTranslationResponse(BaseModel):
    """批量翻译响应"""
    translations: List[str] = Field(..., description="翻译结果列表")
    direction: str = Field(..., description="翻译方向")
    total_latency_ms: float = Field(..., description="总延迟（毫秒）")
    count: int = Field(..., description="翻译数量")


class ModelInfo(BaseModel):
    """模型信息"""
    name: str
    direction: str
    vocab_size: int
    max_length: int
    device: str
    loaded: bool
    quantized: bool = False
    dtype: str = "fp32"


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    models_loaded: Dict[str, bool]
    cache_stats: Optional[Dict[str, Any]] = None


# ====================================
# 翻译服务
# ====================================

class TranslationService:
    """
    翻译服务类
    
    管理模型加载和翻译请求处理。
    支持FP16量化模型GPU加速推理。
    """
    
    def __init__(
        self,
        zh_en_model_path: str = None,
        en_zh_model_path: str = None,
        enable_cache: bool = True,
        device: Optional[str] = None
    ):
        """初始化翻译服务"""
        self.enable_cache = enable_cache
        
        # 确定设备
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.info("使用 CPU")
        
        # 自动查找模型路径
        self.zh_en_model_path = zh_en_model_path or self._find_model("zh2en")
        self.en_zh_model_path = en_zh_model_path or self._find_model("en2zh")
        
        # 模型和分词器
        self.models: Dict[str, MarianMTModel] = {}
        self.tokenizers: Dict[str, MarianTokenizer] = {}
        
        # Prompt Cache
        self.cache = None
        if enable_cache and HAS_LOCAL_MODULES:
            try:
                self.cache = PromptCache(CacheConfig(max_size_mb=512))
            except Exception as e:
                logger.warning(f"初始化缓存失败: {e}")

        
        # 统计
        self.request_count = 0
        self.total_latency = 0.0
        
        # 量化信息
        self.model_info: Dict[str, Dict] = {}
    
    def _find_model(self, direction: str) -> Optional[str]:
        """自动查找可用模型"""
        # 检查环境变量
        env_model_path = os.environ.get("MODEL_PATH", "")
        if env_model_path:
            env_path = Path(env_model_path) / direction / "quantized"
            if env_path.exists() and (env_path / "config.json").exists():
                logger.info(f"使用环境变量模型路径: {env_path}")
                return str(env_path)
        
        # 按优先级搜索
        for path in MODEL_SEARCH_PATHS.get(direction, []):
            full_path = project_root / path
            if full_path.exists() and (full_path / "config.json").exists():
                logger.info(f"找到{direction}模型: {path}")
                return str(full_path)
        
        logger.warning(f"未找到{direction}模型")
        return None
    
    def _is_quantized_model(self, model_path: str) -> tuple:
        """检测模型是否为量化模型"""
        path = Path(model_path)
        stats_file = path / "quantization_stats.json"
        
        if stats_file.exists():
            import json
            try:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    dtype = stats.get("dtype", "")
                    gpu_compatible = stats.get("gpu_compatible", False)
                    return True, dtype, gpu_compatible
            except:
                pass
        
        # 检查模型文件是否是FP16
        model_file = path / "pytorch_model.bin"
        if model_file.exists():
            try:
                state_dict = torch.load(model_file, map_location='cpu')
                for key, tensor in state_dict.items():
                    if tensor.dtype == torch.float16:
                        return True, "fp16", True
                    break
            except:
                pass
        
        return False, None, False
    
    def load_models(self) -> None:
        """加载模型"""
        # 加载中→英模型
        if self.zh_en_model_path:
            self._load_model("zh2en", self.zh_en_model_path)
        else:
            logger.warning("中→英模型路径未设置")
        
        # 加载英→中模型
        if self.en_zh_model_path:
            self._load_model("en2zh", self.en_zh_model_path)
        else:
            logger.warning("英→中模型路径未设置")
    
    def _load_model(self, direction: str, model_path: str) -> None:
        """加载单个模型"""
        path = Path(model_path)
        if not path.exists():
            logger.warning(f"{direction}模型路径不存在: {model_path}")
            return
        
        try:
            # 检测是否为量化模型
            is_quantized, dtype, gpu_compatible = self._is_quantized_model(model_path)
            
            logger.info(f"加载{direction}模型: {model_path}")
            if is_quantized:
                logger.info(f"检测到{dtype.upper()}量化模型")
            
            # 加载分词器
            self.tokenizers[direction] = MarianTokenizer.from_pretrained(model_path)
            
            # 加载模型
            model = MarianMTModel.from_pretrained(model_path)
            
            # 处理FP16量化模型
            if is_quantized and dtype == "fp16" and gpu_compatible:
                if self.device.type == "cuda":
                    model = model.half().to(self.device)
                    logger.info(f"{direction} FP16量化模型运行在GPU上")
                else:
                    model = model.to(self.device)
                    logger.info(f"{direction} FP16量化模型运行在CPU上")
            else:
                model = model.to(self.device)
            
            model.eval()
            self.models[direction] = model
            
            # 保存模型信息
            self.model_info[direction] = {
                "quantized": is_quantized,
                "dtype": dtype if is_quantized else "fp32",
                "gpu_compatible": gpu_compatible if is_quantized else True
            }
            
            logger.info(f"{direction}模型加载完成")
            
        except Exception as e:
            logger.error(f"加载{direction}模型失败: {e}")
    
    def is_model_loaded(self, direction: str) -> bool:
        """检查模型是否已加载"""
        return direction in self.models
    
    @torch.no_grad()
    def translate(
        self,
        text: str,
        direction: str = "zh2en",
        use_cache: bool = True
    ) -> tuple:
        """
        翻译文本
        
        参数：
            text: 输入文本
            direction: 翻译方向
            use_cache: 是否使用缓存
            
        返回：
            tuple: (翻译结果, 延迟, 是否缓存命中)
        """
        # 检查缓存
        if use_cache and self.cache:
            cached_result = self.cache.get(f"{direction}:{text}")
            if cached_result:
                return cached_result, 0.0, True
        
        # 检查模型
        if direction not in self.models:
            raise ValueError(f"模型未加载: {direction}")
        
        model = self.models[direction]
        tokenizer = self.tokenizers[direction]
        
        # 计时
        start_time = time.perf_counter()
        
        # 分词
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # 生成
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_beams=4,
            early_stopping=True
        )
        
        # 解码
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 计时结束
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # 更新缓存
        if use_cache and self.cache:
            self.cache.put(f"{direction}:{text}", result)
        
        # 更新统计
        self.request_count += 1
        self.total_latency += latency_ms
        
        return result, latency_ms, False
    
    def batch_translate(
        self,
        texts: List[str],
        direction: str = "zh2en"
    ) -> tuple:
        """
        批量翻译
        
        参数：
            texts: 文本列表
            direction: 翻译方向
            
        返回：
            tuple: (翻译结果列表, 总延迟)
        """
        if direction not in self.models:
            raise ValueError(f"模型未加载: {direction}")
        
        model = self.models[direction]
        tokenizer = self.tokenizers[direction]
        
        start_time = time.perf_counter()
        
        # 分词
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # 生成
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_beams=4,
            early_stopping=True
        )
        
        # 解码
        results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return results, latency_ms
    
    def get_model_info(self, direction: str) -> Dict[str, Any]:
        """获取模型信息"""
        base_info = {
            "name": f"Helsinki-NLP-opus-mt-{direction.replace('2', '-')}",
            "direction": direction,
            "vocab_size": 0,
            "max_length": 512,
            "device": str(self.device),
            "loaded": False,
            "quantized": False,
            "dtype": "fp32"
        }
        
        if direction not in self.models:
            return base_info
        
        tokenizer = self.tokenizers[direction]
        base_info.update({
            "vocab_size": tokenizer.vocab_size,
            "max_length": 512,
            "device": str(self.device),
            "loaded": True
        })
        
        # 添加量化信息
        if direction in self.model_info:
            base_info.update(self.model_info[direction])
        
        return base_info
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计"""
        stats = {
            "request_count": self.request_count,
            "avg_latency_ms": self.total_latency / self.request_count if self.request_count > 0 else 0,
        }
        
        if self.cache:
            stats["cache"] = self.cache.get_stats()
        
        return stats


# ====================================
# FastAPI 应用
# ====================================

# 全局翻译服务
translation_service: Optional[TranslationService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global translation_service
    
    # 启动时加载模型
    logger.info("初始化翻译服务...")
    translation_service = TranslationService()
    translation_service.load_models()
    logger.info("翻译服务初始化完成")
    
    yield
    
    # 关闭时清理
    logger.info("关闭翻译服务...")


# 创建 FastAPI 应用
app = FastAPI(
    title="NMT 翻译服务",
    description="工业级中英双向神经网络翻译 API",
    version="1.0.0",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====================================
# API 端点
# ====================================

@app.get("/", tags=["Root"])
async def root():
    """根路径"""
    return {
        "name": "NMT 翻译服务",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    健康检查
    
    返回服务状态和模型加载状态。
    """
    global translation_service
    
    if translation_service is None:
        return HealthResponse(
            status="initializing",
            models_loaded={"zh2en": False, "en2zh": False}
        )
    
    # 动态检测模型状态
    models_status = {
        "zh2en": translation_service.is_model_loaded("zh2en"),
        "en2zh": translation_service.is_model_loaded("en2zh"),
    }
    
    return HealthResponse(
        status="healthy" if all(models_status.values()) else "partial_healthy",
        models_loaded=models_status,
        cache_stats=translation_service.get_stats().get("cache") if translation_service else None
    )



@app.post("/translate", response_model=TranslationResponse, tags=["Translation"])
async def translate(request: TranslationRequest):
    """
    同步翻译
    
    翻译单条文本，支持缓存。
    """
    global translation_service
    
    if translation_service is None:
        raise HTTPException(status_code=503, detail="服务初始化中")
    
    if not translation_service.is_model_loaded(request.direction):
        raise HTTPException(
            status_code=400,
            detail=f"模型未加载: {request.direction}"
        )
    
    try:
        result, latency_ms, cached = translation_service.translate(
            request.text,
            request.direction,
            request.use_cache
        )
        
        return TranslationResponse(
            translation=result,
            direction=request.direction,
            input_length=len(request.text),
            output_length=len(result),
            time_cost=latency_ms / 1000.0,  # 转换为秒
            from_cache=cached
        )
    except Exception as e:
        logger.error(f"翻译失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", response_model=BatchTranslationResponse, tags=["Translation"])
async def batch_translate(request: BatchTranslationRequest):
    """
    批量翻译
    
    翻译多条文本。
    """
    global translation_service
    
    if translation_service is None:
        raise HTTPException(status_code=503, detail="服务初始化中")
    
    if not translation_service.is_model_loaded(request.direction):
        raise HTTPException(
            status_code=400,
            detail=f"模型未加载: {request.direction}"
        )
    
    if len(request.texts) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"批量大小超过限制: {MAX_BATCH_SIZE}"
        )
    
    try:
        results, latency_ms = translation_service.batch_translate(
            request.texts,
            request.direction
        )
        
        return BatchTranslationResponse(
            translations=results,
            direction=request.direction,
            total_latency_ms=latency_ms,
            count=len(results)
        )
    except Exception as e:
        logger.error(f"批量翻译失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def get_models():
    """
    获取模型信息
    
    返回所有可用模型的信息。
    """
    global translation_service
    
    if translation_service is None:
        return []
    
    return [
        ModelInfo(**translation_service.get_model_info("zh2en")),
        ModelInfo(**translation_service.get_model_info("en2zh")),
    ]


@app.websocket("/stream")
async def websocket_translate(websocket: WebSocket):
    """
    流式翻译（WebSocket）
    
    支持实时翻译请求。
    """
    global translation_service
    
    await websocket.accept()
    
    try:
        while True:
            # 接收请求
            data = await websocket.receive_json()
            
            text = data.get("text", "")
            direction = data.get("direction", "zh2en")
            
            if not text:
                await websocket.send_json({"error": "文本不能为空"})
                continue
            
            if translation_service is None:
                await websocket.send_json({"error": "服务初始化中"})
                continue
            
            try:
                result, latency_ms, cached = translation_service.translate(
                    text, direction
                )
                
                await websocket.send_json({
                    "text": result,
                    "direction": direction,
                    "latency_ms": latency_ms,
                    "cached": cached
                })
            except Exception as e:
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        logger.info("WebSocket 连接断开")


@app.get("/stats", tags=["Stats"])
async def get_stats():
    """
    获取服务统计
    
    返回请求统计和缓存统计。
    """
    global translation_service
    
    if translation_service is None:
        return {"status": "initializing"}
    
    return translation_service.get_stats()


# ====================================
# DeepSeek 协同翻译
# ====================================

# DeepSeek API 配置（多级fallback：环境变量 > 硬编码默认值）
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "") or "sk-e88f9ef93099465c8a13ac792688e1a8"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"



class DeepSeekRequest(BaseModel):
    """DeepSeek翻译请求"""
    text: str = Field(..., description="待翻译文本")
    direction: str = Field(default="zh2en", description="翻译方向")
    termPrompt: Optional[str] = Field(default="", description="术语提示")

class DeepSeekResponse(BaseModel):
    """DeepSeek翻译响应"""
    translation: str = Field(..., description="翻译结果")
    time_cost: float = Field(..., description="翻译耗时")


async def call_deepseek_stream(
    text: str,
    direction: str,
    term_prompt: str = "",
    on_token: callable = None
) -> str:
    """
    调用DeepSeek API进行流式翻译
    
    参数：
        text: 待翻译文本
        direction: 翻译方向
        term_prompt: 术语提示
        on_token: Token回调函数
        
    返回：
        翻译结果
    """
    import httpx
    
    if not DEEPSEEK_API_KEY:
        raise ValueError("DeepSeek API Key未配置，请设置环境变量 DEEPSEEK_API_KEY")
    
    # 构建提示词
    if direction == "zh2en":
        system_prompt = "你是一个专业的中译英翻译助手。请将用户提供的中文文本翻译成自然、准确的英文。只输出翻译结果，不要添加任何解释或说明。"
    else:
        system_prompt = "你是一个专业的英译中翻译助手。请将用户提供的英文文本翻译成自然、准确的中文。只输出翻译结果，不要添加任何解释或说明。"
    
    if term_prompt:
        system_prompt += term_prompt
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "stream": True,
        "temperature": 0.3,
        "max_tokens": 2048
    }
    
    result_text = ""
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            async with client.stream(
                "POST",
                DEEPSEEK_API_URL,
                headers=headers,
                json=payload
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error(f"DeepSeek API 错误: {response.status_code} - {error_text.decode()}")
                    raise ValueError(f"DeepSeek API 错误: {response.status_code}")
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    result_text += content
                                    if on_token:
                                        await on_token(content)
                        except json.JSONDecodeError:
                            continue
        except httpx.ConnectError:
            logger.error("无法连接到 DeepSeek API，请检查网络连接")
            raise ValueError("网络连接失败，请检查是否可以访问 api.deepseek.com")
        except Exception as e:
            logger.error(f"请求 DeepSeek 发生异常: {str(e)}")
            raise e

    
    return result_text


@app.post("/translate-deepseek", response_model=DeepSeekResponse, tags=["DeepSeek"])
async def translate_with_deepseek(request: DeepSeekRequest):
    """
    使用DeepSeek进行翻译
    
    联网调用DeepSeek API进行高质量翻译。
    """
    start_time = time.perf_counter()
    
    try:
        result = await call_deepseek_stream(
            request.text,
            request.direction,
            request.termPrompt or ""
        )
        
        time_cost = time.perf_counter() - start_time
        
        return DeepSeekResponse(
            translation=result,
            time_cost=time_cost
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"DeepSeek翻译失败: {e}")
        raise HTTPException(status_code=500, detail=f"DeepSeek翻译失败: {str(e)}")


@app.websocket("/stream-deepseek")
async def stream_deepseek_translate(websocket: WebSocket):
    """
    DeepSeek流式翻译（WebSocket）
    
    支持逐Token返回翻译结果。
    """
    await websocket.accept()
    
    try:
        while True:
            # 接收请求
            data = await websocket.receive_json()
            
            text = data.get("text", "")
            direction = data.get("direction", "zh2en")
            term_prompt = data.get("termPrompt", "")
            
            if not text:
                await websocket.send_json({"type": "error", "message": "文本不能为空"})
                continue
            
            try:
                # Token回调
                async def on_token(content: str):
                    try:
                        await websocket.send_json({
                            "type": "token",
                            "content": content
                        })
                    except:
                        pass # 忽略发送失败
                
                result = await call_deepseek_stream(
                    text,
                    direction,
                    term_prompt,
                    on_token
                )
                
                # 发送完成信号
                await websocket.send_json({
                    "type": "done",
                    "content": result
                })

                
            except ValueError as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
            except Exception as e:
                logger.error(f"DeepSeek流式翻译失败: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"翻译失败: {str(e)}"
                })
                
    except WebSocketDisconnect:
        logger.info("DeepSeek WebSocket 连接断开")


@app.get("/deepseek-status", tags=["DeepSeek"])
async def deepseek_status():
    """
    检查DeepSeek API状态
    
    返回API配置状态。
    """
    return {
        "configured": bool(DEEPSEEK_API_KEY),
        "message": "DeepSeek API已配置" if DEEPSEEK_API_KEY else "未配置DeepSeek API Key，请设置环境变量 DEEPSEEK_API_KEY"
    }


# ====================================
# 启动入口
# ====================================

def start_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    reload: bool = False
):
    """
    启动服务器
    
    参数：
        host: 主机地址
        port: 端口号
        reload: 是否启用热重载
    """
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        workers=1
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NMT 翻译服务")
    parser.add_argument("--host", default=DEFAULT_HOST, help="主机地址")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="端口号")
    parser.add_argument("--reload", action="store_true", help="启用热重载")
    
    args = parser.parse_args()
    
    start_server(args.host, args.port, args.reload)
