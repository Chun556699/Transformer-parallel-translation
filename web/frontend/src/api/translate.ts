/**
 * 翻译 API 服务
 * 
 * 功能说明：
 *   - 封装与后端翻译服务的通信
 *   - 支持同步翻译和流式翻译
 *   - 提供错误处理和重试机制
 * 
 * 依赖：
 *   - FastAPI 后端服务
 */

// ============================================================================
// 类型定义
// ============================================================================

/**
 * 翻译请求参数
 */
export interface TranslateRequest {
  /** 待翻译文本 */
  text: string
  /** 翻译方向：zh2en 或 en2zh */
  direction: 'zh2en' | 'en2zh'
  /** 是否使用缓存 */
  use_cache?: boolean
}

/**
 * 翻译响应结果
 */
export interface TranslateResponse {
  /** 翻译结果 */
  translation: string
  /** 翻译方向 */
  direction: string
  /** 输入文本长度 */
  input_length: number
  /** 输出文本长度 */
  output_length: number
  /** 翻译耗时（秒） */
  time_cost: number
  /** 是否来自缓存 */
  from_cache: boolean
}

/**
 * 批量翻译响应
 */
export interface BatchTranslateResponse {
  /** 翻译结果列表 */
  translations: string[]
  /** 总耗时（秒） */
  total_time: number
  /** 各条翻译耗时 */
  individual_times: number[]
}

/**
 * 模型信息
 */
export interface ModelInfo {
  /** 已加载的模型 */
  loaded_models: string[]
  /** 设备信息 */
  device: string
  /** 缓存统计 */
  cache_stats: {
    zh2en: { size: number; max_size: number; hit_rate: number }
    en2zh: { size: number; max_size: number; hit_rate: number }
  }
}

/**
 * API 错误
 */
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public detail?: string
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

// ============================================================================
// API 配置
// ============================================================================

/** API 基础路径（开发环境使用代理） */
const API_BASE = '/api'

/** 请求超时时间（毫秒） */
const REQUEST_TIMEOUT = 30000

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * 带超时的 fetch 请求
 * 
 * @param url - 请求 URL
 * @param options - 请求选项
 * @param timeout - 超时时间（毫秒）
 * @returns 响应对象
 */
async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeout: number = REQUEST_TIMEOUT
): Promise<Response> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeout)
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    })
    return response
  } finally {
    clearTimeout(timeoutId)
  }
}

/**
 * 处理 API 响应
 * 
 * @param response - fetch 响应对象
 * @returns 解析后的 JSON 数据
 * @throws ApiError - 当响应状态码非 2xx 时
 */
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail: string | undefined
    try {
      const errorData = await response.json()
      detail = errorData.detail || errorData.message
    } catch {
      // 忽略 JSON 解析错误
    }
    throw new ApiError(
      `请求失败: ${response.statusText}`,
      response.status,
      detail
    )
  }
  return response.json()
}

// ============================================================================
// API 函数
// ============================================================================

/**
 * 同步翻译
 * 
 * @param request - 翻译请求参数
 * @returns 翻译响应结果
 */
export async function translate(
  request: TranslateRequest
): Promise<TranslateResponse> {
  const response = await fetchWithTimeout(
    `${API_BASE}/translate`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    }
  )
  return handleResponse<TranslateResponse>(response)
}

/**
 * 批量翻译
 * 
 * @param texts - 待翻译文本列表
 * @param direction - 翻译方向
 * @returns 批量翻译响应
 */
export async function batchTranslate(
  texts: string[],
  direction: 'zh2en' | 'en2zh'
): Promise<BatchTranslateResponse> {
  const response = await fetchWithTimeout(
    `${API_BASE}/batch`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ texts, direction }),
    },
    60000 // 批量翻译超时时间较长
  )
  return handleResponse<BatchTranslateResponse>(response)
}

/**
 * 获取模型信息
 * 
 * @returns 模型信息
 */
export async function getModels(): Promise<ModelInfo> {
  const response = await fetchWithTimeout(
    `${API_BASE}/models`,
    { method: 'GET' }
  )
  return handleResponse<ModelInfo>(response)
}

/**
 * 健康检查
 * 
 * @returns 服务状态
 */
export async function healthCheck(): Promise<{ status: string; timestamp: string }> {
  const response = await fetchWithTimeout(
    `${API_BASE}/health`,
    { method: 'GET' },
    5000 // 健康检查超时较短
  )
  return handleResponse<{ status: string; timestamp: string }>(response)
}

/**
 * 创建流式翻译 WebSocket 连接
 * 
 * @param onMessage - 收到消息时的回调
 * @param onError - 发生错误时的回调
 * @param onClose - 连接关闭时的回调
 * @returns WebSocket 实例和发送函数
 */
export function createStreamConnection(
  onMessage: (data: { token: string; done: boolean }) => void,
  onError?: (error: Event) => void,
  onClose?: () => void
): {
  send: (text: string, direction: 'zh2en' | 'en2zh') => void
  close: () => void
} {
  // 构建 WebSocket URL
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const wsUrl = `${wsProtocol}//${window.location.host}${API_BASE}/stream`
  
  const ws = new WebSocket(wsUrl)
  
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      onMessage(data)
    } catch (error) {
      console.error('解析 WebSocket 消息失败:', error)
    }
  }
  
  ws.onerror = (error) => {
    console.error('WebSocket 错误:', error)
    onError?.(error)
  }
  
  ws.onclose = () => {
    onClose?.()
  }
  
  return {
    send: (text: string, direction: 'zh2en' | 'en2zh') => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ text, direction }))
      }
    },
    close: () => {
      ws.close()
    },
  }
}
