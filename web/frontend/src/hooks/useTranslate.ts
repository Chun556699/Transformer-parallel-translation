/**
 * 翻译 Hook
 * 
 * 功能说明：
 *   - 封装翻译状态管理
 *   - 处理翻译请求和错误
 *   - 管理翻译历史
 * 
 * 依赖：
 *   - React 18
 *   - 翻译 API 服务
 */

import { useState, useCallback, useRef, useEffect } from 'react'
import { translate as translateApi, ApiError } from '../api/translate'

// ============================================================================
// 类型定义
// ============================================================================

/**
 * 翻译历史记录
 */
export interface TranslationHistory {
  /** 唯一标识 */
  id: string
  /** 源文本 */
  sourceText: string
  /** 翻译结果 */
  targetText: string
  /** 翻译方向 */
  direction: 'zh2en' | 'en2zh'
  /** 翻译时间 */
  timestamp: Date
  /** 耗时（秒） */
  timeCost: number
  /** 是否来自缓存 */
  fromCache: boolean
}

/**
 * 翻译状态
 */
export interface TranslateState {
  /** 是否正在翻译 */
  loading: boolean
  /** 翻译结果 */
  result: string
  /** 错误信息 */
  error: string | null
  /** 耗时（秒） */
  timeCost: number | null
  /** 是否来自缓存 */
  fromCache: boolean
}

/**
 * 翻译 Hook 返回值
 */
export interface UseTranslateReturn {
  /** 翻译状态 */
  state: TranslateState
  /** 执行翻译 */
  translate: (text: string, direction: 'zh2en' | 'en2zh') => Promise<void>
  /** 清除结果 */
  clear: () => void
  /** 翻译历史 */
  history: TranslationHistory[]
  /** 清除历史 */
  clearHistory: () => void
}

// ============================================================================
// 常量
// ============================================================================

/** 历史记录最大数量 */
const MAX_HISTORY_SIZE = 50

/** localStorage 键名 */
const HISTORY_STORAGE_KEY = 'nmt_translation_history'

// ============================================================================
// Hook 实现
// ============================================================================

/**
 * 翻译 Hook
 * 
 * @returns 翻译状态和操作函数
 * 
 * @example
 * ```tsx
 * const { state, translate, clear, history } = useTranslate()
 * 
 * const handleTranslate = async () => {
 *   await translate('你好世界', 'zh2en')
 * }
 * ```
 */
export function useTranslate(): UseTranslateReturn {
  // 翻译状态
  const [state, setState] = useState<TranslateState>({
    loading: false,
    result: '',
    error: null,
    timeCost: null,
    fromCache: false,
  })
  
  // 翻译历史
  const [history, setHistory] = useState<TranslationHistory[]>(() => {
    // 从 localStorage 恢复历史记录
    try {
      const saved = localStorage.getItem(HISTORY_STORAGE_KEY)
      if (saved) {
        const parsed = JSON.parse(saved)
        return parsed.map((item: TranslationHistory) => ({
          ...item,
          timestamp: new Date(item.timestamp),
        }))
      }
    } catch (error) {
      console.warn('恢复翻译历史失败:', error)
    }
    return []
  })
  
  // 用于取消请求的 ref
  const abortControllerRef = useRef<AbortController | null>(null)
  
  // 持久化历史记录
  useEffect(() => {
    try {
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history))
    } catch (error) {
      console.warn('保存翻译历史失败:', error)
    }
  }, [history])
  
  /**
   * 执行翻译
   */
  const translate = useCallback(async (
    text: string,
    direction: 'zh2en' | 'en2zh'
  ): Promise<void> => {
    // 验证输入
    const trimmedText = text.trim()
    if (!trimmedText) {
      setState((prev: TranslateState) => ({
        ...prev,
        error: '请输入要翻译的文本',
      }))
      return
    }
    
    // 取消之前的请求
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    abortControllerRef.current = new AbortController()
    
    // 开始翻译
    setState({
      loading: true,
      result: '',
      error: null,
      timeCost: null,
      fromCache: false,
    })
    
    try {
      const response = await translateApi({
        text: trimmedText,
        direction,
        use_cache: true,
      })
      
      // 更新状态
      setState({
        loading: false,
        result: response.translation,
        error: null,
        timeCost: response.time_cost,
        fromCache: response.from_cache,
      })
      
      // 添加到历史记录
      const historyItem: TranslationHistory = {
        id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        sourceText: trimmedText,
        targetText: response.translation,
        direction,
        timestamp: new Date(),
        timeCost: response.time_cost,
        fromCache: response.from_cache,
      }
      
      setHistory((prev: TranslationHistory[]) => {
        const newHistory = [historyItem, ...prev]
        // 限制历史记录数量
        if (newHistory.length > MAX_HISTORY_SIZE) {
          newHistory.pop()
        }
        return newHistory
      })
      
    } catch (error) {
      // 忽略取消的请求
      if (error instanceof Error && error.name === 'AbortError') {
        return
      }
      
      // 处理错误
      let errorMessage = '翻译失败，请稍后重试'
      if (error instanceof ApiError) {
        errorMessage = error.detail || error.message
      } else if (error instanceof Error) {
        errorMessage = error.message
      }
      
      setState({
        loading: false,
        result: '',
        error: errorMessage,
        timeCost: null,
        fromCache: false,
      })
    }
  }, [])
  
  /**
   * 清除翻译结果
   */
  const clear = useCallback(() => {
    // 取消正在进行的请求
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    
    setState({
      loading: false,
      result: '',
      error: null,
      timeCost: null,
      fromCache: false,
    })
  }, [])
  
  /**
   * 清除历史记录
   */
  const clearHistory = useCallback(() => {
    setHistory([])
    localStorage.removeItem(HISTORY_STORAGE_KEY)
  }, [])
  
  // 组件卸载时取消请求
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
    }
  }, [])
  
  return {
    state,
    translate,
    clear,
    history,
    clearHistory,
  }
}

export default useTranslate
