/**
 * 双协同翻译器组件
 * 
 * 功能说明：
 *   - 本地NMT模型翻译
 *   - DeepSeek API 联网翻译
 *   - 双协同翻译结果对比
 *   - 流式Token显示
 *   - 术语库集成
 */

import React, { useState, useCallback, useEffect, useRef } from 'react'
import { Term, TermManager } from './TermManager'

// ============================================================================
// 类型定义
// ============================================================================

type TranslationDirection = 'zh2en' | 'en2zh'

interface TranslationResult {
  text: string
  source: 'nmt' | 'deepseek'
  timeCost: number
  streaming?: boolean
}

interface StreamMessage {
  type: 'token' | 'done' | 'error'
  content?: string
  message?: string
}

// ============================================================================
// 常量
// ============================================================================

const API_BASE = '/api'
const TERMS_STORAGE_KEY = 'nmt_term_database'

const LANGUAGES = {
  zh: { code: 'zh', name: '中文', flag: '🇨🇳' },
  en: { code: 'en', name: 'English', flag: '🇺🇸' }
}

// ============================================================================
// 图标组件
// ============================================================================

const SwapIcon = () => (
  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M7 16V4m0 0L3 8m4-4l4 4M17 8v12m0 0l4-4m-4 4l-4-4" />
  </svg>
)

const CopyIcon = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
    <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
  </svg>
)

const ClearIcon = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M18 6L6 18M6 6l12 12" />
  </svg>
)

const BookIcon = () => (
  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
  </svg>
)

const SparklesIcon = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
  </svg>
)

// ============================================================================
// 主组件
// ============================================================================

export const Translator: React.FC = () => {
  // 状态
  const [sourceText, setSourceText] = useState('')
  const [direction, setDirection] = useState<TranslationDirection>('zh2en')
  const [showTermManager, setShowTermManager] = useState(false)
  const [terms, setTerms] = useState<Term[]>([])
  
  // 翻译模式
  const [dualMode, setDualMode] = useState(false)  // 双协同模式
  const [useDeepSeek, setUseDeepSeek] = useState(false)
  
  // 翻译结果
  const [nmtResult, setNmtResult] = useState<TranslationResult | null>(null)
  const [deepseekResult, setDeepseekResult] = useState<TranslationResult | null>(null)
  
  // 流式显示
  const [nmtStreaming, setNmtStreaming] = useState(false)
  const [deepseekStreaming, setDeepseekStreaming] = useState(false)
  const [nmtStreamText, setNmtStreamText] = useState('')
  const [deepseekStreamText, setDeepseekStreamText] = useState('')
  
  // 错误
  const [error, setError] = useState<string | null>(null)
  
  // Refs
  const wsRef = useRef<WebSocket | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  
  // 语言
  const sourceLang = direction === 'zh2en' ? LANGUAGES.zh : LANGUAGES.en
  const targetLang = direction === 'zh2en' ? LANGUAGES.en : LANGUAGES.zh
  
  // 加载术语库
  useEffect(() => {
    const loadTerms = () => {
      try {
        const saved = localStorage.getItem(TERMS_STORAGE_KEY)
        if (saved) {
          setTerms(JSON.parse(saved))
        }
      } catch (e) {
        console.warn('加载术语库失败:', e)
      }
    }
    loadTerms()
  }, [])
  
  // 构建术语提示词
  const buildTermPrompt = useCallback(() => {
    if (terms.length === 0) return ''
    
    const relevantTerms = terms.filter(t => {
      const sourceInText = sourceText.includes(t.source) || sourceText.includes(t.target)
      return sourceInText
    })
    
    if (relevantTerms.length === 0) return ''
    
    const termList = relevantTerms.map(t => 
      direction === 'zh2en' ? `"${t.source}" → "${t.target}"` : `"${t.target}" → "${t.source}"`
    ).join('\n')
    
    return `\n\n请使用以下术语翻译：\n${termList}`
  }, [terms, sourceText, direction])
  
  // NMT翻译（本地模型）
  const translateNMT = useCallback(async (text: string) => {
    setNmtStreaming(true)
    setNmtStreamText('')
    setError(null)
    
    const startTime = performance.now()
    
    try {
      const response = await fetch(`${API_BASE}/translate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          direction,
          use_cache: true
        })
      })
      
      if (!response.ok) {
        throw new Error('NMT翻译失败')
      }
      
      const data = await response.json()
      const timeCost = (performance.now() - startTime) / 1000
      
      // 模拟流式输出效果
      const chars = data.translation.split('')
      let currentText = ''
      
      for (let i = 0; i < chars.length; i++) {
        currentText += chars[i]
        setNmtStreamText(currentText)
        await new Promise(r => setTimeout(r, 20))
      }
      
      setNmtResult({
        text: data.translation,
        source: 'nmt',
        timeCost
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'NMT翻译失败')
    } finally {
      setNmtStreaming(false)
    }
  }, [direction])
  
  // DeepSeek翻译（流式）
  const translateDeepSeek = useCallback(async (text: string) => {
    setDeepseekStreaming(true)
    setDeepseekStreamText('')
    
    const startTime = performance.now()
    
    try {
      // 使用WebSocket进行流式翻译
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${wsProtocol}//${window.location.host}${API_BASE}/stream-deepseek`
    
    // 使用 ref 避免 stale closure 问题
    const streamTextRef = { current: '' }
    
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws
    
    ws.onopen = () => {
      const termPrompt = buildTermPrompt()
      ws.send(JSON.stringify({
        text,
        direction,
        termPrompt
      }))
    }
    
    ws.onmessage = (event) => {
      try {
        const data: StreamMessage = JSON.parse(event.data)
        
        if (data.type === 'token' && data.content) {
          streamTextRef.current += data.content
          setDeepseekStreamText(streamTextRef.current)
        } else if (data.type === 'done') {
          const timeCost = (performance.now() - startTime) / 1000
          setDeepseekResult({
            text: data.content || streamTextRef.current,
            source: 'deepseek',
            timeCost
          })
          setDeepseekStreaming(false)
          ws.close()
        } else if (data.type === 'error') {
          setError(`DeepSeek 错误: ${data.message || '未知错误'}`)
          setDeepseekStreaming(false)
          ws.close()
        }
      } catch (e) {
        console.error('解析 WebSocket 消息失败:', e)
        setError('解析服务器响应失败')
        setDeepseekStreaming(false)
        ws.close()
      }
    }
    
    ws.onerror = (err) => {
      console.error('WebSocket Error:', err)
      setError('DeepSeek 连接建立失败，正在尝试 Fallback 模式...')
      setDeepseekStreaming(false)
    }

      
      // 如果WebSocket不可用，fallback到HTTP
      ws.onclose = (event) => {
        if (event.code === 1006) {
          // 连接失败，使用HTTP fallback
          fetch(`${API_BASE}/translate-deepseek`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              text,
              direction,
              termPrompt: buildTermPrompt()
            })
          })
          .then(res => res.json())
          .then(data => {
            const timeCost = (performance.now() - startTime) / 1000
            
            // 模拟流式输出
            const chars = data.translation.split('')
            let currentText = ''
            
            const streamChars = async () => {
              for (let i = 0; i < chars.length; i++) {
                currentText += chars[i]
                setDeepseekStreamText(currentText)
                await new Promise(r => setTimeout(r, 15))
              }
              
              setDeepseekResult({
                text: data.translation,
                source: 'deepseek',
                timeCost
              })
              setDeepseekStreaming(false)
            }
            streamChars()
          })
          .catch(() => {
            setError('DeepSeek翻译失败')
            setDeepseekStreaming(false)
          })
        }
      }
      
    } catch (e) {
      setError(e instanceof Error ? e.message : 'DeepSeek翻译失败')
      setDeepseekStreaming(false)
    }
  }, [direction, buildTermPrompt])
  
  // 执行翻译
  const handleTranslate = useCallback(async () => {
    if (!sourceText.trim()) return
    
    setError(null)
    setNmtResult(null)
    setDeepseekResult(null)
    setNmtStreamText('')
    setDeepseekStreamText('')
    
    const text = sourceText.trim()
    
    if (dualMode) {
      // 双协同模式：同时翻译
      translateNMT(text)
      translateDeepSeek(text)
    } else if (useDeepSeek) {
      // 仅DeepSeek
      translateDeepSeek(text)
    } else {
      // 仅NMT
      translateNMT(text)
    }
  }, [sourceText, dualMode, useDeepSeek, translateNMT, translateDeepSeek])
  
  // 清除
  const handleClear = useCallback(() => {
    setSourceText('')
    setNmtResult(null)
    setDeepseekResult(null)
    setNmtStreamText('')
    setDeepseekStreamText('')
    setError(null)
    wsRef.current?.close()
  }, [])
  
  // 复制
  const handleCopy = useCallback(async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
    } catch (e) {
      console.error('复制失败:', e)
    }
  }, [])
  
  // 交换语言
  const handleSwap = useCallback(() => {
    setDirection(prev => prev === 'zh2en' ? 'en2zh' : 'zh2en')
    const result = nmtResult?.text || deepseekResult?.text
    if (result) {
      setSourceText(result)
      setNmtResult(null)
      setDeepseekResult(null)
    }
  }, [nmtResult, deepseekResult])
  
  // 键盘快捷键
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault()
        handleTranslate()
      }
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [handleTranslate])
  
  // 清理WebSocket
  useEffect(() => {
    return () => {
      wsRef.current?.close()
    }
  }, [])
  
  return (
    <div className="max-w-7xl mx-auto p-6 space-y-8 animate-fade-in">
      {/* 顶部控制面板 */}
      <div className="flex flex-col md:flex-row items-center justify-between gap-6">
        <div className="flex items-center gap-4">
          <button
            onClick={() => setShowTermManager(true)}
            className="group flex items-center gap-3 px-5 py-2.5 rounded-xl bg-white border border-slate-200 hover:border-primary/50 shadow-sm transition-all duration-300 text-slate-700 hover:text-primary"
          >
            <BookIcon />
            <span className="font-medium text-sm">术语管理</span>
            {terms.length > 0 && (
              <span className="flex items-center justify-center min-w-[20px] h-5 px-1.5 rounded-md bg-primary/10 text-primary text-[10px] font-bold">
                {terms.length}
              </span>
            )}
          </button>
        </div>
        
        <div className="flex items-center p-1.5 rounded-2xl bg-white border border-slate-200 shadow-sm">
          <button
            onClick={() => { setDualMode(false); setUseDeepSeek(false); }}
            className={`px-6 py-2 rounded-xl text-sm font-semibold transition-all duration-300 ${!dualMode && !useDeepSeek ? 'bg-blue-600 text-white shadow-md shadow-blue-500/30' : 'text-slate-500 hover:text-slate-900'}`}
          >
            标准模式
          </button>
          <button
            onClick={() => { setDualMode(false); setUseDeepSeek(true); }}
            className={`px-6 py-2 rounded-xl text-sm font-semibold transition-all duration-300 ${!dualMode && useDeepSeek ? 'bg-emerald-500 text-white shadow-md shadow-emerald-500/30' : 'text-slate-500 hover:text-slate-900'}`}
          >
            联网模式
          </button>
          <button
            onClick={() => { setDualMode(true); setUseDeepSeek(false); }}
            className={`px-6 py-2 rounded-xl text-sm font-semibold transition-all duration-300 ${dualMode ? 'bg-purple-600 text-white shadow-md shadow-purple-500/30' : 'text-slate-500 hover:text-slate-900'}`}
          >
            双协同对比
          </button>
        </div>
      </div>
      
      {/* 核心翻译工作区 */}
      <div className="premium-card relative bg-white border border-slate-200 shadow-xl">
        {/* 背景光晕装饰 - 改为浅色 */}
        <div className="absolute top-0 left-1/4 w-1/2 h-px bg-gradient-to-r from-transparent via-primary/20 to-transparent" />
        
        {/* 语言导航栏 */}
        <div className="flex items-center justify-between px-8 py-5 border-b border-slate-100 bg-slate-50/50">
          <div className="flex items-center gap-4 min-w-[140px]">
            <span className="text-3xl">{sourceLang.flag}</span>
            <span className="text-lg font-bold tracking-tight text-slate-800">{sourceLang.name}</span>
          </div>
          
          <button 
            onClick={handleSwap} 
            className="group relative flex items-center justify-center w-12 h-12 rounded-full bg-white border border-slate-200 hover:border-primary hover:bg-primary/5 transition-all duration-500 active:scale-95 shadow-sm"
          >
            <div className="group-hover:rotate-180 transition-transform duration-500 text-slate-600 group-hover:text-primary">
              <SwapIcon />
            </div>
          </button>
          
          <div className="flex items-center gap-4 min-w-[140px] justify-end">
            <span className="text-lg font-bold tracking-tight text-slate-800">{targetLang.name}</span>
            <span className="text-3xl">{targetLang.flag}</span>
          </div>
        </div>
        
        {/* 内容网格 */}
        <div className={`grid divide-y lg:divide-y-0 lg:divide-x divide-slate-100 ${dualMode ? 'lg:grid-cols-3' : 'lg:grid-cols-2'}`}>
          {/* 输入区 */}
          <div className="relative group p-2 bg-white">
            <textarea
              ref={textareaRef}
              value={sourceText}
              onChange={e => setSourceText(e.target.value)}
              placeholder={direction === 'zh2en' ? '请输入待翻译文本...' : 'Enter text to translate...'}
              className="textarea-premium text-slate-800 placeholder:text-slate-400"
              disabled={nmtStreaming || deepseekStreaming}
            />
            
            <div className="px-6 py-4 flex items-center justify-between bg-slate-50/50 border-t border-slate-100">
              <span className="text-[10px] font-medium text-slate-400 uppercase tracking-widest">
                {sourceText.length} 字符
              </span>
              {sourceText && (
                <button onClick={handleClear} className="text-slate-400 hover:text-error transition-colors">
                  <ClearIcon />
                </button>
              )}
            </div>
          </div>
          
          {/* NMT 翻译区 */}
          {(!useDeepSeek || dualMode) && (
            <div className="relative flex flex-col bg-white">
              <div className="flex-1 p-6 overflow-auto min-h-[256px]">
                {nmtStreaming ? (
                  <div className="text-lg leading-relaxed text-slate-700">
                    {nmtStreamText}<span className="streaming-cursor bg-blue-600" />
                  </div>
                ) : nmtResult ? (
                  <div className="text-lg leading-relaxed text-slate-900 animate-fade-in">{nmtResult.text}</div>
                ) : (
                  <div className="h-full flex items-center justify-center text-slate-400 italic bg-slate-50/30 rounded-lg">
                    {dualMode ? '本地端推理结果...' : '等待输入...'}
                  </div>
                )}
              </div>
              
              <div className="px-6 py-4 flex items-center justify-between border-t border-slate-100 bg-slate-50">
                <div className="flex items-center gap-3">
                  <span className="glass-tag text-blue-600 border-blue-200 bg-blue-50">本地 NMT</span>
                  {nmtResult && (
                    <span className="flex items-center gap-1.5 px-2 py-1 rounded bg-blue-100 text-blue-600 text-xs font-bold shadow-sm">
                      <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      {nmtResult.timeCost.toFixed(3)}s
                    </span>
                  )}
                </div>
                {nmtResult && <button onClick={() => handleCopy(nmtResult.text)} className="p-2 text-slate-400 hover:text-blue-600"><CopyIcon /></button>}
              </div>
            </div>
          )}
          
          {/* DeepSeek 翻译区 */}
          {(useDeepSeek || dualMode) && (
            <div className="relative flex flex-col bg-emerald-50/50">
              <div className="flex-1 p-6 overflow-auto min-h-[256px]">
                {deepseekStreaming ? (
                  <div className="text-lg leading-relaxed text-slate-700">
                    {deepseekStreamText}<span className="streaming-cursor bg-emerald-500" />
                  </div>
                ) : deepseekResult ? (
                  <div className="text-lg leading-relaxed text-slate-900 animate-fade-in">{deepseekResult.text}</div>
                ) : (
                  <div className="h-full flex items-center justify-center text-slate-400 italic bg-white/50 rounded-lg">
                    {dualMode ? '联网 AI 推理结果...' : '等待输入...'}
                  </div>
                )}
              </div>
              
              <div className="px-6 py-4 flex items-center justify-between border-t border-emerald-100 bg-emerald-50/50">
                <div className="flex items-center gap-3">
                  <span className="glass-tag text-emerald-600 border-emerald-200 bg-emerald-50">DeepSeek AI</span>
                  {deepseekResult && (
                    <span className="flex items-center gap-1.5 px-2 py-1 rounded bg-emerald-100 text-emerald-600 text-xs font-bold shadow-sm">
                      <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      {deepseekResult.timeCost.toFixed(2)}s
                    </span>
                  )}
                </div>
                {deepseekResult && <button onClick={() => handleCopy(deepseekResult.text)} className="p-2 text-slate-400 hover:text-emerald-600"><CopyIcon /></button>}
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* 翻译触发按钮 */}
      <div className="flex flex-col items-center gap-4">
        <button
          onClick={handleTranslate}
          disabled={!sourceText.trim() || nmtStreaming || deepseekStreaming}
          className="btn-premium group min-w-[280px] shadow-xl shadow-primary/20 hover:shadow-primary/40"
        >
          <div className="flex items-center justify-center gap-3">
            {nmtStreaming || deepseekStreaming ? (
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <SparklesIcon />
            )}
            <span className="text-lg tracking-wide font-bold">
              {nmtStreaming || deepseekStreaming ? '正在精准推理...' : '立即翻译'}
            </span>
          </div>
        </button>
        
        <div className="flex items-center gap-2 text-[11px] text-slate-400 font-medium uppercase tracking-[0.1em]">
          <span>按</span>
          <kbd className="px-2 py-1 rounded bg-white border border-slate-200 text-slate-500 shadow-sm font-sans">Ctrl</kbd>
          <span>+</span>
          <kbd className="px-2 py-1 rounded bg-white border border-slate-200 text-slate-500 shadow-sm font-sans">Enter</kbd>
          <span>立即翻译</span>
        </div>
      </div>

      {/* 错误浮窗 */}
      {error && (
        <div className="fixed bottom-8 right-8 max-w-sm p-4 rounded-2xl bg-white border border-error/20 animate-slide-in shadow-2xl z-50">
          <div className="flex items-start gap-3">
            <div className="mt-0.5 text-error"><ClearIcon /></div>
            <div>
              <h4 className="font-bold text-error text-sm">系统错误</h4>
              <p className="text-xs text-slate-600 mt-1 leading-relaxed">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* 术语库管理弹窗 */}
      <TermManager
        isOpen={showTermManager}
        onClose={() => setShowTermManager(false)}
        onTermsChange={setTerms}
      />
    </div>
  )

}

export default Translator
