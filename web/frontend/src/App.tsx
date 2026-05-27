/**
 * NMT翻译系统主应用
 * 
 * 功能说明：
 *   - 专业亮色主题界面
 *   - 双协同翻译支持
 *   - 服务状态监控
 */

import React, { useState, useEffect } from 'react'
import Translator from './components/Translator'

// ============================================================================
// 类型定义
// ============================================================================

type ServiceStatus = 'checking' | 'online' | 'offline'

interface ModelInfo {
  loaded: boolean
  quantized: boolean
  dtype: string
  device: string
}

// ============================================================================
// 主组件
// ============================================================================

const App: React.FC = () => {
  const [serviceStatus, setServiceStatus] = useState<ServiceStatus>('checking')
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  
  // 检查服务状态
  useEffect(() => {
    const checkService = async () => {
      try {
        const response = await fetch('/api/health')
        if (response.ok) {
          setServiceStatus('online')
          // 获取模型信息
          const modelsRes = await fetch('/api/models')
          if (modelsRes.ok) {
            const models = await modelsRes.json()
            if (models.value && models.value.length > 0) {
              setModelInfo(models.value[0])
            }
          }
        } else {
          setServiceStatus('offline')
        }
      } catch {
        setServiceStatus('offline')
      }
    }
    
    checkService()
    const interval = setInterval(checkService, 30000)
    return () => clearInterval(interval)
  }, [])
  
  return (
    <div className="min-h-screen flex flex-col selection:bg-primary/30 selection:text-primary">
      {/* 顶部导航栏 - 高级玻璃拟态 */}
      <header className="sticky top-0 z-40 bg-white/70 backdrop-blur-2xl border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          {/* Logo & Title */}
          <div className="flex items-center gap-5">
            <div 
              className="w-12 h-12 rounded-2xl flex items-center justify-center text-white font-black text-xl shadow-xl shadow-primary/20 rotate-3 hover:rotate-0 transition-transform duration-500"
              style={{ background: 'linear-gradient(135deg, var(--primary), var(--accent-purple))' }}
            >
              N
            </div>
            <div>
              <h1 className="text-xl font-black tracking-tight text-slate-800">
                NMT <span className="text-primary">智能翻译系统</span>
              </h1>
              <div className="flex items-center gap-2 mt-0.5">
                <span className="w-1.5 h-1.5 rounded-full bg-success animate-pulse" />
                <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">
                  工业级神经机器翻译系统
                </p>
              </div>
            </div>
          </div>
          
          {/* 右侧实时状态 */}
          <div className="flex items-center gap-6">
            {modelInfo && (
              <div className="hidden lg:flex items-center gap-4 px-4 py-2 rounded-xl bg-slate-100 border border-slate-200">
                <div className="flex flex-col items-end">
                  <span className="text-[10px] font-bold text-slate-500 uppercase tracking-tighter">引擎精度</span>
                  <span className="text-xs font-mono text-primary">{modelInfo.dtype.toUpperCase()}</span>
                </div>
                <div className="w-px h-8 bg-slate-200" />
                <div className="flex flex-col">
                  <span className="text-[10px] font-bold text-slate-500 uppercase tracking-tighter">硬件加速</span>
                  <span className="text-xs font-mono text-accent-cyan">{modelInfo.device.toUpperCase()}</span>
                </div>
              </div>
            )}
            
            <div className={`flex items-center gap-3 px-4 py-2 rounded-xl border transition-all duration-500 ${
              serviceStatus === 'online' ? 'bg-success/5 border-success/20 text-success' : 
              serviceStatus === 'offline' ? 'bg-error/5 border-error/20 text-error' : 'bg-warning/5 border-warning/20 text-warning'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                serviceStatus === 'online' ? 'bg-success shadow-[0_0_10px_var(--success)]' :
                serviceStatus === 'offline' ? 'bg-error shadow-[0_0_10px_var(--error)]' : 'bg-warning animate-pulse'
              }`} />
              <span className="text-xs font-black uppercase tracking-widest">
                {serviceStatus === 'online' ? '系统就绪' :
                 serviceStatus === 'offline' ? '连接断开' : '同步中'}
              </span>
            </div>
          </div>
        </div>
      </header>

      
      {/* 主内容区域 */}
      <main className="flex-1 py-6">
        {serviceStatus === 'offline' ? (
          <div className="max-w-md mx-auto p-8 text-center animate-fade-in bg-white rounded-2xl shadow-lg border border-slate-100 mt-10">
            <div 
              className="w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4"
              style={{ background: 'rgba(239, 68, 68, 0.1)' }}
            >
              <svg className="w-8 h-8" style={{ color: 'var(--error)' }} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
            </div>
            <h2 className="text-xl font-semibold mb-2 text-slate-800">翻译服务不可用</h2>
            <p className="mb-4 text-slate-500">
              无法连接到翻译服务器，请确保后端服务已启动。
            </p>
            <button onClick={() => window.location.reload()} className="btn-primary">
              重新检查
            </button>
          </div>
        ) : (
          <Translator />
        )}
      </main>
      
      {/* 底部信息 - 简洁大气 */}
      <footer className="py-12 border-t border-slate-200 bg-white/50">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex flex-col items-center md:items-start gap-2">
            <p className="text-xs font-bold text-slate-500 uppercase tracking-[0.3em]">
              核心技术
            </p>
            <div className="flex items-center gap-4 text-slate-500 text-sm font-medium">
              <span>MarianMT</span>
              <span className="w-1 h-1 rounded-full bg-slate-300" />
              <span>FP16 量化</span>
              <span className="w-1 h-1 rounded-full bg-slate-300" />
              <span>CUDA 加速</span>
              <span className="w-1 h-1 rounded-full bg-slate-300" />
              <span>DeepSeek AI</span>
            </div>
          </div>
          
          <div className="text-center md:text-right">
            <p className="text-sm font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-accent-purple">
              新一代神经机器翻译系统
            </p>
            <p className="text-[10px] text-slate-400 mt-1 uppercase tracking-widest">
              NMT © 2026 /// 作者淳飞
            </p>
          </div>
        </div>
      </footer>

    </div>
  )
}

export default App
