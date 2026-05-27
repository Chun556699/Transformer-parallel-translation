/**
 * 术语库管理组件
 * 
 * 功能说明：
 *   - 术语库的增删改查
 *   - 支持中英文术语对照
 *   - 本地存储持久化
 *   - 导入导出功能
 */

import React, { useState, useCallback, useEffect } from 'react'

// ============================================================================
// 类型定义
// ============================================================================

/** 术语条目 */
export interface Term {
  id: string
  source: string      // 源语言术语
  target: string      // 目标语言术语
  category: string    // 分类
  note?: string       // 备注
  createdAt: number   // 创建时间
}

/** 术语库Props */
interface TermManagerProps {
  isOpen: boolean
  onClose: () => void
  onTermsChange?: (terms: Term[]) => void
}

// ============================================================================
// 常量
// ============================================================================

const STORAGE_KEY = 'nmt_term_database'
const CATEGORIES = ['通用', '计算机', '医学', '法律', '金融', '工程', '其他']

// ============================================================================
// 辅助函数
// ============================================================================

/** 生成唯一ID */
const generateId = () => `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`

/** 从localStorage加载术语库 */
const loadTerms = (): Term[] => {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) {
      return JSON.parse(saved)
    }
  } catch (e) {
    console.warn('加载术语库失败:', e)
  }
  return []
}

/** 保存术语库到localStorage */
const saveTerms = (terms: Term[]) => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(terms))
  } catch (e) {
    console.warn('保存术语库失败:', e)
  }
}

// ============================================================================
// 图标组件
// ============================================================================

const CloseIcon = () => (
  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M18 6L6 18M6 6l12 12" />
  </svg>
)

const AddIcon = () => (
  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M12 5v14M5 12h14" />
  </svg>
)

const EditIcon = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7" />
    <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z" />
  </svg>
)

const DeleteIcon = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2" />
  </svg>
)

const ExportIcon = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" />
  </svg>
)

const ImportIcon = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" />
  </svg>
)

// ============================================================================
// 主组件
// ============================================================================

export const TermManager: React.FC<TermManagerProps> = ({ isOpen, onClose, onTermsChange }) => {
  // 状态
  const [terms, setTerms] = useState<Term[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [isEditing, setIsEditing] = useState(false)
  const [editingTerm, setEditingTerm] = useState<Term | null>(null)
  
  // 表单状态
  const [formSource, setFormSource] = useState('')
  const [formTarget, setFormTarget] = useState('')
  const [formCategory, setFormCategory] = useState('通用')
  const [formNote, setFormNote] = useState('')
  
  // 加载术语库
  useEffect(() => {
    if (isOpen) {
      const loaded = loadTerms()
      setTerms(loaded)
    }
  }, [isOpen])
  
  // 保存术语库
  const handleSaveTerms = useCallback((newTerms: Term[]) => {
    setTerms(newTerms)
    saveTerms(newTerms)
    onTermsChange?.(newTerms)
  }, [onTermsChange])
  
  // 添加/更新术语
  const handleSubmit = useCallback(() => {
    if (!formSource.trim() || !formTarget.trim()) return
    
    let newTerms: Term[]
    
    if (editingTerm) {
      // 更新
      newTerms = terms.map(t => 
        t.id === editingTerm.id 
          ? { ...t, source: formSource.trim(), target: formTarget.trim(), category: formCategory, note: formNote.trim() }
          : t
      )
    } else {
      // 添加
      const newTerm: Term = {
        id: generateId(),
        source: formSource.trim(),
        target: formTarget.trim(),
        category: formCategory,
        note: formNote.trim(),
        createdAt: Date.now()
      }
      newTerms = [newTerm, ...terms]
    }
    
    handleSaveTerms(newTerms)
    resetForm()
  }, [terms, editingTerm, formSource, formTarget, formCategory, formNote, handleSaveTerms])
  
  // 删除术语
  const handleDelete = useCallback((id: string) => {
    const newTerms = terms.filter(t => t.id !== id)
    handleSaveTerms(newTerms)
  }, [terms, handleSaveTerms])
  
  // 编辑术语
  const handleEdit = useCallback((term: Term) => {
    setEditingTerm(term)
    setFormSource(term.source)
    setFormTarget(term.target)
    setFormCategory(term.category)
    setFormNote(term.note || '')
    setIsEditing(true)
  }, [])
  
  // 重置表单
  const resetForm = useCallback(() => {
    setIsEditing(false)
    setEditingTerm(null)
    setFormSource('')
    setFormTarget('')
    setFormCategory('通用')
    setFormNote('')
  }, [])
  
  // 导出术语库
  const handleExport = useCallback(() => {
    const data = JSON.stringify(terms, null, 2)
    const blob = new Blob([data], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `术语库_${new Date().toISOString().slice(0, 10)}.json`
    a.click()
    URL.revokeObjectURL(url)
  }, [terms])
  
  // 导入术语库
  const handleImport = useCallback(() => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.json'
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return
      
      try {
        const text = await file.text()
        const imported = JSON.parse(text) as Term[]
        if (Array.isArray(imported)) {
          const newTerms = [...imported, ...terms]
          handleSaveTerms(newTerms)
        }
      } catch (err) {
        alert('导入失败：文件格式错误')
      }
    }
    input.click()
  }, [terms, handleSaveTerms])
  
  // 过滤术语
  const filteredTerms = terms.filter(term => {
    const matchSearch = term.source.includes(searchQuery) || 
                        term.target.includes(searchQuery) ||
                        (term.note?.includes(searchQuery) ?? false)
    const matchCategory = selectedCategory === 'all' || term.category === selectedCategory
    return matchSearch && matchCategory
  })
  
  if (!isOpen) return null
  
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div 
        className="card w-full max-w-4xl max-h-[85vh] m-4 flex flex-col animate-fade-in"
        onClick={e => e.stopPropagation()}
      >
        {/* 标题栏 */}
        <div className="flex items-center justify-between p-4 border-b border-[var(--border-color)]">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg flex items-center justify-center"
                 style={{ background: 'linear-gradient(135deg, var(--accent-cyan), var(--accent-blue))' }}>
              <svg className="w-5 h-5 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
            </div>
            <div>
              <h2 className="text-lg font-semibold">术语库管理</h2>
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                共 {terms.length} 条术语
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <button onClick={handleImport} className="btn-secondary text-sm flex items-center gap-1">
              <ImportIcon /> 导入
            </button>
            <button onClick={handleExport} className="btn-secondary text-sm flex items-center gap-1">
              <ExportIcon /> 导出
            </button>
            <button onClick={onClose} className="btn-icon">
              <CloseIcon />
            </button>
          </div>
        </div>
        
        {/* 主内容区 */}
        <div className="flex-1 flex overflow-hidden">
          {/* 左侧：术语列表 */}
          <div className="flex-1 flex flex-col overflow-hidden">
            {/* 搜索和过滤 */}
            <div className="p-4 border-b border-[var(--border-color)] flex gap-3">
              <input
                type="text"
                placeholder="搜索术语..."
                className="input-field flex-1"
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
              />
              <select
                className="input-field w-32"
                value={selectedCategory}
                onChange={e => setSelectedCategory(e.target.value)}
              >
                <option value="all">全部分类</option>
                {CATEGORIES.map(cat => (
                  <option key={cat} value={cat}>{cat}</option>
                ))}
              </select>
              <button
                onClick={() => { resetForm(); setIsEditing(true); }}
                className="btn-primary flex items-center gap-1"
              >
                <AddIcon /> 添加
              </button>
            </div>
            
            {/* 术语列表 */}
            <div className="flex-1 overflow-y-auto p-4 space-y-2">
              {filteredTerms.length === 0 ? (
                <div className="text-center py-12" style={{ color: 'var(--text-muted)' }}>
                  {searchQuery ? '未找到匹配的术语' : '暂无术语，点击"添加"创建'}
                </div>
              ) : (
                filteredTerms.map(term => (
                  <div 
                    key={term.id}
                    className="p-4 rounded-lg transition-all duration-200 hover:border-[var(--primary)]"
                    style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="tag tag-primary text-xs">{term.category}</span>
                        </div>
                        <div className="flex items-center gap-4">
                          <div className="flex-1">
                            <div className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>源术语</div>
                            <div className="font-medium">{term.source}</div>
                          </div>
                          <svg className="w-4 h-4 flex-shrink-0" style={{ color: 'var(--text-muted)' }} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M5 12h14M12 5l7 7-7 7" />
                          </svg>
                          <div className="flex-1">
                            <div className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>目标术语</div>
                            <div className="font-medium" style={{ color: 'var(--accent-cyan)' }}>{term.target}</div>
                          </div>
                        </div>
                        {term.note && (
                          <div className="mt-2 text-sm" style={{ color: 'var(--text-muted)' }}>
                            备注：{term.note}
                          </div>
                        )}
                      </div>
                      <div className="flex items-center gap-1 ml-4">
                        <button
                          onClick={() => handleEdit(term)}
                          className="btn-icon"
                          title="编辑"
                        >
                          <EditIcon />
                        </button>
                        <button
                          onClick={() => handleDelete(term.id)}
                          className="btn-icon"
                          style={{ color: 'var(--error)' }}
                          title="删除"
                        >
                          <DeleteIcon />
                        </button>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
          
          {/* 右侧：编辑表单 */}
          {isEditing && (
            <div 
              className="w-80 border-l border-[var(--border-color)] p-4 flex flex-col animate-slide-in"
              style={{ background: 'var(--bg-tertiary)' }}
            >
              <h3 className="text-sm font-medium mb-4">
                {editingTerm ? '编辑术语' : '添加新术语'}
              </h3>
              
              <div className="space-y-4 flex-1">
                <div>
                  <label className="block text-sm mb-1" style={{ color: 'var(--text-muted)' }}>
                    源术语 <span style={{ color: 'var(--error)' }}>*</span>
                  </label>
                  <input
                    type="text"
                    className="input-field"
                    placeholder="输入源语言术语"
                    value={formSource}
                    onChange={e => setFormSource(e.target.value)}
                  />
                </div>
                
                <div>
                  <label className="block text-sm mb-1" style={{ color: 'var(--text-muted)' }}>
                    目标术语 <span style={{ color: 'var(--error)' }}>*</span>
                  </label>
                  <input
                    type="text"
                    className="input-field"
                    placeholder="输入目标语言术语"
                    value={formTarget}
                    onChange={e => setFormTarget(e.target.value)}
                  />
                </div>
                
                <div>
                  <label className="block text-sm mb-1" style={{ color: 'var(--text-muted)' }}>
                    分类
                  </label>
                  <select
                    className="input-field"
                    value={formCategory}
                    onChange={e => setFormCategory(e.target.value)}
                  >
                    {CATEGORIES.map(cat => (
                      <option key={cat} value={cat}>{cat}</option>
                    ))}
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm mb-1" style={{ color: 'var(--text-muted)' }}>
                    备注
                  </label>
                  <textarea
                    className="input-field h-20 resize-none"
                    placeholder="可选备注信息"
                    value={formNote}
                    onChange={e => setFormNote(e.target.value)}
                  />
                </div>
              </div>
              
              <div className="flex gap-2 mt-4">
                <button onClick={resetForm} className="btn-secondary flex-1">
                  取消
                </button>
                <button 
                  onClick={handleSubmit}
                  className="btn-primary flex-1"
                  disabled={!formSource.trim() || !formTarget.trim()}
                >
                  {editingTerm ? '保存' : '添加'}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default TermManager
