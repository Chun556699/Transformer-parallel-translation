/**
 * React 应用入口文件
 * 
 * 功能说明：
 *   - 初始化 React 应用
 *   - 配置全局样式
 *   - 挂载根组件到 DOM
 * 
 * 依赖：
 *   - React 18 (createRoot API)
 *   - Tailwind CSS
 */

import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './styles/index.css'

// ============================================================================
// 应用挂载
// ============================================================================

/**
 * 获取根 DOM 节点并挂载 React 应用
 * 使用 React 18 的 createRoot API 支持并发特性
 */
const rootElement = document.getElementById('root')

if (!rootElement) {
  throw new Error('无法找到根节点 #root，请检查 index.html')
}

// 创建 React 根节点
const root = ReactDOM.createRoot(rootElement)

// 渲染应用（开发模式下启用严格模式）
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
