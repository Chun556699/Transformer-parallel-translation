/**
 * Tailwind CSS 配置文件
 * 
 * 功能说明：
 *   - 配置 Tailwind 扫描范围
 *   - 自定义主题扩展
 *   - 配置插件
 */

/** @type {import('tailwindcss').Config} */
export default {
  // 内容扫描路径
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  
  // 主题配置
  theme: {
    extend: {
      // 自定义颜色
      colors: {
        primary: {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
        },
      },
      
      // 自定义字体
      fontFamily: {
        sans: [
          'Source Han Sans SC',
          'PingFang SC',
          'Microsoft YaHei',
          'Noto Sans CJK SC',
          'sans-serif',
        ],
        mono: [
          'JetBrains Mono',
          'Fira Code',
          'Consolas',
          'monospace',
        ],
      },
      
      // 自定义间距
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '128': '32rem',
      },
      
      // 自定义动画
      animation: {
        'fade-in': 'fadeIn 0.3s ease-out',
        'pulse-slow': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      
      // 自定义阴影
      boxShadow: {
        'card': '0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)',
        'card-hover': '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
      },
    },
  },
  
  // 插件配置
  plugins: [],
}
