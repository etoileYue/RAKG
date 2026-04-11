/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['Sora', 'Noto Sans SC', 'sans-serif'],
        body: ['Noto Sans SC', 'PingFang SC', 'sans-serif'],
      },
      colors: {
        app: {
          bg: '#f5f6f8',
          panel: '#ffffff',
          border: '#e6e8ec',
          text: '#111827',
          muted: '#6b7280',
          primary: '#10b981',
          primarySoft: '#d1fae5',
          info: '#0ea5e9',
          warning: '#f59e0b',
          danger: '#ef4444',
        },
      },
      boxShadow: {
        panel: '0 1px 2px rgba(17, 24, 39, 0.06), 0 8px 18px rgba(17, 24, 39, 0.03)',
      },
      keyframes: {
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(6px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        fadeInUp: 'fadeInUp 240ms ease-out',
      },
    },
  },
  plugins: [],
};
