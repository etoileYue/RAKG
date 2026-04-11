import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const proxyTarget = process.env.VITE_PROXY_TARGET ?? 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: proxyTarget,
        changeOrigin: true,
      },
      '/docs': {
        target: proxyTarget,
        changeOrigin: true,
      },
      '/openapi.json': {
        target: proxyTarget,
        changeOrigin: true,
      },
      '/redoc': {
        target: proxyTarget,
        changeOrigin: true,
      },
    },
  },
});
