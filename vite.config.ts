import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  root: '.',
  resolve: {
    alias: { '@': resolve(__dirname, 'src') }
  },
  server: {
    host: true,
    port: 5173,
    cors: true,
    proxy: {
      '/ai': 'http://localhost:3000'
    }
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true
  }
});