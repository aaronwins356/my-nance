#!/usr/bin/env node
const express = require('express');
const cors = require('cors');
const axios = require('axios');
require('dotenv').config();
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;
const LM_STUDIO_URL = process.env.LM_STUDIO_URL || 'http://127.0.0.1:1234';

app.use(cors());
app.use(express.json());

async function main() {
  if (process.env.NODE_ENV !== 'production') {
    // Development: use Vite dev server middleware
    const { createServer: createViteServer } = require('vite');
    const vite = await createViteServer({
      root: path.resolve(__dirname),
      server: { middlewareMode: 'html' }
    });
    app.use(vite.middlewares);
  } else {
    // Production: serve static build files
    const staticPath = path.join(__dirname, 'frontend', 'dist');
    app.use(express.static(staticPath));
    // SPA fallback
    app.get('*', (req, res) => {
      res.sendFile(path.join(staticPath, 'index.html'));
    });
  }

  // Health check endpoint
  app.get('/ai/health', async (req, res) => {
    try {
      const response = await axios.get(`${LM_STUDIO_URL}/health`);
      res.json(response.data);
    } catch (error) {
      console.error('Health check failed:', error.message);
      res.status(500).json({ error: 'Health check failed' });
    }
  });

  // Proxy helper
  async function proxyPost(endpoint, body, res) {
    try {
      const response = await axios.post(`${LM_STUDIO_URL}${endpoint}`, body);
      res.json(response.data);
    } catch (error) {
      console.error(`Proxy to ${endpoint} failed:`, error.message);
      const status = error.response?.status || 500;
      res.status(status).json({ error: error.message });
    }
  }

  // AI endpoints
  app.post('/ai/signal', (req, res) => proxyPost('/signal', req.body, res));
  app.post('/ai/portfolio', (req, res) => proxyPost('/portfolio', req.body, res));

  app.listen(PORT, () => {
    console.log(`Server listening on port ${PORT}`);
  });
}

main();
