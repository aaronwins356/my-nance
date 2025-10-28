const express = require('express');
const cors = require('cors');
const axios = require('axios');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 3000;
const LM_STUDIO_URL = process.env.LM_STUDIO_URL || 'http://127.0.0.1:1234';

// Health check
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
async function proxyPost(path, body, res) {
  try {
    const url = `${LM_STUDIO_URL}${path}`;
    const response = await axios.post(url, body);
    res.json(response.data);
  } catch (error) {
    console.error(`Proxy to ${path} failed:`, error.message);
    const status = error.response?.status || 500;
    res.status(status).json({ error: error.message });
  }
}

// Signal endpoint
app.post('/ai/signal', async (req, res) => {
  await proxyPost('/signal', req.body, res);
});

// Portfolio endpoint
app.post('/ai/portfolio', async (req, res) => {
  await proxyPost('/portfolio', req.body, res);
});

app.listen(PORT, () => {
  console.log(`AI proxy server listening on port ${PORT}`);
});
