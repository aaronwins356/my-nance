import Alpine from 'alpinejs';
import i18next from 'i18next';
import en from './i18n/en.json';
import Chart from 'chart.js/auto';
import './styles.scss';

Alpine.data('dashboard', () => ({
  ticker: '',
  risk: 'moderate',
  loading: false,
  signal: null,
  confidence: '',
  summary: '',
  allocation: null,
  chart: null,
  badgeClass: '',

  async analyze() {
    this.loading = true;
    this.signal = null;
    this.allocation = null;
    try {
      // Signal request
      const sigRes = await fetch('/ai/signal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker: this.ticker })
      });
      if (!sigRes.ok) throw new Error((await sigRes.json()).error || 'Signal error');
      const sig = await sigRes.json();
      this.signal = sig.signal;
      this.confidence = (sig.confidence * 100).toFixed(2);
      this.summary = sig.summary;
      this.badgeClass = {
        bullish: 'badge-success',
        bearish: 'badge-danger',
        neutral: 'badge-secondary'
      }[this.signal] || 'badge-secondary';

      // Portfolio request
      const portRes = await fetch('/ai/portfolio', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          cash: 0.2,
          btc: 0.1,
          stocks: 0.7,
          risk_level: this.risk,
          signal: this.signal
        })
      });
      if (!portRes.ok) throw new Error((await portRes.json()).error || 'Portfolio error');
      const port = await portRes.json();
      const data = { ...port };
      delete data.advice;
      this.allocation = data;

      // Render chart
      const labels = Object.keys(data);
      const values = Object.values(data).map(v => (v * 100).toFixed(2));
      if (this.chart) this.chart.destroy();
      const ctx = document.getElementById('portfolioChart');
      this.chart = new Chart(ctx, {
        type: 'pie',
        data: {
          labels,
          datasets: [{ data: values, backgroundColor: ['#198754', '#dc3545', '#6c757d'] }]
        }
      });
    } catch (e) {
      console.error(e);
      alert(i18next.t('error'));
    } finally {
      this.loading = false;
    }
  }
}));

// Initialize i18next and Alpine
i18next.init({
  lng: 'en',
  resources: { en: { translation: en } }
}).then(() => Alpine.start());
