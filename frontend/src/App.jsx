import React, { useState } from 'react';
import { Loader2, Search } from 'lucide-react';
import ResultCard from './components/ResultCard';
import { predictArticle } from './api';

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleAnalyze = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const data = await predictArticle(text);
      setResult(data);
    } catch (err) {
      setError('Failed to analyze the article. Ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Fake News Intelligence</h1>
        <p>Powered by Hybrid Graph-Augmented Transformer(HGAT)</p>
      </header>

      <main className="main-content">
        <div className="input-section">
          <div className="textarea-container">
            <textarea
              placeholder="Paste the news article or text you want to verify here..."
              value={text}
              onChange={(e) => setText(e.target.value)}
              disabled={loading}
            />
          </div>

          {error && <div style={{ color: 'var(--fake-color)', marginTop: '1rem' }}>{error}</div>}

          <div className="button-group">
            <button
              className="analyze-btn"
              onClick={handleAnalyze}
              disabled={loading || !text.trim()}
            >
              {loading ? (
                <>
                  <Loader2 className="loading-spinner" size={20} />
                  Analyzing...
                </>
              ) : (
                <>
                  <Search size={20} />
                  Verify Article
                </>
              )}
            </button>
          </div>
        </div>

        <ResultCard result={result} />
      </main>
    </div>
  );
}

export default App;
