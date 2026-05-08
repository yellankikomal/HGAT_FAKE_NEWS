import React from 'react';
import { AlertCircle, CheckCircle2 } from 'lucide-react';

export default function ResultCard({ result }) {
  if (!result) return null;

  const isFake = result.prediction === "FAKE";
  
  return (
    <div className={`result-card ${isFake ? 'fake' : 'real'}`}>
      <div className="result-header">
        {isFake ? (
          <AlertCircle size={40} color="var(--fake-color)" />
        ) : (
          <CheckCircle2 size={40} color="var(--real-color)" />
        )}
        <h2>{result.prediction}</h2>
      </div>
      
      <div className="metrics-grid">
        <div className="metric-item">
          <span className="metric-label">Confidence</span>
          <span className="metric-value">{result.confidence}%</span>
        </div>
        
        <div className="metric-item">
          <span className="metric-label">Fusion β Score</span>
          <span className="metric-value">{result.beta.toFixed(3)}</span>
        </div>
      </div>
      
      <div className="beta-explanation">
        <strong>What is Fusion β?</strong> This score (0 to 1) shows how the Heterogeneous Graph Attention Network (HGAT) weighted the information. Closer to 1 means it relied more on the text content, while closer to 0 means it relied more on the extracted entity graph.
      </div>
    </div>
  );
}
