import { useState } from 'react'
import './WordCard.css'

export default function WordCard({ word, isSelected, onClick }) {
  const conf = word.confidence ?? 0
  const pct = Math.round(conf * 100)
  const confColor = pct >= 80 ? '#00b894' : pct >= 50 ? '#fdcb6e' : '#e17055'

  return (
    <div
      className={`word-card ${isSelected ? 'word-card--selected' : ''}`}
      onClick={onClick}
    >
      {/* Top row */}
      <div className="wc-top">
        <span className="wc-word">{word.text || '—'}</span>
        <span className="wc-region">{word.region_type}</span>
      </div>

      {/* Confidence bar */}
      <div className="wc-conf">
        <div className="wc-conf-bar">
          <div
            className="wc-conf-fill"
            style={{ width: `${pct}%`, background: confColor }}
          />
        </div>
        <span className="wc-conf-pct" style={{ color: confColor }}>{pct}%</span>
      </div>

      {/* Expanded content */}
      {isSelected && (
        <div className="wc-expanded scale-in">
          {/* Heatmap */}
          {word.heatmap_base64 && (
            <div className="wc-heatmap-section">
              <h4>Attention Heatmap</h4>
              <img
                src={`data:image/png;base64,${word.heatmap_base64}`}
                alt={`Heatmap for "${word.text}"`}
                className="wc-heatmap"
                style={{ mixBlendMode: 'normal', opacity: 0.95 }}
              />
            </div>
          )}

          {/* Explanation */}
          {word.explanation && !word.explanation.error && (
            <div className="wc-explanation">
              <h4>Explanation</h4>
              <ul className="explanation-bullets">
                <li>
                  <span className="bullet-label">👁 Visual</span>
                  {word.explanation.visual_reason}
                </li>
                <li>
                  <span className="bullet-label">🧠 Context</span>
                  {word.explanation.context_reason}
                </li>
                {word.explanation.rejected?.map((r, i) => (
                  <li key={i}>
                    <span className="bullet-label">✗ Rejected</span>
                    <strong>"{r.word}"</strong> — {r.reason}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* All candidates */}
          {word.alternatives?.length > 0 && (
            <div className="wc-candidates">
              <h4>All Candidates</h4>
              <table className="cand-table">
                <thead>
                  <tr>
                    <th>Rank</th><th>Word</th><th>Visual</th>
                    <th>Context</th><th>Final</th>
                  </tr>
                </thead>
                <tbody>
                  {word.alternatives.map((c) => (
                    <tr key={c.rank} className={c.rank === 1 ? 'cand-best' : ''}>
                      <td>#{c.rank}</td>
                      <td className="mono">{c.word}</td>
                      <td>{(c.visual_score * 100).toFixed(1)}%</td>
                      <td>{(c.context_score * 100).toFixed(1)}%</td>
                      <td><strong>{(c.final_score * 100).toFixed(1)}%</strong></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
