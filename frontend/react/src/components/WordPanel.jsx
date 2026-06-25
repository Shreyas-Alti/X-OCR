import { useState } from 'react'
import WordCard from './WordCard.jsx'
import './WordPanel.css'

export default function WordPanel({ words, selectedWord, onSelectWord, timings }) {
  const [filter, setFilter] = useState('')

  const filtered = words.filter(
    (w) => !filter || w.text.toLowerCase().includes(filter.toLowerCase())
  )

  const totalMs = timings
    ? Object.values(timings).reduce((a, b) => a + b, 0) * 1000
    : null

  return (
    <div className="word-panel">
      {/* Header */}
      <div className="word-panel-header">
        <h2>Word Results</h2>
        {totalMs !== null && (
          <span className="timing-badge">⏱ {totalMs.toFixed(0)}ms total</span>
        )}
      </div>

      {/* Timing breakdown */}
      {timings && (
        <div className="timing-row">
          {Object.entries(timings).map(([mod, secs]) => (
            <div key={mod} className="timing-chip">
              <span className="timing-mod">{mod.replace('_', ' ')}</span>
              <span className="timing-val">{(secs * 1000).toFixed(0)}ms</span>
            </div>
          ))}
        </div>
      )}

      {/* Search */}
      <input
        className="word-search"
        type="search"
        placeholder="🔎 Filter words…"
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
      />

      {/* Word cards */}
      <div className="word-list">
        {filtered.length === 0 && (
          <p className="no-words">No words match your filter.</p>
        )}
        {filtered.map((word, i) => (
          <WordCard
            key={i}
            word={word}
            isSelected={selectedWord === word}
            onClick={() => onSelectWord(selectedWord === word ? null : word)}
          />
        ))}
      </div>
    </div>
  )
}
