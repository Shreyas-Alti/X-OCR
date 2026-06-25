import './Header.css'

export default function Header({ onRun, loading, hasFile }) {
  return (
    <header className="header">
      <div className="header-brand">
        <span className="header-logo">🔍</span>
        <div>
          <h1 className="header-title">X-OCR</h1>
          <span className="header-subtitle">Explainable OCR System</span>
        </div>
      </div>

      <nav className="header-nav">
        <a href="http://localhost:8000/docs" target="_blank" rel="noopener" className="nav-link">
          API Docs
        </a>
        <a href="https://github.com" target="_blank" rel="noopener" className="nav-link">
          GitHub
        </a>
      </nav>

      <div className="header-actions">
        <button
          className={`btn-run ${loading ? 'btn-loading' : ''}`}
          onClick={onRun}
          disabled={!hasFile || loading}
        >
          {loading ? (
            <>
              <span className="spin" style={{ display: 'inline-block' }}>⚙️</span>
              &nbsp;Processing…
            </>
          ) : (
            '🚀 Run X-OCR'
          )}
        </button>
      </div>
    </header>
  )
}
