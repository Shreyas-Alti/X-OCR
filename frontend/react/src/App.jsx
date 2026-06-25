import { useState, useCallback, useRef } from 'react'
import './App.css'
import DocumentPanel from './components/DocumentPanel.jsx'
import WordPanel from './components/WordPanel.jsx'
import Header from './components/Header.jsx'
import UploadZone from './components/UploadZone.jsx'

const API_URL = import.meta.env.VITE_API_URL || ''

export default function App() {
  const [imageFile, setImageFile] = useState(null)
  const [imageUrl, setImageUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [selectedWord, setSelectedWord] = useState(null)

  const handleFileSelect = useCallback((file) => {
    setImageFile(file)
    setImageUrl(URL.createObjectURL(file))
    setResult(null)
    setError(null)
    setSelectedWord(null)
  }, [])

  const handleRun = useCallback(async () => {
    if (!imageFile) return
    setLoading(true)
    setError(null)
    setResult(null)
    setSelectedWord(null)

    try {
      const form = new FormData()
      form.append('file', imageFile)
      const resp = await fetch(`${API_URL}/api/ocr`, {
        method: 'POST',
        body: form,
      })
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}))
        throw new Error(errData.detail || `HTTP ${resp.status}`)
      }
      const data = await resp.json()
      setResult(data)
    } catch (e) {
      setError(e.message || 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [imageFile])

  // Collect all words across all regions for the right panel
  const allWords = result
    ? result.regions.flatMap((r) =>
        r.words.map((w) => ({ ...w, region_type: r.region_type, bbox: r.bbox }))
      )
    : []

  return (
    <div className="app">
      <Header onRun={handleRun} loading={loading} hasFile={!!imageFile} />

      {!imageFile ? (
        <UploadZone onFileSelect={handleFileSelect} />
      ) : (
        <main className="main-layout">
          {/* Left panel — document view */}
          <section className="panel panel-left">
            <DocumentPanel
              imageUrl={imageUrl}
              regions={result?.regions || []}
              selectedWord={selectedWord}
              onSelectWord={setSelectedWord}
              loading={loading}
            />
          </section>

          {/* Right panel — word cards */}
          <section className="panel panel-right">
            {error && (
              <div className="error-banner fade-in">
                <span>⚠️</span>
                <span>{error}</span>
              </div>
            )}
            {loading && <LoadingSkeleton />}
            {!loading && result && (
              <WordPanel
                words={allWords}
                selectedWord={selectedWord}
                onSelectWord={setSelectedWord}
                timings={result.timings}
              />
            )}
            {!loading && !result && !error && (
              <div className="empty-state">
                <div className="empty-icon">🔍</div>
                <p>Click <strong>Run X-OCR</strong> to analyse the document.</p>
              </div>
            )}
          </section>
        </main>
      )}
    </div>
  )
}

function LoadingSkeleton() {
  return (
    <div className="skeleton-list fade-in">
      {[...Array(5)].map((_, i) => (
        <div key={i} className="skeleton-card">
          <div className="skeleton" style={{ height: 20, width: '40%', marginBottom: 8 }} />
          <div className="skeleton" style={{ height: 10, width: '100%', marginBottom: 6 }} />
          <div className="skeleton" style={{ height: 10, width: '80%' }} />
        </div>
      ))}
    </div>
  )
}
