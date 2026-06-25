import { useCallback, useState } from 'react'
import './UploadZone.css'

export default function UploadZone({ onFileSelect }) {
  const [dragging, setDragging] = useState(false)

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file && file.type.startsWith('image/')) onFileSelect(file)
  }, [onFileSelect])

  const handleFileInput = useCallback((e) => {
    const file = e.target.files?.[0]
    if (file) onFileSelect(file)
  }, [onFileSelect])

  return (
    <main className="upload-page">
      <div
        className={`upload-zone ${dragging ? 'dragging' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => document.getElementById('file-input').click()}
      >
        <input
          id="file-input"
          type="file"
          accept="image/*"
          style={{ display: 'none' }}
          onChange={handleFileInput}
        />
        <div className="upload-icon">📄</div>
        <h2 className="upload-title">Drop a document image here</h2>
        <p className="upload-sub">
          or <span className="upload-link">browse files</span>
        </p>
        <p className="upload-formats">JPEG · PNG · TIFF · BMP · WEBP</p>
      </div>

      <div className="feature-grid">
        {FEATURES.map((f) => (
          <div key={f.title} className="feature-card">
            <span className="feature-icon">{f.icon}</span>
            <h3>{f.title}</h3>
            <p>{f.desc}</p>
          </div>
        ))}
      </div>
    </main>
  )
}

const FEATURES = [
  { icon: '🔤', title: 'Handwriting OCR', desc: 'TrOCR fine-tuned on IAM handwriting with beam search for top-5 candidates.' },
  { icon: '🌡️', title: 'Visual Heatmaps', desc: 'Attention rollout & GradCAM show which pixels drove each character prediction.' },
  { icon: '🧠', title: 'Context Fusion', desc: 'LLM scores candidates in sentence context. Visual + context scores fused (0.7/0.3).' },
  { icon: '💬', title: 'NL Explanations', desc: '3-bullet natural language explanation for every recognized word via Claude/Qwen.' },
]
