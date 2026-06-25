import { useRef, useState, useEffect } from 'react'
import './DocumentPanel.css'

const REGION_COLORS = {
  header:   '#6c5ce7',
  question: '#00b894',
  answer:   '#fdcb6e',
  other:    '#74b9ff',
}

export default function DocumentPanel({ imageUrl, regions, selectedWord, onSelectWord, loading }) {
  const canvasRef = useRef(null)
  const imgRef = useRef(null)
  const [imgDims, setImgDims] = useState(null)

  // Draw bounding boxes on canvas overlay
  useEffect(() => {
    if (!canvasRef.current || !imgDims || regions.length === 0) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const scaleX = canvas.width / imgDims.natural.w
    const scaleY = canvas.height / imgDims.natural.h

    regions.forEach((region) => {
      const [x1, y1, x2, y2] = region.bbox
      const color = REGION_COLORS[region.region_type] || '#a29bfe'
      const sx = x1 * scaleX, sy = y1 * scaleY
      const sw = (x2 - x1) * scaleX, sh = (y2 - y1) * scaleY

      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.strokeRect(sx, sy, sw, sh)

      ctx.fillStyle = color + '22'
      ctx.fillRect(sx, sy, sw, sh)

      // Label
      ctx.fillStyle = color
      ctx.font = '11px Inter, sans-serif'
      ctx.fillRect(sx, sy - 16, Math.min(sw, 90), 16)
      ctx.fillStyle = '#fff'
      ctx.fillText(region.region_type, sx + 4, sy - 3)
    })
  }, [regions, imgDims])

  const handleImgLoad = (e) => {
    setImgDims({
      natural: { w: e.target.naturalWidth, h: e.target.naturalHeight },
      rendered: { w: e.target.offsetWidth, h: e.target.offsetHeight },
    })
  }

  return (
    <div className="doc-panel">
      <div className="doc-panel-header">
        <h2>Document View</h2>
        {regions.length > 0 && (
          <div className="region-legend">
            {Object.entries(REGION_COLORS).map(([type, color]) => (
              <span key={type} className="legend-item">
                <span className="legend-dot" style={{ background: color }} />
                {type}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="doc-image-wrapper">
        <img
          ref={imgRef}
          src={imageUrl}
          alt="Uploaded document"
          className="doc-image"
          onLoad={handleImgLoad}
        />
        {imgDims && regions.length > 0 && (
          <canvas
            ref={canvasRef}
            className="doc-canvas"
            width={imgRef.current?.offsetWidth || 400}
            height={imgRef.current?.offsetHeight || 400}
          />
        )}
        {loading && (
          <div className="doc-loading-overlay">
            <div className="loading-spinner spin" />
            <p>Analysing document…</p>
          </div>
        )}
      </div>

      {regions.length > 0 && (
        <div className="doc-stats">
          <span>{regions.length} region{regions.length !== 1 ? 's' : ''} detected</span>
          <span>·</span>
          <span>{regions.reduce((acc, r) => acc + r.words.length, 0)} words recognised</span>
        </div>
      )}
    </div>
  )
}
