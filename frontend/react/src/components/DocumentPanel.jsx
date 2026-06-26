import { useRef, useState, useEffect, useCallback } from 'react'
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
  const wrapperRef = useRef(null)
  const [imgDims, setImgDims] = useState(null)

  // Draw bounding boxes — called whenever dims or regions change
  const drawBoxes = useCallback((canvas, dims) => {
    if (!canvas || !dims || regions.length === 0) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const scaleX = canvas.width / dims.natural.w
    const scaleY = canvas.height / dims.natural.h

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

      ctx.fillStyle = color
      ctx.font = '11px Inter, sans-serif'
      ctx.fillRect(sx, sy - 16, Math.min(sw, 90), 16)
      ctx.fillStyle = '#fff'
      ctx.fillText(region.region_type, sx + 4, sy - 3)
    })
  }, [regions])

  // Re-draw when dims or regions change
  useEffect(() => {
    drawBoxes(canvasRef.current, imgDims)
  }, [regions, imgDims, drawBoxes])

  // Update canvas dimensions from current image rendered size
  const syncCanvasDims = useCallback(() => {
    if (!imgRef.current) return
    const img = imgRef.current
    const newDims = {
      natural:  { w: img.naturalWidth,  h: img.naturalHeight },
      rendered: { w: img.offsetWidth,   h: img.offsetHeight  },
    }
    // Update canvas size to match current rendered size
    if (canvasRef.current) {
      canvasRef.current.width  = img.offsetWidth  || 400
      canvasRef.current.height = img.offsetHeight || 400
    }
    setImgDims(newDims)
  }, [])

  // Set dims on image load
  const handleImgLoad = useCallback(() => {
    syncCanvasDims()
  }, [syncCanvasDims])

  // Fix 8: ResizeObserver — redraw canvas whenever the image wrapper resizes
  // (e.g. browser window resize). Without this, bbox overlays drift off.
  useEffect(() => {
    if (!wrapperRef.current) return
    const observer = new ResizeObserver(() => {
      syncCanvasDims()
    })
    observer.observe(wrapperRef.current)
    return () => observer.disconnect()
  }, [syncCanvasDims])

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

      <div className="doc-image-wrapper" ref={wrapperRef}>
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
