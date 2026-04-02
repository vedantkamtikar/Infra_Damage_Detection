import React, { useState, useRef } from 'react';

const API = 'http://localhost:8000';

function ImageDetect() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragover, setDragover] = useState(false);
  const fileRef = useRef();

  const handleFile = (file) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file (JPEG, PNG)');
      return;
    }
    setSelectedFile(file);
    setResult(null);
    setError(null);

    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragover(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  };

  const handleDetect = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const res = await fetch(`${API}/detect`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);

      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Detection failed. Is the FastAPI server running?');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (fileRef.current) fileRef.current.value = '';
  };

  const confClass = (c) => c >= 0.7 ? 'high' : c >= 0.4 ? 'medium' : 'low';

  return (
    <div className="animate-slide-up" id="image-detect-page">
      {/* Upload Zone */}
      {!preview && (
        <div
          className={`upload-zone ${dragover ? 'dragover' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragover(true); }}
          onDragLeave={() => setDragover(false)}
          onDrop={handleDrop}
          id="upload-dropzone"
        >
          <div className="upload-zone-icon">📸</div>
          <div className="upload-zone-text">
            <strong>Drag & drop</strong> an image here, or <strong>click to browse</strong>
          </div>
          <div className="upload-zone-hint">
            Supports JPEG, PNG — image will be processed by YOLO26m for crack detection
          </div>
          <input
            type="file"
            ref={fileRef}
            accept="image/*"
            onChange={(e) => handleFile(e.target.files[0])}
          />
        </div>
      )}

      {/* Preview & Results */}
      {preview && (
        <>
          {/* Action Bar */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: '20px',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                📁 {selectedFile?.name}
              </span>
              <span style={{
                fontSize: '11px',
                color: 'var(--text-muted)',
                fontFamily: 'var(--font-mono)',
              }}>
                ({(selectedFile?.size / 1024).toFixed(1)} KB)
              </span>
            </div>
            <div style={{ display: 'flex', gap: '10px' }}>
              <button
                className="btn btn-secondary"
                onClick={handleReset}
                id="reset-upload-btn"
              >
                ↻ Reset
              </button>
              <button
                className="btn btn-primary"
                onClick={handleDetect}
                disabled={loading}
                id="run-detection-btn"
              >
                {loading ? '⏳ Detecting...' : '🔍 Run Detection'}
              </button>
            </div>
          </div>

          {/* Loading */}
          {loading && (
            <div className="spinner-wrapper">
              <div className="spinner"></div>
              <span>Running YOLO26m inference...</span>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="card" style={{
              borderColor: 'rgba(239, 68, 68, 0.3)',
              background: 'rgba(239, 68, 68, 0.05)',
              marginBottom: '20px',
            }}>
              <div style={{ color: 'var(--accent-red)', fontSize: '13px' }}>
                ⚠️ {error}
              </div>
            </div>
          )}

          {/* Image Compare */}
          {!loading && (
            <div className="image-compare" id="detection-results-compare">
              <div className="image-compare-panel">
                <div className="image-compare-label">Original Image</div>
                <img src={preview} alt="Original upload" />
              </div>
              <div className="image-compare-panel">
                <div className="image-compare-label">
                  {result ? 'Annotated Result' : 'Awaiting Detection'}
                </div>
                {result ? (
                  <img src={result.annotated_image} alt="Annotated result" />
                ) : (
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    minHeight: '300px',
                    color: 'var(--text-muted)',
                    fontSize: '13px',
                    flexDirection: 'column',
                    gap: '12px',
                  }}>
                    <span style={{ fontSize: '36px', opacity: 0.3 }}>🔍</span>
                    Click "Run Detection" to analyze
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Detection Results */}
          {result && !loading && (
            <div className="card" style={{ marginTop: '20px' }}>
              <div className="card-header">
                <div className="card-title">
                  <span>🎯</span> Detection Results
                </div>
                <span className={`conf-badge ${result.count > 0 ? 'high' : 'low'}`}>
                  {result.count} detection{result.count !== 1 ? 's' : ''}
                </span>
              </div>
              {result.count === 0 ? (
                <div className="empty-state" style={{ padding: '24px' }}>
                  <div className="empty-state-text">No damage detected</div>
                  <div className="empty-state-hint">
                    The model did not find cracks in this image at ≥30% confidence
                  </div>
                </div>
              ) : (
                <div className="data-table-wrapper">
                  <table className="data-table" id="upload-results-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Class</th>
                        <th>Confidence</th>
                        <th>Bounding Box</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.detections.map((det, i) => (
                        <tr key={i}>
                          <td className="mono" style={{ color: 'var(--text-muted)' }}>{i + 1}</td>
                          <td><span className="tag">{det.class}</span></td>
                          <td>
                            <span className={`conf-badge ${confClass(det.confidence)}`}>
                              {(det.confidence * 100).toFixed(1)}%
                            </span>
                          </td>
                          <td className="mono" style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                            [{det.bbox.map(v => Math.round(v)).join(', ')}]
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default ImageDetect;
