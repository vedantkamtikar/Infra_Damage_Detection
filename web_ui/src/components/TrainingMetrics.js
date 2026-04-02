import React, { useState, useEffect } from 'react';

const API = 'http://localhost:8000';

function TrainingMetrics() {
  const [metrics, setMetrics] = useState(null);
  const [evalImages, setEvalImages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [modalImage, setModalImage] = useState(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const [metricsRes, imagesRes] = await Promise.all([
          fetch(`${API}/training/metrics`),
          fetch(`${API}/training/images`),
        ]);
        const metricsData = await metricsRes.json();
        const imagesData = await imagesRes.json();
        setMetrics(metricsData);
        setEvalImages(imagesData.images || []);
      } catch (err) {
        console.error('Training metrics fetch error:', err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="spinner-wrapper">
        <div className="spinner"></div>
        <span>Loading training data...</span>
      </div>
    );
  }

  const rows = metrics?.data || [];

  // Find best metrics across all epochs
  const bestMap50 = rows.length > 0
    ? Math.max(...rows.map(r => r['metrics/mAP50(B)'] || 0))
    : 0;
  const bestMap5095 = rows.length > 0
    ? Math.max(...rows.map(r => r['metrics/mAP50-95(B)'] || 0))
    : 0;
  const bestPrecision = rows.length > 0
    ? Math.max(...rows.map(r => r['metrics/precision(B)'] || 0))
    : 0;
  const bestRecall = rows.length > 0
    ? Math.max(...rows.map(r => r['metrics/recall(B)'] || 0))
    : 0;

  const bestMap50Epoch = rows.findIndex(r => (r['metrics/mAP50(B)'] || 0) === bestMap50) + 1;
  const bestMap5095Epoch = rows.findIndex(r => (r['metrics/mAP50-95(B)'] || 0) === bestMap5095) + 1;

  // SVG chart helper
  const renderChart = (dataKeys, colors, labels, title, yLabel) => {
    if (rows.length === 0) return null;
    const width = 600;
    const height = 200;
    const padL = 50, padR = 20, padT = 10, padB = 30;
    const chartW = width - padL - padR;
    const chartH = height - padT - padB;

    const allVals = dataKeys.flatMap(k => rows.map(r => r[k] || 0));
    const maxVal = Math.max(...allVals, 0.01);
    const minVal = Math.min(...allVals, 0);
    const range = maxVal - minVal || 1;

    return (
      <div className="chart-container">
        <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet">
          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map(pct => {
            const y = padT + chartH * (1 - pct);
            const val = minVal + range * pct;
            return (
              <g key={pct}>
                <line x1={padL} y1={y} x2={padL + chartW} y2={y}
                      stroke="rgba(255,255,255,0.06)" strokeWidth="1" />
                <text x={padL - 8} y={y + 4} fill="var(--text-muted)"
                      fontSize="9" textAnchor="end" fontFamily="var(--font-mono)">
                  {val.toFixed(2)}
                </text>
              </g>
            );
          })}

          {/* Axis labels */}
          <text x={padL + chartW / 2} y={height - 4} fill="var(--text-muted)"
                fontSize="10" textAnchor="middle">Epoch</text>

          {/* Lines */}
          {dataKeys.map((key, ki) => {
            const points = rows.map((r, i) => {
              const x = padL + (i / Math.max(rows.length - 1, 1)) * chartW;
              const y = padT + chartH * (1 - ((r[key] || 0) - minVal) / range);
              return `${x},${y}`;
            }).join(' ');

            return (
              <polyline
                key={key}
                points={points}
                fill="none"
                stroke={colors[ki]}
                strokeWidth="2"
                strokeLinejoin="round"
                strokeLinecap="round"
                opacity="0.85"
              />
            );
          })}

          {/* Legend */}
          {labels.map((label, i) => (
            <g key={i}>
              <line x1={padL + i * 120} y1={padT - 2} x2={padL + i * 120 + 20} y2={padT - 2}
                    stroke={colors[i]} strokeWidth="2" />
              <text x={padL + i * 120 + 26} y={padT + 2} fill="var(--text-secondary)"
                    fontSize="9">{label}</text>
            </g>
          ))}
        </svg>
      </div>
    );
  };

  return (
    <div className="animate-slide-up" id="training-metrics-page">
      {/* Run Info */}
      {metrics?.run_name && (
        <div className="card" style={{ marginBottom: '20px', padding: '14px 20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
              📂 Training Run: <strong style={{ color: 'var(--accent-cyan)' }}>{metrics.run_name}</strong>
            </span>
            <span className="mono" style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              {rows.length} epochs
            </span>
          </div>
        </div>
      )}

      {/* Best Metric Cards */}
      <div className="stat-grid">
        <div className="stat-card cyan metric-card">
          <div className="stat-label">Best mAP50</div>
          <div className="stat-value cyan">{(bestMap50 * 100).toFixed(1)}%</div>
          <div className="metric-best">epoch {bestMap50Epoch}</div>
        </div>
        <div className="stat-card green metric-card">
          <div className="stat-label">Best mAP50-95</div>
          <div className="stat-value green">{(bestMap5095 * 100).toFixed(1)}%</div>
          <div className="metric-best">epoch {bestMap5095Epoch}</div>
        </div>
        <div className="stat-card purple metric-card">
          <div className="stat-label">Best Precision</div>
          <div className="stat-value purple">{(bestPrecision * 100).toFixed(1)}%</div>
          <div className="metric-best">peak value</div>
        </div>
        <div className="stat-card orange metric-card">
          <div className="stat-label">Best Recall</div>
          <div className="stat-value orange">{(bestRecall * 100).toFixed(1)}%</div>
          <div className="metric-best">peak value</div>
        </div>
      </div>

      {rows.length > 0 && (
        <>
          {/* Training + Validation Loss Charts */}
          <div className="section-grid">
            <div className="card">
              <div className="card-header">
                <div className="card-title"><span>📉</span> Training Loss</div>
              </div>
              {renderChart(
                ['train/box_loss', 'train/cls_loss', 'train/dfl_loss'],
                ['#ef4444', '#3b82f6', '#10b981'],
                ['Box', 'Cls', 'DFL'],
                'Training Loss', 'Loss'
              )}
            </div>
            <div className="card">
              <div className="card-header">
                <div className="card-title"><span>📉</span> Validation Loss</div>
              </div>
              {renderChart(
                ['val/box_loss', 'val/cls_loss', 'val/dfl_loss'],
                ['#ef4444', '#3b82f6', '#10b981'],
                ['Box', 'Cls', 'DFL'],
                'Validation Loss', 'Loss'
              )}
            </div>
          </div>

          {/* mAP + Precision/Recall Charts */}
          <div className="section-grid">
            <div className="card">
              <div className="card-header">
                <div className="card-title"><span>📈</span> mAP Metrics</div>
              </div>
              {renderChart(
                ['metrics/mAP50(B)', 'metrics/mAP50-95(B)'],
                ['#8b5cf6', '#f59e0b'],
                ['mAP50', 'mAP50-95'],
                'mAP', 'Score'
              )}
            </div>
            <div className="card">
              <div className="card-header">
                <div className="card-title"><span>📈</span> Precision & Recall</div>
              </div>
              {renderChart(
                ['metrics/precision(B)', 'metrics/recall(B)'],
                ['#06b6d4', '#ef4444'],
                ['Precision', 'Recall'],
                'Precision & Recall', 'Score'
              )}
            </div>
          </div>
        </>
      )}

      {rows.length === 0 && (
        <div className="card">
          <div className="empty-state">
            <div className="empty-state-icon">📊</div>
            <div className="empty-state-text">No training data available</div>
            <div className="empty-state-hint">
              Run training first, then start the FastAPI server to view metrics
            </div>
          </div>
        </div>
      )}

      {/* Model Config */}
      <div className="card" style={{ marginTop: '4px' }}>
        <div className="card-header">
          <div className="card-title"><span>🧠</span> Training Configuration</div>
        </div>
        <div className="param-grid">
          <div className="param-item">
            <span className="param-label">Architecture</span>
            <span className="param-value text-accent">YOLO26m</span>
          </div>
          <div className="param-item">
            <span className="param-label">Max Epochs</span>
            <span className="param-value" style={{ color: 'var(--text-primary)' }}>300</span>
          </div>
          <div className="param-item">
            <span className="param-label">Image Size</span>
            <span className="param-value" style={{ color: 'var(--text-primary)' }}>640×640</span>
          </div>
          <div className="param-item">
            <span className="param-label">Batch Size</span>
            <span className="param-value" style={{ color: 'var(--text-primary)' }}>10</span>
          </div>
          <div className="param-item">
            <span className="param-label">Optimizer</span>
            <span className="param-value text-green">AdamW</span>
          </div>
          <div className="param-item">
            <span className="param-label">LR (initial)</span>
            <span className="param-value" style={{ color: 'var(--text-primary)' }}>0.0001</span>
          </div>
          <div className="param-item">
            <span className="param-label">Patience</span>
            <span className="param-value" style={{ color: 'var(--text-primary)' }}>30</span>
          </div>
          <div className="param-item">
            <span className="param-label">Frozen Layers</span>
            <span className="param-value text-purple">10</span>
          </div>
        </div>
      </div>

      {/* Evaluation Images Gallery */}
      {evalImages.length > 0 && (
        <div className="card" style={{ marginTop: '20px' }}>
          <div className="card-header">
            <div className="card-title"><span>🖼️</span> Evaluation Plots</div>
            <span className="card-subtitle">{evalImages.length} images</span>
          </div>
          <div className="gallery-grid" id="eval-gallery">
            {evalImages.map((img, i) => (
              <div
                className="gallery-item"
                key={i}
                onClick={() => setModalImage(`${API}/training/images${img.path}`)}
              >
                <img
                  src={`${API}/training/images${img.path}`}
                  alt={img.name}
                  loading="lazy"
                />
                <div className="gallery-item-label">
                  {img.name.replace(/\.(png|jpg|jpeg)$/i, '').replace(/[_-]/g, ' ')}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Modal */}
      {modalImage && (
        <div className="modal-overlay" onClick={() => setModalImage(null)} id="training-image-modal">
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setModalImage(null)}>✕</button>
            <img src={modalImage} alt="Evaluation plot" />
          </div>
        </div>
      )}
    </div>
  );
}

export default TrainingMetrics;
