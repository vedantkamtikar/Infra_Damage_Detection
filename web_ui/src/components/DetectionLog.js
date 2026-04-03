import React, { useState, useEffect, useCallback } from 'react';

const API = 'http://localhost:8000';
const PAGE_SIZE = 15;

function DetectionLog() {
  const [activeTab, setActiveTab] = useState('infrastructure');

  // Infrastructure detection state
  const [logs, setLogs] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [minConf, setMinConf] = useState(0);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState(null);
  const [modalImage, setModalImage] = useState(null);

  // Pothole detection state
  const [potholeLogs, setPotholeLogs] = useState([]);
  const [potholeTotal, setPotholeTotal] = useState(0);
  const [potholePage, setPotholePage] = useState(0);
  const [potholeMinConf, setPotholeMinConf] = useState(0);
  const [potholeLoading, setPotholeLoading] = useState(true);
  const [potholeExpandedId, setPotholeExpandedId] = useState(null);

  // Fetch infrastructure logs
  const fetchLogs = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(
        `${API}/logs?limit=${PAGE_SIZE}&offset=${page * PAGE_SIZE}&min_conf=${minConf}`
      );
      const data = await res.json();
      setLogs(data.logs || []);
      setTotal(data.total || 0);
    } catch (err) {
      console.error('Fetch logs error:', err);
      setLogs([]);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  }, [page, minConf]);

  // Fetch pothole logs
  const fetchPotholeLogs = useCallback(async () => {
    setPotholeLoading(true);
    try {
      const res = await fetch(
        `${API}/pothole-logs?limit=${PAGE_SIZE}&offset=${potholePage * PAGE_SIZE}&min_conf=${potholeMinConf}`
      );
      const data = await res.json();
      setPotholeLogs(data.logs || []);
      setPotholeTotal(data.total || 0);
    } catch (err) {
      console.error('Fetch pothole logs error:', err);
      setPotholeLogs([]);
      setPotholeTotal(0);
    } finally {
      setPotholeLoading(false);
    }
  }, [potholePage, potholeMinConf]);

  useEffect(() => {
    fetchLogs();
  }, [fetchLogs]);

  useEffect(() => {
    fetchPotholeLogs();
  }, [fetchPotholeLogs]);

  const formatTimestamp = (ts) => {
    if (!ts) return '—';
    const date = new Date(ts * 1000);
    return date.toLocaleString();
  };

  const confClass = (c) => c >= 0.7 ? 'high' : c >= 0.4 ? 'medium' : 'low';
  const totalPages = Math.ceil(total / PAGE_SIZE);
  const potholeTotalPages = Math.ceil(potholeTotal / PAGE_SIZE);

  return (
    <div className="animate-slide-up" id="detection-log-page">
      {/* Tab Switcher */}
      <div className="log-tabs" id="log-tab-switcher">
        <button
          className={`log-tab ${activeTab === 'infrastructure' ? 'active' : ''}`}
          onClick={() => setActiveTab('infrastructure')}
          id="tab-infrastructure"
        >
          <span className="log-tab-icon">🏗️</span>
          <span className="log-tab-label">Infrastructure Damage</span>
          <span className="log-tab-count">{total}</span>
        </button>
        <button
          className={`log-tab ${activeTab === 'pothole' ? 'active' : ''}`}
          onClick={() => setActiveTab('pothole')}
          id="tab-pothole"
        >
          <span className="log-tab-icon">🕳️</span>
          <span className="log-tab-label">Pothole Detections</span>
          <span className="log-tab-count">{potholeTotal}</span>
        </button>
      </div>

      {/* ═══════════════════════════════════════════
          INFRASTRUCTURE DAMAGE TAB
          ═══════════════════════════════════════════ */}
      {activeTab === 'infrastructure' && (
        <>
          {/* Filters */}
          <div className="card" style={{ marginBottom: '20px' }}>
            <div className="card-header" style={{ marginBottom: '12px' }}>
              <div className="card-title">
                <span>🎚️</span> Filters
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                {total} total record{total !== 1 ? 's' : ''}
              </div>
            </div>
            <div className="input-group">
              <label htmlFor="conf-slider">Min Confidence</label>
              <input
                type="range"
                id="conf-slider"
                className="slider"
                min="0"
                max="1"
                step="0.05"
                value={minConf}
                onChange={(e) => {
                  setMinConf(parseFloat(e.target.value));
                  setPage(0);
                }}
              />
              <span className="slider-value">{(minConf * 100).toFixed(0)}%</span>
            </div>
          </div>

          {/* Table */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <span>📋</span> Detection Records
              </div>
              <button
                className="btn btn-secondary"
                style={{ padding: '6px 12px', fontSize: '11px' }}
                onClick={fetchLogs}
                id="refresh-logs-btn"
              >
                ↻ Refresh
              </button>
            </div>

            {loading ? (
              <div className="spinner-wrapper">
                <div className="spinner"></div>
                <span>Loading detections...</span>
              </div>
            ) : logs.length === 0 ? (
              <div className="empty-state">
                <div className="empty-state-icon">📭</div>
                <div className="empty-state-text">No detections found</div>
                <div className="empty-state-hint">
                  {minConf > 0
                    ? 'Try lowering the confidence threshold'
                    : 'Run the drone mission to generate detections'}
                </div>
              </div>
            ) : (
              <>
                <div className="data-table-wrapper">
                  <table className="data-table" id="detection-log-table">
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>Timestamp</th>
                        <th>Class</th>
                        <th>Confidence</th>
                        <th>X</th>
                        <th>Y</th>
                        <th>Z</th>
                        <th>Frame</th>
                      </tr>
                    </thead>
                    <tbody>
                      {logs.map(log => (
                        <React.Fragment key={log.id}>
                          <tr
                            onClick={() => setExpandedId(expandedId === log.id ? null : log.id)}
                            style={{ cursor: 'pointer' }}
                          >
                            <td className="mono" style={{ color: 'var(--text-muted)' }}>
                              #{log.id}
                            </td>
                            <td style={{ fontSize: '12px' }}>
                              {formatTimestamp(log.timestamp)}
                            </td>
                            <td>
                              <span className="tag">{log.label}</span>
                            </td>
                            <td>
                              <span className={`conf-badge ${confClass(log.confidence)}`}>
                                {(log.confidence * 100).toFixed(1)}%
                              </span>
                            </td>
                            <td className="mono" style={{ fontSize: '12px' }}>{log.location.x}</td>
                            <td className="mono" style={{ fontSize: '12px' }}>{log.location.y}</td>
                            <td className="mono" style={{ fontSize: '12px' }}>{log.location.z}</td>
                            <td>
                              {log.image ? (
                                <button
                                  className="btn btn-secondary"
                                  style={{ padding: '3px 8px', fontSize: '10px' }}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setModalImage(log.image);
                                  }}
                                >
                                  🖼 View
                                </button>
                              ) : (
                                <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>—</span>
                              )}
                            </td>
                          </tr>
                          {expandedId === log.id && (
                            <tr className="det-detail-row">
                              <td colSpan="8">
                                <div className="det-detail-content">
                                  {log.image && (
                                    <img
                                      src={log.image}
                                      alt={`Detection ${log.id}`}
                                      className="det-detail-image"
                                      onClick={() => setModalImage(log.image)}
                                      style={{ cursor: 'pointer' }}
                                    />
                                  )}
                                  <div className="det-detail-info">
                                    <p><strong>Detection ID:</strong> #{log.id}</p>
                                    <p><strong>Class:</strong> {log.label}</p>
                                    <p><strong>Confidence:</strong> {(log.confidence * 100).toFixed(2)}%</p>
                                    <p><strong>Timestamp:</strong> {formatTimestamp(log.timestamp)}</p>
                                    <p>
                                      <strong>NED Position:</strong>{' '}
                                      <span className="mono">
                                        ({log.location.x}, {log.location.y}, {log.location.z})
                                      </span>
                                    </p>
                                  </div>
                                </div>
                              </td>
                            </tr>
                          )}
                        </React.Fragment>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Pagination */}
                <div className="pagination" id="detection-pagination">
                  <button
                    className="pagination-btn"
                    disabled={page === 0}
                    onClick={() => setPage(p => p - 1)}
                  >
                    ← Prev
                  </button>
                  <span className="pagination-info">
                    Page {page + 1} of {totalPages || 1}
                  </span>
                  <button
                    className="pagination-btn"
                    disabled={page >= totalPages - 1}
                    onClick={() => setPage(p => p + 1)}
                  >
                    Next →
                  </button>
                </div>
              </>
            )}
          </div>
        </>
      )}

      {/* ═══════════════════════════════════════════
          POTHOLE DETECTIONS TAB
          ═══════════════════════════════════════════ */}
      {activeTab === 'pothole' && (
        <>
          {/* Filters */}
          <div className="card" style={{ marginBottom: '20px' }}>
            <div className="card-header" style={{ marginBottom: '12px' }}>
              <div className="card-title">
                <span>🎚️</span> Filters
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                {potholeTotal} total record{potholeTotal !== 1 ? 's' : ''}
              </div>
            </div>
            <div className="input-group">
              <label htmlFor="pothole-conf-slider">Min Confidence</label>
              <input
                type="range"
                id="pothole-conf-slider"
                className="slider"
                min="0"
                max="1"
                step="0.05"
                value={potholeMinConf}
                onChange={(e) => {
                  setPotholeMinConf(parseFloat(e.target.value));
                  setPotholePage(0);
                }}
              />
              <span className="slider-value">{(potholeMinConf * 100).toFixed(0)}%</span>
            </div>
          </div>

          {/* Table */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                <span>🕳️</span> Pothole Records
              </div>
              <button
                className="btn btn-secondary"
                style={{ padding: '6px 12px', fontSize: '11px' }}
                onClick={fetchPotholeLogs}
                id="refresh-pothole-logs-btn"
              >
                ↻ Refresh
              </button>
            </div>

            {potholeLoading ? (
              <div className="spinner-wrapper">
                <div className="spinner"></div>
                <span>Loading pothole detections...</span>
              </div>
            ) : potholeLogs.length === 0 ? (
              <div className="empty-state">
                <div className="empty-state-icon">🛣️</div>
                <div className="empty-state-text">No pothole detections found</div>
                <div className="empty-state-hint">
                  {potholeMinConf > 0
                    ? 'Try lowering the confidence threshold'
                    : 'Run the road inspection mission to detect potholes'}
                </div>
              </div>
            ) : (
              <>
                <div className="data-table-wrapper">
                  <table className="data-table" id="pothole-log-table">
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>Timestamp</th>
                        <th>Class</th>
                        <th>Confidence</th>
                        <th>X</th>
                        <th>Y</th>
                        <th>Z</th>
                        <th>Frame</th>
                      </tr>
                    </thead>
                    <tbody>
                      {potholeLogs.map(log => (
                        <React.Fragment key={log.id}>
                          <tr
                            onClick={() => setPotholeExpandedId(potholeExpandedId === log.id ? null : log.id)}
                            style={{ cursor: 'pointer' }}
                          >
                            <td className="mono" style={{ color: 'var(--text-muted)' }}>
                              #{log.id}
                            </td>
                            <td style={{ fontSize: '12px' }}>
                              {formatTimestamp(log.timestamp)}
                            </td>
                            <td>
                              <span className="tag pothole-tag">{log.label}</span>
                            </td>
                            <td>
                              <span className={`conf-badge ${confClass(log.confidence)}`}>
                                {(log.confidence * 100).toFixed(1)}%
                              </span>
                            </td>
                            <td className="mono" style={{ fontSize: '12px' }}>{log.location.x}</td>
                            <td className="mono" style={{ fontSize: '12px' }}>{log.location.y}</td>
                            <td className="mono" style={{ fontSize: '12px' }}>{log.location.z}</td>
                            <td>
                              {log.image ? (
                                <button
                                  className="btn btn-secondary"
                                  style={{ padding: '3px 8px', fontSize: '10px' }}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setModalImage(log.image);
                                  }}
                                >
                                  🖼 View
                                </button>
                              ) : (
                                <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>—</span>
                              )}
                            </td>
                          </tr>
                          {potholeExpandedId === log.id && (
                            <tr className="det-detail-row pothole-detail-row">
                              <td colSpan="8">
                                <div className="det-detail-content">
                                  {log.image && (
                                    <img
                                      src={log.image}
                                      alt={`Pothole ${log.id}`}
                                      className="det-detail-image"
                                      onClick={() => setModalImage(log.image)}
                                      style={{ cursor: 'pointer' }}
                                    />
                                  )}
                                  <div className="det-detail-info">
                                    <p><strong>Pothole ID:</strong> #{log.id}</p>
                                    <p><strong>Confidence:</strong> {(log.confidence * 100).toFixed(2)}%</p>
                                    <p><strong>Timestamp:</strong> {formatTimestamp(log.timestamp)}</p>
                                    <p>
                                      <strong>Drone Position (NED):</strong>{' '}
                                      <span className="mono">
                                        ({log.location.x}, {log.location.y}, {log.location.z})
                                      </span>
                                    </p>
                                  </div>
                                </div>
                              </td>
                            </tr>
                          )}
                        </React.Fragment>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Pagination */}
                <div className="pagination" id="pothole-pagination">
                  <button
                    className="pagination-btn"
                    disabled={potholePage === 0}
                    onClick={() => setPotholePage(p => p - 1)}
                  >
                    ← Prev
                  </button>
                  <span className="pagination-info">
                    Page {potholePage + 1} of {potholeTotalPages || 1}
                  </span>
                  <button
                    className="pagination-btn"
                    disabled={potholePage >= potholeTotalPages - 1}
                    onClick={() => setPotholePage(p => p + 1)}
                  >
                    Next →
                  </button>
                </div>
              </>
            )}
          </div>
        </>
      )}

      {/* Modal */}
      {modalImage && (
        <div
          className="modal-overlay"
          onClick={() => setModalImage(null)}
          id="image-modal"
        >
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <button
              className="modal-close"
              onClick={() => setModalImage(null)}
            >
              ✕
            </button>
            <img src={modalImage} alt="Detection frame" />
          </div>
        </div>
      )}
    </div>
  );
}

export default DetectionLog;
