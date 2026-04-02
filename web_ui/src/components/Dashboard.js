import React, { useState, useEffect } from 'react';

const API = 'http://localhost:8000';

function Dashboard({ onNavigate }) {
  const [stats, setStats] = useState(null);
  const [recentLogs, setRecentLogs] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const [statsRes, logsRes] = await Promise.all([
          fetch(`${API}/stats`),
          fetch(`${API}/logs?limit=8`),
        ]);
        const statsData = await statsRes.json();
        const logsData = await logsRes.json();
        setStats(statsData);
        setRecentLogs(logsData.logs || []);
      } catch (err) {
        console.error('Dashboard fetch error:', err);
        // Use demo data if API not available
        setStats({
          total_detections: 0,
          avg_confidence: 0,
          max_confidence: 0,
          min_confidence: 0,
          unique_classes: [],
          class_counts: {},
          confidence_distribution: Array.from({length: 10}, (_, i) => ({
            range: `${(i*0.1).toFixed(1)}-${((i+1)*0.1).toFixed(1)}`,
            count: 0
          })),
        });
        setRecentLogs([]);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  const formatTimestamp = (ts) => {
    if (!ts) return '—';
    const date = new Date(ts * 1000);
    return date.toLocaleString();
  };

  const confClass = (c) => c >= 0.7 ? 'high' : c >= 0.4 ? 'medium' : 'low';

  if (loading) {
    return (
      <div className="spinner-wrapper">
        <div className="spinner"></div>
        <span>Loading dashboard data...</span>
      </div>
    );
  }

  const maxBin = stats ? Math.max(...stats.confidence_distribution.map(b => b.count), 1) : 1;

  return (
    <div className="animate-slide-up" id="dashboard-page">
      {/* Stat Cards */}
      <div className="stat-grid">
        <div className="stat-card cyan">
          <div className="stat-label">Total Detections</div>
          <div className="stat-value cyan">{stats?.total_detections ?? 0}</div>
          <div className="stat-detail">
            {stats?.unique_classes?.length ?? 0} class{stats?.unique_classes?.length !== 1 ? 'es' : ''}
          </div>
        </div>
        <div className="stat-card green">
          <div className="stat-label">Avg Confidence</div>
          <div className="stat-value green">
            {stats?.avg_confidence ? (stats.avg_confidence * 100).toFixed(1) + '%' : '—'}
          </div>
          <div className="stat-detail">
            max {stats?.max_confidence ? (stats.max_confidence * 100).toFixed(1) + '%' : '—'}
          </div>
        </div>
        <div className="stat-card purple">
          <div className="stat-label">Detection Classes</div>
          <div className="stat-value purple">{stats?.unique_classes?.length ?? 0}</div>
          <div className="stat-detail">
            {stats?.unique_classes?.join(', ') || 'none'}
          </div>
        </div>
        <div className="stat-card orange">
          <div className="stat-label">Confidence Range</div>
          <div className="stat-value orange">
            {stats?.min_confidence ? (stats.min_confidence * 100).toFixed(0) : 0}–
            {stats?.max_confidence ? (stats.max_confidence * 100).toFixed(0) : 0}%
          </div>
          <div className="stat-detail">min – max</div>
        </div>
      </div>

      {/* Main Grid */}
      <div className="section-grid">
        {/* Recent Detections Table */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <span>🔍</span> Recent Detections
            </div>
            <button
              className="btn btn-secondary"
              style={{ padding: '6px 12px', fontSize: '11px' }}
              onClick={() => onNavigate && onNavigate('detections')}
              id="dashboard-view-all-btn"
            >
              View All →
            </button>
          </div>
          {recentLogs.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">📭</div>
              <div className="empty-state-text">No detections yet</div>
              <div className="empty-state-hint">
                Run the drone mission or start the FastAPI server
              </div>
            </div>
          ) : (
            <div className="data-table-wrapper">
              <table className="data-table" id="recent-detections-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Timestamp</th>
                    <th>Class</th>
                    <th>Confidence</th>
                    <th>Position (NED)</th>
                  </tr>
                </thead>
                <tbody>
                  {recentLogs.map(log => (
                    <tr key={log.id}>
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
                      <td>
                        <div className="ned-coords">
                          <span>x: {log.location.x}</span>
                          <span>y: {log.location.y}</span>
                          <span>z: {log.location.z}</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Confidence Distribution */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <span>📊</span> Confidence Distribution
            </div>
          </div>
          {stats?.total_detections === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">📉</div>
              <div className="empty-state-text">No data</div>
              <div className="empty-state-hint">Run detections to populate the histogram</div>
            </div>
          ) : (
            <div className="histogram" id="confidence-histogram">
              {stats?.confidence_distribution.map((bin, i) => {
                const pct = (bin.count / maxBin) * 100;
                const hue = 180 + (i * 15);  // cyan → green → yellow
                return (
                  <div className="histogram-bar-group" key={i}>
                    <div
                      className="histogram-bar"
                      style={{
                        height: `${Math.max(pct, 2)}%`,
                        background: `hsl(${hue}, 70%, 55%)`,
                        opacity: bin.count > 0 ? 1 : 0.2,
                      }}
                      title={`${bin.range}: ${bin.count} detections`}
                    ></div>
                    <span className="histogram-label">{bin.range.split('-')[0]}</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* System Overview */}
      <div className="card" style={{ marginTop: '4px' }}>
        <div className="card-header">
          <div className="card-title">
            <span>⚙️</span> System Overview
          </div>
        </div>
        <div className="param-grid">
          <div className="param-item">
            <span className="param-label">Model</span>
            <span className="param-value text-accent">YOLO26m</span>
          </div>
          <div className="param-item">
            <span className="param-label">Weights</span>
            <span className="param-value" style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>best_run9.pt</span>
          </div>
          <div className="param-item">
            <span className="param-label">Conf Threshold</span>
            <span className="param-value text-green">0.30</span>
          </div>
          <div className="param-item">
            <span className="param-label">Database</span>
            <span className="param-value" style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>detections.db</span>
          </div>
          <div className="param-item">
            <span className="param-label">Simulation</span>
            <span className="param-value" style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>UE 4.27 + AirSim</span>
          </div>
          <div className="param-item">
            <span className="param-label">Class Detection</span>
            <span className="param-value text-orange">crack</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
