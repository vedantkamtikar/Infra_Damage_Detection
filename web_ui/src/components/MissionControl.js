import React, { useState, useEffect, useRef, useCallback } from 'react';

const API = 'http://localhost:8000';

function MissionControl() {
  const [missionData, setMissionData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [missionRunning, setMissionRunning] = useState(false);
  const [missionElapsed, setMissionElapsed] = useState(0);
  const [missionPid, setMissionPid] = useState(null);
  const [startError, setStartError] = useState(null);
  const [feedActive, setFeedActive] = useState(false);
  const feedRef = useRef(null);
  const feedIntervalRef = useRef(null);
  const statusIntervalRef = useRef(null);

  // Fetch mission config
  useEffect(() => {
    async function fetchMission() {
      try {
        const res = await fetch(`${API}/mission`);
        const data = await res.json();
        setMissionData(data);
      } catch (err) {
        console.error('Mission fetch error:', err);
        setMissionData({
          model: 'YOLO26m (best_run9.pt)',
          conf_threshold: 0.3,
          orbit: { center: { x: 19.0, y: 0.0 }, radius_m: 19.0, altitude_m: 10.0, velocity_mps: 3.0 },
          vision: { target_fps: 18, log_cooldown_s: 2.0, camera: 'front_center (Camera 0)' },
          pipeline: [
            'Capture RGB frame via AirSim',
            'Run YOLO26m inference (conf ≥ 0.30)',
            'Extract class, confidence, bounding box',
            'Log to SQLite with NED coordinates',
            'Display live annotated feed',
          ],
        });
      } finally {
        setLoading(false);
      }
    }
    fetchMission();
  }, []);

  // Poll mission status
  const checkStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API}/mission/status`);
      const data = await res.json();
      setMissionRunning(data.running);
      if (data.running) {
        setMissionElapsed(data.elapsed_s || 0);
        setMissionPid(data.pid || null);
      } else {
        setMissionPid(null);
      }
    } catch (err) {
      // Server not available
    }
  }, []);

  useEffect(() => {
    checkStatus();
    statusIntervalRef.current = setInterval(checkStatus, 2000);
    return () => clearInterval(statusIntervalRef.current);
  }, [checkStatus]);

  // Live feed polling
  useEffect(() => {
    if (missionRunning) {
      setFeedActive(true);
      feedIntervalRef.current = setInterval(() => {
        if (feedRef.current) {
          // Append cache-buster to force reload
          feedRef.current.src = `${API}/stream/latest?t=${Date.now()}`;
        }
      }, 200); // ~5 FPS on the web side
    } else {
      if (feedIntervalRef.current) clearInterval(feedIntervalRef.current);
      // Keep last frame visible for a bit after mission ends
    }
    return () => {
      if (feedIntervalRef.current) clearInterval(feedIntervalRef.current);
    };
  }, [missionRunning]);

  const handleStart = async () => {
    setStartError(null);
    try {
      const res = await fetch(`${API}/mission/start`, { method: 'POST' });
      const data = await res.json();
      if (data.error) {
        setStartError(data.error);
      } else {
        setMissionRunning(true);
        setMissionPid(data.pid);
        setMissionElapsed(0);
        setFeedActive(true);
      }
    } catch (err) {
      setStartError('Cannot reach server. Is FastAPI running?');
    }
  };

  const handleStop = async () => {
    try {
      await fetch(`${API}/mission/stop`, { method: 'POST' });
      setMissionRunning(false);
      setMissionPid(null);
    } catch (err) {
      console.error('Stop error:', err);
    }
  };

  const handleRoadInspect = async () => {
    setStartError(null);
    try {
      const res = await fetch(`${API}/mission/road-inspect/start`, { method: 'POST' });
      const data = await res.json();
      if (data.error) {
        setStartError(data.error);
      } else {
        setMissionRunning(true);
        setMissionPid(data.pid);
        setMissionElapsed(0);
        setFeedActive(true);
      }
    } catch (err) {
      setStartError('Cannot reach server. Is FastAPI running?');
    }
  };

  const formatElapsed = (secs) => {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  if (loading) {
    return (
      <div className="spinner-wrapper">
        <div className="spinner"></div>
        <span>Loading mission data...</span>
      </div>
    );
  }

  const orbit = missionData?.orbit || {};
  const vision = missionData?.vision || {};

  return (
    <div className="animate-slide-up" id="mission-control-page">

      {/* ══ Live Feed + Controls ══ */}
      <div className="card" style={{ marginBottom: '20px', padding: '0', overflow: 'hidden' }}>
        {/* Feed Header Bar */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '14px 20px',
          borderBottom: '1px solid var(--border-subtle)',
          background: 'rgba(255,255,255,0.02)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{
              width: '10px', height: '10px', borderRadius: '50%',
              background: missionRunning ? 'var(--accent-green)' : 'var(--text-muted)',
              animation: missionRunning ? 'pulse 1.5s ease-in-out infinite' : 'none',
              boxShadow: missionRunning ? '0 0 8px rgba(16, 185, 129, 0.5)' : 'none',
            }}></div>
            <span style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>
              📹 Live Inspection Feed
            </span>
            {missionRunning && (
              <span className="mono" style={{
                fontSize: '12px', color: 'var(--accent-green)',
                padding: '2px 10px',
                background: 'rgba(16, 185, 129, 0.1)',
                borderRadius: '20px',
                border: '1px solid rgba(16, 185, 129, 0.2)',
              }}>
                ● LIVE — {formatElapsed(missionElapsed)}
              </span>
            )}
            {!missionRunning && feedActive && (
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                Feed paused
              </span>
            )}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            {missionPid && (
              <span className="mono" style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                PID {missionPid}
              </span>
            )}
            {!missionRunning ? (
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  className="btn btn-primary"
                  onClick={handleStart}
                  id="start-inspection-btn"
                  style={{ padding: '8px 20px', fontSize: '13px' }}
                >
                  Inspect Building
                </button>
                <button
                  className="btn"
                  onClick={handleRoadInspect}
                  id="road-inspect-btn"
                  style={{
                    padding: '8px 20px',
                    fontSize: '13px',
                    background: 'rgba(251, 191, 36, 0.1)',
                    color: 'var(--accent-orange)',
                    border: '1px solid rgba(251, 191, 36, 0.3)',
                  }}
                >
                  Inspect Road
                </button>
              </div>
            ) : (
              <button
                className="btn btn-danger"
                onClick={handleStop}
                id="stop-inspection-btn"
                style={{ padding: '8px 20px', fontSize: '13px' }}
              >
                ⏹ Stop Mission
              </button>
            )}
          </div>
        </div>

        {/* Feed Display Area */}
        <div style={{
          background: '#000',
          minHeight: '400px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative',
        }}
          id="live-feed-container"
        >
          {feedActive || missionRunning ? (
            <img
              ref={feedRef}
              src={`${API}/stream/latest?t=${Date.now()}`}
              alt="Live drone inspection feed"
              onError={(e) => {
                // If frame not available yet, show placeholder styling
                e.target.style.display = 'none';
              }}
              onLoad={(e) => {
                e.target.style.display = 'block';
              }}
              style={{
                maxWidth: '100%',
                maxHeight: '500px',
                display: 'block',
                margin: '0 auto',
              }}
            />
          ) : null}

          {/* Placeholder when no feed */}
          {!feedActive && !missionRunning && (
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '16px',
              color: 'var(--text-muted)',
              padding: '40px',
            }}>
              <span style={{ fontSize: '56px', opacity: 0.25 }}>📡</span>
              <span style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
                No active feed
              </span>
              <span style={{ fontSize: '12px', maxWidth: '360px', textAlign: 'center', lineHeight: 1.5 }}>
                Press <strong style={{ color: 'var(--accent-cyan)' }}>Start Inspection</strong> to
                launch orbit.py and begin streaming the drone's camera feed with real-time YOLO detections
              </span>
            </div>
          )}

          {/* Waiting for first frame */}
          {missionRunning && missionElapsed < 5 && (
            <div style={{
              position: 'absolute',
              bottom: '16px',
              left: '50%',
              transform: 'translateX(-50%)',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '6px 16px',
              background: 'rgba(0,0,0,0.7)',
              borderRadius: '20px',
              fontSize: '11px',
              color: 'var(--accent-cyan)',
            }}>
              <div className="spinner" style={{ width: '14px', height: '14px', borderWidth: '2px' }}></div>
              Connecting to AirSim...
            </div>
          )}
        </div>

        {/* Error bar */}
        {startError && (
          <div style={{
            padding: '10px 20px',
            background: 'rgba(239, 68, 68, 0.08)',
            borderTop: '1px solid rgba(239, 68, 68, 0.2)',
            fontSize: '12px',
            color: 'var(--accent-red)',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}>
            ⚠️ {startError}
            <button
              onClick={() => setStartError(null)}
              style={{
                marginLeft: 'auto',
                background: 'none',
                color: 'var(--text-muted)',
                fontSize: '14px',
              }}
            >✕</button>
          </div>
        )}
      </div>

      {/* ══ Mission Parameters ══ */}
      <div className="card" style={{ marginTop: '20px' }}>
        <div className="card-header">
          <div className="card-title"><span>⚙️</span> Mission Parameters</div>
        </div>
        <div className="param-grid">
          <div className="param-item">
            <span className="param-label">Orbit Radius</span>
            <span className="param-value text-accent">{orbit.radius_m || 19}m</span>
          </div>
          <div className="param-item">
            <span className="param-label">Altitude</span>
            <span className="param-value text-green">{orbit.altitude_m || 10}m</span>
          </div>
          <div className="param-item">
            <span className="param-label">Velocity</span>
            <span className="param-value text-purple">{orbit.velocity_mps || 3} m/s</span>
          </div>
          <div className="param-item">
            <span className="param-label">Center</span>
            <span className="param-value" style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
              ({orbit.center?.x || 0}, {orbit.center?.y || 0})
            </span>
          </div>
          <div className="param-item">
            <span className="param-label">Target FPS</span>
            <span className="param-value text-orange">{vision.target_fps || 18}</span>
          </div>
          <div className="param-item">
            <span className="param-label">Log Cooldown</span>
            <span className="param-value" style={{ color: 'var(--text-primary)' }}>
              {vision.log_cooldown_s || 2}s
            </span>
          </div>
          <div className="param-item">
            <span className="param-label">Camera</span>
            <span className="param-value" style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
              {vision.camera || 'Camera 0'}
            </span>
          </div>
          <div className="param-item">
            <span className="param-label">Conf Threshold</span>
            <span className="param-value text-accent">{missionData?.conf_threshold || 0.3}</span>
          </div>
        </div>
      </div>

      {/* Software Stack */}
      <div className="card" style={{ marginTop: '20px' }}>
        <div className="card-header">
          <div className="card-title"><span>🖥️</span> Software Stack</div>
        </div>
        <div className="data-table-wrapper">
          <table className="data-table" id="software-stack-table">
            <thead>
              <tr>
                <th>Component</th>
                <th>Technology</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['Simulation', 'Unreal Engine 4.27 + AirSim'],
                ['Language', 'Python 3.10'],
                ['Object Detection', 'Ultralytics YOLO26m'],
                ['Image Processing', 'OpenCV'],
                ['Deep Learning', 'PyTorch + CUDA'],
                ['Database', 'SQLite3'],
                ['Web API', 'FastAPI'],
                ['Frontend', 'React 19'],
              ].map(([comp, tech], i) => (
                <tr key={i}>
                  <td style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{comp}</td>
                  <td className="mono" style={{ fontSize: '12px' }}>{tech}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default MissionControl;
