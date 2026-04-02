import React, { useState, useEffect } from 'react';
import './App.css';
import './components/components.css';
import Dashboard from './components/Dashboard';
import DetectionLog from './components/DetectionLog';
import ImageDetect from './components/ImageDetect';
import TrainingMetrics from './components/TrainingMetrics';
import MissionControl from './components/MissionControl';

const PAGES = [
  { id: 'dashboard',  label: 'Dashboard',        icon: '📊', section: 'Overview' },
  { id: 'detections', label: 'Detection Log',     icon: '🔍', section: 'Overview' },
  { id: 'upload',     label: 'Upload & Detect',   icon: '📤', section: 'Tools' },
  { id: 'training',   label: 'Training Metrics',  icon: '📈', section: 'Tools' },
  { id: 'mission',    label: 'Mission Control',   icon: '🛸', section: 'Mission' },
];

function App() {
  const [activePage, setActivePage] = useState('dashboard');
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('infraspect-theme') || 'dark';
  });

  // Apply theme to document root
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('infraspect-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  const renderPage = () => {
    switch (activePage) {
      case 'dashboard':  return <Dashboard onNavigate={setActivePage} />;
      case 'detections': return <DetectionLog />;
      case 'upload':     return <ImageDetect />;
      case 'training':   return <TrainingMetrics />;
      case 'mission':    return <MissionControl />;
      default:           return <Dashboard onNavigate={setActivePage} />;
    }
  };

  const currentPage = PAGES.find(p => p.id === activePage) || PAGES[0];

  // Group pages by section
  const sections = {};
  PAGES.forEach(p => {
    if (!sections[p.section]) sections[p.section] = [];
    sections[p.section].push(p);
  });

  return (
    <div className="app-layout" id="app-root">
      {/* ── Sidebar ── */}
      <nav className="sidebar" id="sidebar">
        <div className="sidebar-brand">
          <div className="sidebar-brand-icon">
            <div className="brand-dot">IS</div>
            <div className="brand-text">
              <h1>Infraspect</h1>
              <span>Crack Detection System</span>
            </div>
          </div>
        </div>

        <div className="sidebar-nav">
          {Object.entries(sections).map(([section, pages]) => (
            <React.Fragment key={section}>
              <div className="nav-section-label">{section}</div>
              {pages.map(page => (
                <button
                  key={page.id}
                  id={`nav-${page.id}`}
                  className={`nav-item ${activePage === page.id ? 'active' : ''}`}
                  onClick={() => setActivePage(page.id)}
                >
                  <span className="nav-icon">{page.icon}</span>
                  {page.label}
                </button>
              ))}
            </React.Fragment>
          ))}
        </div>

        <div className="sidebar-footer">
          <div className="sidebar-status">
            <div className="status-dot"></div>
            <span className="status-text">System Online</span>
          </div>
        </div>
      </nav>

      {/* ── Main content ── */}
      <main className="main-area">
        <header className="header" id="header">
          <div className="header-title">
            <span className="page-icon">{currentPage.icon}</span>
            <h2>{currentPage.label}</h2>
          </div>
          <div className="header-right">
            <div className="header-badge">
              <span className="header-badge-dot"></span>
              YOLO26m
            </div>
            <button
              className="theme-toggle"
              onClick={toggleTheme}
              id="theme-toggle-btn"
              title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
              aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            >
              <span className="theme-toggle-icon">
                {theme === 'dark' ? '☀️' : '🌙'}
              </span>
            </button>
          </div>
        </header>

        <div className="content" key={activePage}>
          {renderPage()}
        </div>
      </main>
    </div>
  );
}

export default App;
