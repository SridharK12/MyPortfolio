const NAV = [
  { id:'dashboard', label:'Dashboard',  icon:<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg> },
  { id:'payments',  label:'Payments',   icon:<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><rect x="2" y="5" width="20" height="14" rx="2"/><line x1="2" y1="10" x2="22" y2="10"/></svg> },
  { id:'analytics', label:'Analytics',  icon:<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg> },
]

export default function Sidebar({ active, onNav }) {
  return (
    <div className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round">
            <rect x="2" y="5" width="20" height="14" rx="2"/>
            <line x1="2" y1="10" x2="22" y2="10"/>
          </svg>
        </div>
        <div>
          <div className="logo-text">IDA Practice</div>
          <div className="logo-sub">Payment Platform</div>
        </div>
      </div>

      <div className="nav-section">
        <div className="nav-label">Menu</div>
        {NAV.map(n => (
          <button key={n.id} className={`nav-item ${active===n.id?'active':''}`} onClick={() => onNav(n.id)}>
            {n.icon} {n.label}
          </button>
        ))}
      </div>

      <div style={{ padding:'18px 12px 6px' }}>
        <div className="nav-label">System</div>
        <button className="nav-item">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/><path d="M4.93 4.93a10 10 0 0 0 0 14.14"/></svg>
          Settings
        </button>
        <button className="nav-item">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
          API Docs
        </button>
      </div>

      <div className="sidebar-footer">
        <div className="avatar">SK</div>
        <div>
          <div className="footer-name">Sridhar K.</div>
          <div className="footer-role">Admin</div>
        </div>
      </div>
    </div>
  )
}
