import { useState, useEffect, useCallback } from 'react'
import Sidebar from './components/Sidebar'
import Spinner from './components/Spinner'
import Dashboard from './views/Dashboard'
import Payments  from './views/Payments'
import Analytics from './views/Analytics'
import { api } from './api'

const PAGE_TITLES = {
  dashboard: ['Payment Platform', 'Dashboard'],
  payments:  ['Payment Platform', 'Payments'],
  analytics: ['Payment Platform', 'Analytics'],
}

export default function App() {
  const [page,          setPage]         = useState('dashboard')
  const [payments,      setPayments]     = useState([])
  const [total,         setTotal]        = useState(0)
  const [loading,       setLoading]      = useState(false)
  const [backendOk,     setBackendOk]    = useState(null)
  const [filterStatus,  setFS]           = useState('')
  const [filterAccount, setFA]           = useState('')
  const [tablePage,     setTablePage]    = useState(0)
  const [toast,         setToast]        = useState(null)

  const showToast = msg => { setToast(msg); setTimeout(() => setToast(null), 3000) }

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const data = await api.getPayments({ status:filterStatus, fromAccount:filterAccount, page:tablePage })
      const rows = data.content ?? data
      setPayments(rows)
      setTotal(data.totalElements ?? rows.length)
      setBackendOk(true)
    } catch {
      setBackendOk(false)
    } finally {
      setLoading(false)
    }
  }, [filterStatus, filterAccount, tablePage])

  useEffect(() => { load() }, [load])

  const [breadcrumb, pageTitle] = PAGE_TITLES[page] || ['', page]

  return (
    <>
      <Sidebar active={page} onNav={p => setPage(p)}/>
      <div className="main">
        <div className="topbar">
          <div className="topbar-left">
            <div className="breadcrumb">{breadcrumb}</div>
            <div className="page-title">{pageTitle}</div>
          </div>
          <div className="topbar-right">
            {backendOk === false && (
              <span style={{ fontSize:12, background:'#FEF3C7', color:'#92400E', padding:'5px 12px', borderRadius:20, border:'1px solid #FCD34D' }}>
                ⚠ Backend unreachable — add @CrossOrigin to PaymentController
              </span>
            )}
            {backendOk === true && (
              <span style={{ fontSize:12, background:'#ECFDF5', color:'#065F46', padding:'5px 12px', borderRadius:20, border:'1px solid #6EE7B7' }}>
                ● Connected to localhost:8080
              </span>
            )}
            <button className="btn btn-sm" onClick={load} disabled={loading}>
              {loading ? <Spinner size={13}/> : <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/></svg>}
              Refresh
            </button>
          </div>
        </div>
        <div className="content">
          {page === 'dashboard' && <Dashboard payments={payments} onNav={setPage}/>}
          {page === 'payments'  && <Payments payments={payments} loading={loading} total={total} page={tablePage} setPage={setTablePage} filterStatus={filterStatus} setFS={setFS} filterAccount={filterAccount} setFA={setFA} onRefresh={load} showToast={showToast}/>}
          {page === 'analytics' && <Analytics payments={payments}/>}
        </div>
      </div>
      {toast && (
        <div className="toast">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#4ADE80" strokeWidth="2.5" strokeLinecap="round"><polyline points="20 6 9 17 4 12"/></svg>
          {toast}
        </div>
      )}
    </>
  )
}
