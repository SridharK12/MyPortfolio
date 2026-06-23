import { useState, useRef } from 'react'
import StatusBadge from '../components/StatusBadge'
import Spinner from '../components/Spinner'
import CreateModal from '../components/CreateModal'
import { api } from '../api'

const fmt  = n => new Intl.NumberFormat('en-IN', { style:'currency', currency:'INR', minimumFractionDigits:2 }).format(n)
const fmtD = s => { try { return new Date(s).toLocaleString('en-IN', { day:'2-digit', month:'short', hour:'2-digit', minute:'2-digit' }) } catch { return '—' } }

export default function Payments({ payments, loading, total, page, setPage, filterStatus, setFS, filterAccount, setFA, onRefresh, showToast }) {
  const [showCreate, setShowCreate] = useState(false)
  const [search, setSearch] = useState(filterAccount)
  const inputRef = useRef()

  async function handleAuthorize(id, status) {
    try { await api.authorize(id, status); showToast(`Payment ${status.toLowerCase()}`); onRefresh() }
    catch(e) { alert('Error: ' + e.message) }
  }

  async function handleCancel(id) {
    if (!confirm('Cancel this payment? This cannot be undone.')) return
    try { await api.cancel(id); showToast('Payment cancelled'); onRefresh() }
    catch(e) { alert('Error: ' + e.message) }
  }

  const TABS = ['ALL','PENDING','APPROVED','REJECTED','CANCELLED']

  return (
    <div style={{ animation:'fadeIn 0.3s ease' }}>
      <div style={{ display:'flex', alignItems:'flex-start', justifyContent:'space-between', marginBottom:24 }}>
        <div>
          <h1 style={{ fontSize:22, fontWeight:700, color:'var(--text)' }}>Payments</h1>
          <p style={{ color:'var(--muted)', fontSize:13, marginTop:3 }}>{total} total records</p>
        </div>
        <button className="btn btn-primary" onClick={() => setShowCreate(true)}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
          New Payment
        </button>
      </div>

      {/* Filter bar */}
      <div className="card" style={{ padding:'16px 20px', marginBottom:20 }}>
        <div style={{ display:'flex', alignItems:'center', gap:16, flexWrap:'wrap' }}>

          <div className="tabs">
            {TABS.map(t => (
              <button key={t} className={`tab ${filterStatus===(t==='ALL'?'':t)?'active':''}`}
                onClick={() => { setFS(t==='ALL'?'':t); setPage(0) }}>
                {t}
              </button>
            ))}
          </div>

          <div className="search-wrap" style={{ flex:1, minWidth:180, maxWidth:280 }}>
            <span className="search-icon">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            </span>
            <input ref={inputRef} placeholder="Search by account…" value={search}
              onChange={e => setSearch(e.target.value)}
              onKeyDown={e => { if(e.key==='Enter'){ setFA(search); setPage(0); onRefresh() } }}
              style={{ width:'100%' }}/>
          </div>

          <button className="btn btn-sm" onClick={() => { setFA(search); setPage(0); onRefresh() }}>Search</button>

          {(filterStatus || filterAccount) && (
            <button className="btn btn-sm" style={{ color:'var(--red)', borderColor:'#FECACA' }}
              onClick={() => { setFS(''); setFA(''); setSearch(''); setPage(0) }}>
              Clear filters
            </button>
          )}
        </div>
      </div>

      {/* Table */}
      <div className="card table-wrap">
        <table>
          <thead>
            <tr>
              <th style={{ width:60 }}>#</th>
              <th style={{ width:'14%' }}>From Account</th>
              <th style={{ width:'14%' }}>To Account</th>
              <th style={{ width:140 }}>Amount</th>
              <th style={{ width:120 }}>Status</th>
              <th>Remarks</th>
              <th style={{ width:130 }}>Created</th>
              <th style={{ width:120 }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr><td colSpan={8} className="empty">
                <Spinner size={20}/> <span style={{ marginLeft:8, verticalAlign:'middle' }}>Loading payments…</span>
              </td></tr>
            )}
            {!loading && payments.length === 0 && (
              <tr><td colSpan={8} className="empty">
                <div className="empty-icon" style={{ fontSize:36 }}>📭</div>
                No payments found.{filterStatus || filterAccount ? ' Try adjusting your filters.' : ' Create one to get started.'}
              </td></tr>
            )}
            {!loading && payments.map(p => (
              <tr key={p.paymentId}>
                <td style={{ fontFamily:'monospace', fontSize:12, color:'var(--muted)' }}>#{p.paymentId}</td>
                <td style={{ fontFamily:'monospace', fontSize:12, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }} title={p.fromAccount}>{p.fromAccount}</td>
                <td style={{ fontFamily:'monospace', fontSize:12, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }} title={p.toAccount}>{p.toAccount}</td>
                <td style={{ fontFamily:'monospace', fontSize:13, fontWeight:700 }}>{fmt(p.amount)}</td>
                <td><StatusBadge status={p.status}/></td>
                <td style={{ fontSize:12, color:'var(--muted)', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }} title={p.remarks}>{p.remarks || '—'}</td>
                <td style={{ fontSize:11, color:'var(--muted)' }}>{fmtD(p.createdAt)}</td>
                <td>
                  {p.status === 'PENDING' ? (
                    <div style={{ display:'flex', gap:4 }}>
                      <button className="btn btn-sm btn-success" title="Approve" onClick={() => handleAuthorize(p.paymentId, 'APPROVED')}>
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><polyline points="20 6 9 17 4 12"/></svg>
                        Approve
                      </button>
                      <button className="btn btn-sm btn-danger" title="Reject" onClick={() => handleAuthorize(p.paymentId, 'REJECTED')}>
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                        Reject
                      </button>
                      <button className="btn btn-sm btn-ghost" title="Cancel" style={{ color:'var(--muted)' }} onClick={() => handleCancel(p.paymentId)}>
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="12" r="10"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/></svg>
                      </button>
                    </div>
                  ) : <span style={{ fontSize:12, color:'#CBD5E1' }}>—</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {/* Pagination */}
        {total > 20 && (
          <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', padding:'14px 20px', borderTop:'1px solid var(--border)' }}>
            <span style={{ fontSize:12, color:'var(--muted)' }}>
              Showing {page*20+1}–{Math.min((page+1)*20, total)} of {total}
            </span>
            <div style={{ display:'flex', gap:6 }}>
              <button className="btn btn-sm" disabled={page===0} onClick={() => setPage(p => p-1)}>← Prev</button>
              <span style={{ padding:'6px 12px', fontSize:12, color:'var(--muted)' }}>Page {page+1} of {Math.ceil(total/20)}</span>
              <button className="btn btn-sm" disabled={(page+1)*20>=total} onClick={() => setPage(p => p+1)}>Next →</button>
            </div>
          </div>
        )}
      </div>

      {showCreate && (
        <CreateModal
          onClose={() => setShowCreate(false)}
          onCreated={() => { setShowCreate(false); showToast('Payment created successfully'); onRefresh() }}
        />
      )}
    </div>
  )
}
