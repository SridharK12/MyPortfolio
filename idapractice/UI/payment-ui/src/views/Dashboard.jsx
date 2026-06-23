import { useMemo } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, PieChart, Pie, Legend } from 'recharts'
import StatusBadge from '../components/StatusBadge'

const fmt   = n => new Intl.NumberFormat('en-IN', { style:'currency', currency:'INR', maximumFractionDigits:0 }).format(n)
const fmtD  = s => { try { return new Date(s).toLocaleString('en-IN', { day:'2-digit', month:'short', hour:'2-digit', minute:'2-digit' }) } catch { return '—' } }

const COLORS = { PENDING:'#F59E0B', APPROVED:'#10B981', REJECTED:'#EF4444', CANCELLED:'#94A3B8' }

export default function Dashboard({ payments, onNav }) {
  const stats = useMemo(() => {
    const total  = payments.length
    const volume = payments.reduce((s, p) => s + p.amount, 0)
    const pending = payments.filter(p => p.status === 'PENDING').length
    const approved = payments.filter(p => p.status === 'APPROVED').length
    const rejected = payments.filter(p => p.status === 'REJECTED').length
    const cancelled = payments.filter(p => p.status === 'CANCELLED').length
    const rate = total > 0 ? Math.round((approved / total) * 100) : 0
    return { total, volume, pending, approved, rejected, cancelled, rate }
  }, [payments])

  const pieData = [
    { name:'Approved',  value: stats.approved,  color:'#10B981' },
    { name:'Pending',   value: stats.pending,   color:'#F59E0B' },
    { name:'Rejected',  value: stats.rejected,  color:'#EF4444' },
    { name:'Cancelled', value: stats.cancelled, color:'#94A3B8' },
  ].filter(d => d.value > 0)

  const barData = useMemo(() => {
    const groups = {}
    payments.forEach(p => {
      const d = p.createdAt ? new Date(p.createdAt).toLocaleDateString('en-IN', { day:'2-digit', month:'short' }) : 'Unknown'
      if (!groups[d]) groups[d] = { date:d, APPROVED:0, PENDING:0, REJECTED:0 }
      if (groups[d][p.status] !== undefined) groups[d][p.status] += p.amount
    })
    return Object.values(groups).slice(-7)
  }, [payments])

  const recent = [...payments].sort((a,b) => new Date(b.createdAt) - new Date(a.createdAt)).slice(0,5)

  return (
    <div style={{ animation:'fadeIn 0.3s ease' }}>
      <div style={{ marginBottom:28 }}>
        <h1 style={{ fontSize:22, fontWeight:700, color:'var(--text)' }}>Overview</h1>
        <p style={{ color:'var(--muted)', fontSize:13, marginTop:3 }}>Your payment activity at a glance.</p>
      </div>

      {/* Stat cards */}
      <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:16, marginBottom:24 }}>
        {[
          { label:'Total Volume',    value:fmt(stats.volume),   icon:'💳', bg:'#EFF6FF', ic:'#1D4ED8' },
          { label:'Total Payments',  value:stats.total,         icon:'📋', bg:'#F5F3FF', ic:'#7C3AED' },
          { label:'Pending Review',  value:stats.pending,       icon:'⏳', bg:'#FFFBEB', ic:'#D97706' },
          { label:'Approval Rate',   value:stats.rate+'%',      icon:'✅', bg:'#ECFDF5', ic:'#059669' },
        ].map(s => (
          <div key={s.label} className="card stat-card">
            <div className="stat-icon" style={{ background:s.bg }}>
              <span style={{ fontSize:18 }}>{s.icon}</span>
            </div>
            <div className="stat-label">{s.label}</div>
            <div className="stat-value">{s.value}</div>
          </div>
        ))}
      </div>

      <div style={{ display:'grid', gridTemplateColumns:'1fr 360px', gap:20, marginBottom:24 }}>

        {/* Bar chart */}
        <div className="card" style={{ padding:'20px' }}>
          <div className="card-header" style={{ padding:0, marginBottom:16 }}>
            <div>
              <div className="card-title">Payment Volume by Day</div>
              <div className="card-sub">Last 7 days activity</div>
            </div>
          </div>
          {barData.length === 0
            ? <div className="empty"><div className="empty-icon" style={{ fontSize:32 }}>📊</div>No data yet</div>
            : <ResponsiveContainer width="100%" height={220}>
                <BarChart data={barData} margin={{ left:-10, bottom:0 }}>
                  <XAxis dataKey="date" tick={{ fontSize:11, fill:'#94A3B8' }} axisLine={false} tickLine={false}/>
                  <YAxis tick={{ fontSize:11, fill:'#94A3B8' }} axisLine={false} tickLine={false} tickFormatter={v => '₹'+new Intl.NumberFormat('en-IN',{notation:'compact'}).format(v)}/>
                  <Tooltip formatter={(v,n) => [fmt(v), n]} contentStyle={{ borderRadius:8, border:'1px solid #E2E8F0', fontSize:12 }}/>
                  <Bar dataKey="APPROVED" fill="#10B981" radius={[4,4,0,0]} stackId="a"/>
                  <Bar dataKey="PENDING"  fill="#F59E0B" radius={[4,4,0,0]} stackId="a"/>
                  <Bar dataKey="REJECTED" fill="#EF4444" radius={[4,4,0,0]} stackId="a"/>
                </BarChart>
              </ResponsiveContainer>
          }
        </div>

        {/* Pie chart */}
        <div className="card" style={{ padding:'20px' }}>
          <div className="card-title" style={{ marginBottom:4 }}>Status Distribution</div>
          <div className="card-sub" style={{ marginBottom:16 }}>By count</div>
          {pieData.length === 0
            ? <div className="empty"><div className="empty-icon" style={{ fontSize:32 }}>🥧</div>No data yet</div>
            : <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={pieData} cx="50%" cy="50%" innerRadius={55} outerRadius={85} paddingAngle={3} dataKey="value">
                    {pieData.map((e,i) => <Cell key={i} fill={e.color}/>)}
                  </Pie>
                  <Tooltip contentStyle={{ borderRadius:8, border:'1px solid #E2E8F0', fontSize:12 }}/>
                  <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize:12 }}/>
                </PieChart>
              </ResponsiveContainer>
          }
        </div>
      </div>

      {/* Recent payments */}
      <div className="card">
        <div className="card-header">
          <div>
            <div className="card-title">Recent Payments</div>
            <div className="card-sub">Latest 5 transactions</div>
          </div>
          <button className="btn btn-sm" onClick={() => onNav('payments')}>View all →</button>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th style={{ width:60  }}>#</th>
                <th style={{ width:'18%' }}>From</th>
                <th style={{ width:'18%' }}>To</th>
                <th style={{ width:130 }}>Amount</th>
                <th style={{ width:115 }}>Status</th>
                <th style={{ width:130 }}>Created</th>
              </tr>
            </thead>
            <tbody>
              {recent.length === 0
                ? <tr><td colSpan={6} className="empty">No payments yet</td></tr>
                : recent.map(p => (
                  <tr key={p.paymentId}>
                    <td style={{ fontFamily:'monospace', fontSize:12, color:'var(--muted)' }}>{p.paymentId}</td>
                    <td style={{ fontFamily:'monospace', fontSize:12 }}>{p.fromAccount}</td>
                    <td style={{ fontFamily:'monospace', fontSize:12 }}>{p.toAccount}</td>
                    <td style={{ fontFamily:'monospace', fontWeight:600, fontSize:13 }}>{fmt(p.amount)}</td>
                    <td><StatusBadge status={p.status}/></td>
                    <td style={{ fontSize:11, color:'var(--muted)' }}>{fmtD(p.createdAt)}</td>
                  </tr>
                ))
              }
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
