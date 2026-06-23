import { useMemo } from 'react'
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts'

const fmt = n => new Intl.NumberFormat('en-IN', { style:'currency', currency:'INR', maximumFractionDigits:0 }).format(n)

export default function Analytics({ payments }) {
  const byDay = useMemo(() => {
    const map = {}
    payments.forEach(p => {
      const d = p.createdAt ? new Date(p.createdAt).toLocaleDateString('en-IN', { day:'2-digit', month:'short' }) : 'N/A'
      if (!map[d]) map[d] = { date:d, volume:0, count:0 }
      map[d].volume += p.amount
      map[d].count++
    })
    return Object.values(map).slice(-14)
  }, [payments])

  const byStatus = useMemo(() => {
    const map = {}
    payments.forEach(p => { map[p.status] = (map[p.status]||0) + p.amount })
    return Object.entries(map).map(([status, volume]) => ({ status, volume }))
  }, [payments])

  const topAccounts = useMemo(() => {
    const map = {}
    payments.forEach(p => { map[p.fromAccount] = (map[p.fromAccount]||0) + p.amount })
    return Object.entries(map).sort((a,b) => b[1]-a[1]).slice(0,5).map(([acc, vol]) => ({ acc, vol }))
  }, [payments])

  const totalVol = payments.reduce((s,p) => s+p.amount, 0)

  const STATUS_COLORS = { PENDING:'#F59E0B', APPROVED:'#10B981', REJECTED:'#EF4444', CANCELLED:'#94A3B8' }

  return (
    <div style={{ animation:'fadeIn 0.3s ease' }}>
      <div style={{ marginBottom:28 }}>
        <h1 style={{ fontSize:22, fontWeight:700 }}>Analytics</h1>
        <p style={{ color:'var(--muted)', fontSize:13, marginTop:3 }}>Payment trends and insights.</p>
      </div>

      {/* Summary row */}
      <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:16, marginBottom:24 }}>
        {[
          { label:'Total volume processed', value:fmt(totalVol),         bg:'#EFF6FF', color:'#1D4ED8' },
          { label:'Average payment size',   value:fmt(totalVol / (payments.length||1)), bg:'#F5F3FF', color:'#7C3AED' },
          { label:'Payments this session',  value:payments.length,        bg:'#ECFDF5', color:'#059669' },
        ].map(s => (
          <div key={s.label} className="card" style={{ padding:'20px', borderLeft:`4px solid ${s.color}` }}>
            <div style={{ fontSize:12, color:'var(--muted)', marginBottom:8 }}>{s.label}</div>
            <div style={{ fontSize:24, fontWeight:700, color:s.color }}>{s.value}</div>
          </div>
        ))}
      </div>

      {/* Area chart */}
      <div className="card" style={{ padding:20, marginBottom:20 }}>
        <div className="card-title" style={{ marginBottom:4 }}>Volume Trend</div>
        <div className="card-sub" style={{ marginBottom:20 }}>Daily payment volume</div>
        {byDay.length === 0
          ? <div className="empty"><div style={{ fontSize:32 }}>📈</div><p style={{ marginTop:8 }}>No data yet</p></div>
          : <ResponsiveContainer width="100%" height={240}>
              <AreaChart data={byDay}>
                <defs>
                  <linearGradient id="volGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#1D4ED8" stopOpacity={0.15}/>
                    <stop offset="95%" stopColor="#1D4ED8" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="date" tick={{ fontSize:11, fill:'#94A3B8' }} axisLine={false} tickLine={false}/>
                <YAxis tick={{ fontSize:11, fill:'#94A3B8' }} axisLine={false} tickLine={false} tickFormatter={v => '₹'+new Intl.NumberFormat('en-IN',{notation:'compact'}).format(v)}/>
                <Tooltip formatter={v => fmt(v)} contentStyle={{ borderRadius:8, border:'1px solid #E2E8F0', fontSize:12 }}/>
                <Area type="monotone" dataKey="volume" stroke="#1D4ED8" strokeWidth={2} fill="url(#volGrad)"/>
              </AreaChart>
            </ResponsiveContainer>
        }
      </div>

      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:20 }}>
        {/* Volume by status */}
        <div className="card" style={{ padding:20 }}>
          <div className="card-title" style={{ marginBottom:4 }}>Volume by Status</div>
          <div className="card-sub" style={{ marginBottom:16 }}>Total amount per status</div>
          {byStatus.length === 0
            ? <div className="empty">No data</div>
            : <ResponsiveContainer width="100%" height={200}>
                <BarChart data={byStatus} layout="vertical" margin={{ left:10 }}>
                  <XAxis type="number" tick={{ fontSize:11, fill:'#94A3B8' }} axisLine={false} tickLine={false} tickFormatter={v => '₹'+new Intl.NumberFormat('en-IN',{notation:'compact'}).format(v)}/>
                  <YAxis type="category" dataKey="status" tick={{ fontSize:11, fill:'#64748B' }} axisLine={false} tickLine={false} width={75}/>
                  <Tooltip formatter={v => fmt(v)} contentStyle={{ borderRadius:8, border:'1px solid #E2E8F0', fontSize:12 }}/>
                  <Bar dataKey="volume" radius={[0,4,4,0]}>
                    {byStatus.map((e,i) => <Cell key={i} fill={STATUS_COLORS[e.status]||'#94A3B8'}/>)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
          }
        </div>

        {/* Top senders */}
        <div className="card" style={{ padding:20 }}>
          <div className="card-title" style={{ marginBottom:4 }}>Top Sending Accounts</div>
          <div className="card-sub" style={{ marginBottom:16 }}>By total volume sent</div>
          {topAccounts.length === 0
            ? <div className="empty">No data</div>
            : <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
                {topAccounts.map((a, i) => {
                  const pct = totalVol > 0 ? Math.round((a.vol / totalVol) * 100) : 0
                  return (
                    <div key={a.acc}>
                      <div style={{ display:'flex', justifyContent:'space-between', marginBottom:5 }}>
                        <span style={{ fontSize:12, fontFamily:'monospace', color:'var(--text)' }}>{a.acc}</span>
                        <span style={{ fontSize:12, color:'var(--muted)' }}>{fmt(a.vol)}</span>
                      </div>
                      <div style={{ height:5, background:'#F1F5F9', borderRadius:3 }}>
                        <div style={{ height:'100%', width:`${pct}%`, background:'#1D4ED8', borderRadius:3, transition:'width 0.5s ease' }}/>
                      </div>
                    </div>
                  )
                })}
              </div>
          }
        </div>
      </div>
    </div>
  )
}
