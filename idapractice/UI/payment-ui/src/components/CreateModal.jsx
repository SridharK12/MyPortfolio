import { useState } from 'react'
import { api } from '../api'
import Spinner from './Spinner'

export default function CreateModal({ onClose, onCreated }) {
  const [form, setForm] = useState({ fromAccount:'', toAccount:'', amount:'', remarks:'' })
  const [saving, setSaving] = useState(false)
  const set = k => e => setForm(f => ({ ...f, [k]: e.target.value }))

  async function submit() {
    if (!form.fromAccount || !form.toAccount || !form.amount) { alert('Please fill all required fields'); return }
    if (isNaN(parseFloat(form.amount)) || parseFloat(form.amount) <= 0) { alert('Enter a valid amount'); return }
    setSaving(true)
    try {
      await api.createPayment({ fromAccount:form.fromAccount.trim(), toAccount:form.toAccount.trim(), amount:parseFloat(form.amount), remarks:form.remarks.trim() })
      onCreated()
    } catch(e) { alert('Error: ' + e.message) }
    finally { setSaving(false) }
  }

  return (
    <div className="modal-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal">
        <div style={{ display:'flex', alignItems:'center', gap:14, marginBottom:20 }}>
          <div style={{ width:42, height:42, background:'#EFF6FF', borderRadius:10, display:'flex', alignItems:'center', justifyContent:'center' }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#1D4ED8" strokeWidth="2" strokeLinecap="round"><rect x="2" y="5" width="20" height="14" rx="2"/><line x1="2" y1="10" x2="22" y2="10"/></svg>
          </div>
          <div>
            <div className="modal-title" style={{ marginBottom:0 }}>New Payment</div>
            <div className="modal-sub" style={{ marginBottom:0 }}>Initiate a new payment transaction</div>
          </div>
          <button className="btn btn-ghost btn-sm" style={{ marginLeft:'auto' }} onClick={onClose}>✕</button>
        </div>
        <div style={{ height:1, background:'var(--border)', margin:'0 0 20px' }}/>
        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:14 }}>
          <div className="field"><label>From Account *</label><input placeholder="e.g. HDFC-001" value={form.fromAccount} onChange={set('fromAccount')}/></div>
          <div className="field"><label>To Account *</label><input placeholder="e.g. ICICI-002" value={form.toAccount} onChange={set('toAccount')}/></div>
        </div>
        <div className="field"><label>Amount (₹) *</label><input type="number" min="0.01" step="0.01" placeholder="e.g. 10000.00" value={form.amount} onChange={set('amount')}/></div>
        <div className="field"><label>Remarks</label><input placeholder="Optional description or reference" value={form.remarks} onChange={set('remarks')}/></div>
        <div style={{ display:'flex', gap:10, justifyContent:'flex-end', marginTop:24 }}>
          <button className="btn" onClick={onClose}>Cancel</button>
          <button className="btn btn-primary" onClick={submit} disabled={saving} style={{ minWidth:130 }}>
            {saving ? <><Spinner size={14}/>&nbsp;Creating…</> : 'Create Payment'}
          </button>
        </div>
      </div>
    </div>
  )
}
