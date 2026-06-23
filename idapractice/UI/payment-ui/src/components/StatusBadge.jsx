const STATUS = {
  PENDING:   { bg:'#FEF3C7', color:'#92400E', border:'#FCD34D', dot:true  },
  APPROVED:  { bg:'#D1FAE5', color:'#065F46', border:'#6EE7B7', dot:false },
  REJECTED:  { bg:'#FEE2E2', color:'#991B1B', border:'#FCA5A5', dot:false },
  CANCELLED: { bg:'#F1F5F9', color:'#475569', border:'#CBD5E1', dot:false },
}
export default function StatusBadge({ status }) {
  const s = STATUS[status] || STATUS.PENDING
  return (
    <span className="badge" style={{ background:s.bg, color:s.color, border:`1px solid ${s.border}` }}>
      {s.dot && <span className="badge-dot" style={{ background:s.border }}/>}
      {status}
    </span>
  )
}
