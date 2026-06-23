// All requests go to /v1/... which Vite proxies to http://localhost:8080/v1/...

export const api = {
  async getPayments({ status, fromAccount, page = 0, size = 20 }) {
    const q = new URLSearchParams({ page, size })
    if (status)      q.set('status', status)
    if (fromAccount) q.set('fromAccount', fromAccount)
    const r = await fetch(`/v1/payments?${q}`)
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    return r.json()
  },

  async createPayment(body) {
    const r = await fetch('/v1/payments', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!r.ok) throw new Error(await r.text())
    return r.json()
  },

  async authorize(id, status, remarks = '') {
    const r = await fetch(`/v1/payments/${id}/authorization`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status, remarks }),
    })
    if (!r.ok) throw new Error(await r.text())
    return r.json()
  },

  async cancel(id) {
    const r = await fetch(`/v1/payments/${id}`, { method: 'DELETE' })
    if (!r.ok) throw new Error(await r.text())
  },
}
