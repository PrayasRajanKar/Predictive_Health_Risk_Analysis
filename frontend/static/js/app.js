// ─── Config ──────────────────────────────────────────────────────────────────
const API = '/api';

// ─── Navigation ──────────────────────────────────────────────────────────────
function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById(`page-${name}`).classList.add('active');
  document.querySelector(`[data-page="${name}"]`).classList.add('active');

  if (name === 'dashboard') loadDashboard();
  if (name === 'patients') loadPatients();
}

// ─── Date display ─────────────────────────────────────────────────────────────
document.getElementById('dash-date').textContent = new Date().toLocaleDateString('en-IN', {
  weekday: 'short', year: 'numeric', month: 'short', day: 'numeric'
});

// ─── API helpers ─────────────────────────────────────────────────────────────
async function apiFetch(path, options = {}) {
  try {
    const res = await fetch(`${API}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options
    });
    return await res.json();
  } catch (e) {
    console.error('API error:', e);
    return { success: false, message: e.message };
  }
}

// ─── Dashboard ────────────────────────────────────────────────────────────────
async function loadDashboard() {
  const data = await apiFetch('/dashboard/stats');
  if (!data.success) return;

  const d = data.data;
  document.getElementById('stat-patients').textContent = d.total_patients;
  document.getElementById('stat-assessments').textContent = d.total_assessments;

  const highRisk = (d.risk_distribution.High || 0) + (d.risk_distribution.Critical || 0);
  document.getElementById('stat-high-risk').textContent = highRisk;
  document.getElementById('stat-avg-risk').textContent = d.average_risk_score + '%';

  // Risk bars
  const total = d.total_assessments || 1;
  ['Low', 'Moderate', 'High', 'Critical'].forEach(cat => {
    const count = d.risk_distribution[cat] || 0;
    const pct = Math.round((count / total) * 100);
    document.getElementById(`bar-${cat.toLowerCase()}`).style.width = pct + '%';
    document.getElementById(`count-${cat.toLowerCase()}`).textContent = count;
  });

  // Disease rates
  const drEl = document.getElementById('disease-rates');
  const rates = d.disease_rates;
  if (Object.values(rates).some(v => v > 0)) {
    const diseaseLabels = {
      diabetes: 'Diabetes', heart_disease: 'Heart Disease',
      hypertension: 'Hypertension', stroke: 'Stroke', obesity: 'Obesity'
    };
    drEl.innerHTML = Object.entries(rates).map(([disease, pct]) => `
      <div class="disease-rate-item">
        <div class="disease-rate-label">
          <span class="disease-rate-name">${diseaseLabels[disease] || disease}</span>
          <span class="disease-rate-pct" style="color:${rateColor(pct)}">${pct}%</span>
        </div>
        <div class="disease-rate-bar">
          <div class="disease-rate-fill" style="width:${pct}%; background: ${rateGradient(pct)}"></div>
        </div>
      </div>
    `).join('');
  } else {
    drEl.innerHTML = '<div class="empty-state">Run assessments to see disease rates</div>';
  }

  // Recent assessments
  const recentEl = document.getElementById('recent-table');
  if (d.recent_assessments && d.recent_assessments.length > 0) {
    recentEl.innerHTML = `
      <table class="data-table">
        <thead><tr>
          <th>Patient</th><th>Date</th><th>Risk Score</th><th>Category</th>
          <th>Top Risk</th>
        </tr></thead>
        <tbody>${d.recent_assessments.map(a => {
          const topDisease = getTopDisease(a.risk_predictions);
          return `<tr>
            <td>${a.patient_name || a.patient_id}</td>
            <td>${formatDate(a.assessed_at)}</td>
            <td><span style="font-family:Space Mono;font-weight:700;color:${rateColor(a.overall_risk_score)}">${a.overall_risk_score}%</span></td>
            <td><span class="risk-pill ${a.risk_category.toLowerCase()}">${a.risk_category}</span></td>
            <td style="color:var(--text-secondary)">${topDisease}</td>
          </tr>`;
        }).join('')}</tbody>
      </table>`;
  } else {
    recentEl.innerHTML = '<div class="empty-state">No assessments yet — run your first risk assessment!</div>';
  }
}

// ─── Assessment ───────────────────────────────────────────────────────────────
async function runAssessment() {
  const payload = {
    patient_id: document.getElementById('f-patient-id').value.trim() || null,
    age: +document.getElementById('f-age').value,
    bmi: +document.getElementById('f-bmi').value,
    blood_pressure_systolic: +document.getElementById('f-bp-sys').value,
    blood_pressure_diastolic: +document.getElementById('f-bp-dia').value,
    blood_glucose: +document.getElementById('f-glucose').value,
    cholesterol: +document.getElementById('f-cholesterol').value,
    heart_rate: +document.getElementById('f-hr').value,
    sleep_hours: +document.getElementById('f-sleep').value,
    smoking: +document.getElementById('f-smoking').value,
    alcohol_consumption: document.getElementById('f-alcohol').value,
    physical_activity: document.getElementById('f-activity').value,
    diet_quality: document.getElementById('f-diet').value,
    stress_level: +document.getElementById('f-stress').value,
    family_history_diabetes: document.getElementById('fh-diabetes').checked ? 1 : 0,
    family_history_heart: document.getElementById('fh-heart').checked ? 1 : 0,
    family_history_hypertension: document.getElementById('fh-hypertension').checked ? 1 : 0,
    family_history_cancer: document.getElementById('fh-cancer').checked ? 1 : 0,
  };

  document.getElementById('loading-overlay').style.display = 'flex';

  const data = await apiFetch('/assess', {
    method: 'POST',
    body: JSON.stringify(payload)
  });

  document.getElementById('loading-overlay').style.display = 'none';

  if (!data.success) {
    alert('Assessment failed: ' + data.message);
    return;
  }

  renderResults(data.data);
}

function renderResults(result) {
  const { predictions, overall_risk_score, risk_category, recommendations, alerts } = result;

  const resultsCol = document.getElementById('results-col');
  resultsCol.style.display = 'flex';

  // Header
  const colors = { Low: '#00f5a0', Moderate: '#f59e0b', High: '#f97316', Critical: '#ef4444' };
  const color = colors[risk_category] || '#fff';
  const headerEl = document.getElementById('results-risk-header');
  headerEl.className = `results-header ${risk_category.toLowerCase()}`;
  headerEl.innerHTML = `
    <div class="results-score" style="color:${color}">${overall_risk_score}%</div>
    <div class="results-category" style="color:${color}">${risk_category} Risk</div>
    <div class="results-desc">${riskDescription(risk_category)}</div>
  `;

  // Gauges
  const diseaseNames = {
    diabetes: 'Diabetes', heart_disease: 'Heart Disease',
    hypertension: 'Hypertension', stroke: 'Stroke', obesity: 'Obesity'
  };
  document.getElementById('risk-gauges').innerHTML = Object.entries(predictions).map(([d, info]) => {
    const c = levelColor(info.level);
    return `
      <div class="gauge-item">
        <div class="gauge-header">
          <span class="gauge-name">${diseaseNames[d] || d}</span>
          <span class="gauge-score" style="color:${c}">${info.probability}%</span>
        </div>
        <div class="gauge-track">
          <div class="gauge-fill" style="width:${info.probability}%; background: linear-gradient(90deg, ${c}88, ${c})"></div>
        </div>
        <div class="gauge-level" style="color:${c}">${info.level} Risk</div>
      </div>
    `;
  }).join('');

  // Alerts
  const alertsCard = document.getElementById('alerts-card');
  if (alerts && alerts.length > 0) {
    alertsCard.style.display = 'block';
    document.getElementById('risk-alerts').innerHTML = alerts.map(a => `
      <div class="alert-item ${a.severity}">
        <span class="alert-icon">${a.severity === 'Critical' ? '🔴' : '🟠'}</span>
        <span class="alert-text">${a.message}</span>
      </div>
    `).join('');
  } else {
    alertsCard.style.display = 'none';
  }

  // Recommendations
  document.getElementById('recommendations').innerHTML = recommendations.map(r => `
    <div class="rec-item">
      <div class="rec-header">
        <span class="rec-category">${r.category}</span>
        <span class="rec-priority ${r.priority}">${r.priority}</span>
      </div>
      <p class="rec-text">${r.recommendation}</p>
    </div>
  `).join('');

  resultsCol.scrollIntoView({ behavior: 'smooth' });
}

// ─── Patients ─────────────────────────────────────────────────────────────────
async function loadPatients() {
  const data = await apiFetch('/patients');
  if (!data.success) return;

  const wrapper = document.getElementById('patients-table-wrapper');
  if (!data.data.length) {
    wrapper.innerHTML = '<div class="empty-state">No patients registered yet.</div>';
    return;
  }

  wrapper.innerHTML = `
    <table class="data-table">
      <thead><tr>
        <th>Patient ID</th><th>Name</th><th>Age</th><th>Gender</th>
        <th>Assessments</th><th>Latest Risk</th><th>Actions</th>
      </tr></thead>
      <tbody>${data.data.map(p => `
        <tr>
          <td style="font-family:Space Mono;font-size:12px;color:var(--cyan)">${p.patient_id}</td>
          <td style="font-weight:500">${p.name}</td>
          <td>${p.age}</td>
          <td>${p.gender}</td>
          <td style="text-align:center">${p.assessment_count}</td>
          <td>${p.latest_risk_category
            ? `<span class="risk-pill ${p.latest_risk_category.toLowerCase()}">${p.latest_risk_category}</span>`
            : '<span style="color:var(--text-muted);font-size:12px">No assessment</span>'
          }</td>
          <td>
            <button class="btn-sm" onclick="assessPatient('${p.patient_id}')">Assess</button>
          </td>
        </tr>
      `).join('')}</tbody>
    </table>`;
}

function assessPatient(patientId) {
  document.getElementById('f-patient-id').value = patientId;
  showPage('assess');
}

// ─── Register ─────────────────────────────────────────────────────────────────
async function registerPatient() {
  const name = document.getElementById('reg-name').value.trim();
  const age = document.getElementById('reg-age').value;
  const gender = document.getElementById('reg-gender').value;

  if (!name || !age) {
    showRegMsg('Please fill in Name and Age.', 'error');
    return;
  }

  const data = await apiFetch('/patients', {
    method: 'POST',
    body: JSON.stringify({
      name, age: +age, gender,
      email: document.getElementById('reg-email').value,
      phone: document.getElementById('reg-phone').value
    })
  });

  if (data.success) {
    showRegMsg(`✓ Patient registered! ID: ${data.data.patient_id}`, 'success');
    document.getElementById('reg-name').value = '';
    document.getElementById('reg-age').value = '';
    document.getElementById('reg-email').value = '';
    document.getElementById('reg-phone').value = '';
  } else {
    showRegMsg('Error: ' + data.message, 'error');
  }
}

function showRegMsg(text, type) {
  const el = document.getElementById('reg-msg');
  el.textContent = text;
  el.className = `reg-msg ${type}`;
  setTimeout(() => { el.textContent = ''; el.className = 'reg-msg'; }, 5000);
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function levelColor(level) {
  return { Low: '#00f5a0', Moderate: '#f59e0b', High: '#f97316', Critical: '#ef4444' }[level] || '#fff';
}
function rateColor(pct) {
  if (pct < 25) return '#00f5a0';
  if (pct < 50) return '#f59e0b';
  if (pct < 75) return '#f97316';
  return '#ef4444';
}
function rateGradient(pct) {
  if (pct < 25) return 'linear-gradient(90deg, #00f5a0, #00c8e0)';
  if (pct < 50) return 'linear-gradient(90deg, #f59e0b, #f97316)';
  if (pct < 75) return 'linear-gradient(90deg, #f97316, #ef4444)';
  return 'linear-gradient(90deg, #ef4444, #dc2626)';
}
function riskDescription(cat) {
  return {
    Low: 'Maintain current healthy habits. Annual checkup recommended.',
    Moderate: 'Some risk factors present. Lifestyle changes advised. Consult physician.',
    High: 'Significant health risks detected. See a doctor within 2 weeks.',
    Critical: 'Immediate medical attention required. Multiple high-risk indicators detected.'
  }[cat] || '';
}
function getTopDisease(predictions) {
  if (!predictions) return '—';
  const top = Object.entries(predictions).sort((a, b) => b[1].probability - a[1].probability)[0];
  const names = { diabetes: 'Diabetes', heart_disease: 'Heart Disease', hypertension: 'Hypertension', stroke: 'Stroke', obesity: 'Obesity' };
  return top ? `${names[top[0]] || top[0]} (${top[1].probability}%)` : '—';
}
function formatDate(dt) {
  if (!dt) return '—';
  return new Date(dt).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
}

// ─── Init ─────────────────────────────────────────────────────────────────────
loadDashboard();
