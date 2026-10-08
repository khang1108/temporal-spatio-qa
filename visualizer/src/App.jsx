import React, { useState, useMemo } from 'react';
import predictionsData from './data/predictions_data.json';
import './App.css';
import { 
  AlertTriangle, 
  CheckCircle, 
  XCircle, 
  Compass, 
  Search, 
  Flame, 
  Eye, 
  Layers, 
  Sparkles,
  ExternalLink,
  ChevronLeft,
  ChevronRight,
  X,
  Copy,
  Check
} from 'lucide-react';

const ITEMS_PER_PAGE = 12;

export default function App() {
  const { metrics, items, total, fails_count, correct_count } = predictionsData;

  // State
  const [filterStatus, setFilterStatus] = useState('fail'); // 'all', 'fail', 'correct'
  const [filterAxis, setFilterAxis] = useState('all');
  const [filterPersp, setFilterPersp] = useState('all');
  const [filterCategory, setFilterCategory] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [isProbeMode, setIsProbeMode] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedItem, setSelectedItem] = useState(null);
  const [copiedId, setCopiedId] = useState(null);

  // Available failure types
  const failureCategories = useMemo(() => {
    const set = new Set();
    items.forEach(it => {
      if (!it.is_correct && it.failure_type) {
        set.add(it.failure_type);
      }
    });
    return Array.from(set);
  }, [items]);

  // Toggle Top 30 Probe Set
  const handleToggleProbe = () => {
    if (!isProbeMode) {
      setIsProbeMode(true);
      setFilterStatus('fail');
      setFilterCategory('Sập bẫy Camera 2D (Egocentric Left/Right Inversion)');
      setCurrentPage(1);
    } else {
      setIsProbeMode(false);
      setFilterCategory('all');
    }
  };

  // Filter items
  const filteredItems = useMemo(() => {
    return items.filter(item => {
      // Status filter
      if (filterStatus === 'fail' && item.is_correct) return false;
      if (filterStatus === 'correct' && !item.is_correct) return false;

      // Axis filter
      if (filterAxis !== 'all' && !item.axis.includes(filterAxis)) return false;

      // Perspective filter
      if (filterPersp !== 'all' && item.perspective_key !== filterPersp) return false;

      // Category filter
      if (filterCategory !== 'all' && item.failure_type !== filterCategory) return false;

      // Search query
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matchQ = item.question.toLowerCase().includes(q);
        const matchAns = item.answer.toLowerCase().includes(q);
        const matchPred = item.output.toLowerCase().includes(q);
        const matchImg = item.image.toLowerCase().includes(q);
        if (!matchQ && !matchAns && !matchPred && !matchImg) return false;
      }

      return true;
    });
  }, [items, filterStatus, filterAxis, filterPersp, filterCategory, searchQuery]);

  // Pagination
  const totalPages = Math.ceil(filteredItems.length / ITEMS_PER_PAGE) || 1;
  const paginatedItems = useMemo(() => {
    const start = (currentPage - 1) * ITEMS_PER_PAGE;
    return filteredItems.slice(start, start + ITEMS_PER_PAGE);
  }, [filteredItems, currentPage]);

  const handleCopyJson = (item) => {
    navigator.clipboard.writeText(JSON.stringify(item, null, 2));
    setCopiedId(item.id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-title-group">
          <h1>
            <span>SpatialMQA Failure Visualizer</span>
            <span className="header-badge">ACL 2025 Long</span>
          </h1>
          <p className="header-subtitle">
            Phân tích cơ chế thất bại tầng Token/Tham số của LLaVA-1.5-7B LoRA trên benchmark SpatialMQA.
            Trực quan hóa hiện tượng <strong>Sập bẫy Camera 2D</strong> và <strong>Sụp đổ biểu diễn không gian ẩn</strong>.
          </p>
        </div>

        <div className="header-actions">
          <button 
            id="probe-toggle-btn"
            className={`btn-probe ${isProbeMode ? 'active' : ''}`}
            onClick={handleToggleProbe}
          >
            <Flame size={16} />
            {isProbeMode ? 'Đang lọc Probe Set (30 ca)' : '🔥 Top Ca Sập Bẫy Camera (Probe Set)'}
          </button>
        </div>
      </header>

      {/* KPI Metrics Dashboard */}
      <section className="metrics-grid">
        <div className="metric-card" style={{ '--card-accent': '#6366f1' }}>
          <div className="metric-header">
            <span className="metric-label">Độ chính xác Tổng (Overall)</span>
            <span className="metric-tag" style={{ background: 'rgba(99, 102, 241, 0.15)', color: '#818cf8' }}>
              Target: 46.85%
            </span>
          </div>
          <div className="metric-value-row">
            <span className="metric-value">{metrics.overall.accuracy}%</span>
            <span className="metric-comparison delta-gap">
              -10.79% vs Paper
            </span>
          </div>
          <p className="metric-subtext">388 đúng / 1,076 mẫu test chuẩn</p>
        </div>

        <div className="metric-card" style={{ '--card-accent': '#f43f5e' }}>
          <div className="metric-header">
            <span className="metric-label">Trục Ngang $A_x$ (left/right)</span>
            <span className="metric-tag" style={{ background: 'rgba(244, 63, 94, 0.15)', color: '#fb7185' }}>
              Nút thắt chính (54% dataset)
            </span>
          </div>
          <div className="metric-value-row">
            <span className="metric-value">{metrics.by_axis.A_x}%</span>
            <span className="metric-comparison delta-gap">
              -19.36% vs Paper (55.71%)
            </span>
          </div>
          <p className="metric-subtext">Sập bẫy đảo ngược góc nhìn Trái/Phải</p>
        </div>

        <div className="metric-card" style={{ '--card-accent': '#10b981' }}>
          <div className="metric-header">
            <span className="metric-label">Trục Chiều Sâu $A_y$ (front/behind)</span>
            <span className="metric-tag" style={{ background: 'rgba(16, 185, 129, 0.15)', color: '#34d399' }}>
              Paper: 29.64%
            </span>
          </div>
          <div className="metric-value-row">
            <span className="metric-value">{metrics.by_axis.A_y}%</span>
            <span className="metric-comparison delta-matched">
              ✓ Khớp chuẩn xác (-0.47%)
            </span>
          </div>
          <p className="metric-subtext">Đã hội tụ tương đương công bố</p>
        </div>

        <div className="metric-card" style={{ '--card-accent': '#06b6d4' }}>
          <div className="metric-header">
            <span className="metric-label">Trục Đứng $A_z$ (above/below)</span>
            <span className="metric-tag" style={{ background: 'rgba(6, 182, 212, 0.15)', color: '#22d3ee' }}>
              Paper: 48.13%
            </span>
          </div>
          <div className="metric-value-row">
            <span className="metric-value">{metrics.by_axis.A_z}%</span>
            <span className="metric-comparison delta-matched">
              ✓ Khớp chuẩn xác (-1.57%)
            </span>
          </div>
          <p className="metric-subtext">Hình học trục trọng lực Z ổn định</p>
        </div>
      </section>

      {/* Perspective Rule Strip */}
      <div className="perspective-strip">
        <div className="persp-item">
          <div className="persp-dot" style={{ background: '#38bdf8' }} />
          <div className="persp-info">
            <span className="persp-name">Q1: Camera Ngoài Cảnh (452 câu)</span>
            <span className="persp-score">{metrics.by_perspective.Q1_OutOfImage}% <small style={{ color: '#94a3b8' }}>(Paper: 53.14%)</small></span>
          </div>
        </div>
        <div className="persp-item">
          <div className="persp-dot" style={{ background: '#f43f5e' }} />
          <div className="persp-info">
            <span className="persp-name">Q2: Nhập Vai Thứ Nhất Ego (590 câu)</span>
            <span className="persp-score" style={{ color: '#fb7185' }}>{metrics.by_perspective.Q2_FirstPerson}% <small style={{ color: '#94a3b8' }}>(Paper: 40.99%)</small></span>
          </div>
        </div>
        <div className="persp-item">
          <div className="persp-dot" style={{ background: '#a855f7' }} />
          <div className="persp-info">
            <span className="persp-name">Q3: Góc Nhìn Thứ 3 (34 câu)</span>
            <span className="persp-score">{metrics.by_perspective.Q3_ThirdPerson}% <small style={{ color: '#94a3b8' }}>(Paper: 64.71%)</small></span>
          </div>
        </div>
      </div>

      {/* Interactive Filter Bar */}
      <section className="filters-container">
        {/* Search Input */}
        <div className="search-input-wrapper">
          <Search size={18} className="search-icon" />
          <input 
            id="filter-search-input"
            type="text" 
            className="search-input"
            placeholder="Tìm kiếm câu hỏi, thực thể (car, clock, person, giraffe...), file ảnh, hoặc lựa chọn..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setCurrentPage(1);
            }}
          />
        </div>

        {/* Primary Status Pills */}
        <div className="filter-pills-row">
          <span className="filter-group-label">Trạng thái:</span>
          <button 
            className={`pill-btn ${filterStatus === 'all' ? 'active' : ''}`}
            onClick={() => { setFilterStatus('all'); setCurrentPage(1); }}
          >
            Tất cả <span className="pill-badge">{total}</span>
          </button>
          <button 
            className={`pill-btn ${filterStatus === 'fail' ? 'active' : ''}`}
            onClick={() => { setFilterStatus('fail'); setCurrentPage(1); }}
          >
            <XCircle size={14} color="#f43f5e" />
            Ca Thất Bại <span className="pill-badge">{fails_count}</span>
          </button>
          <button 
            className={`pill-btn ${filterStatus === 'correct' ? 'active' : ''}`}
            onClick={() => { setFilterStatus('correct'); setCurrentPage(1); }}
          >
            <CheckCircle size={14} color="#10b981" />
            Ca Chính Xác <span className="pill-badge">{correct_count}</span>
          </button>
        </div>

        {/* Dropdowns for Axis, Perspective, Failure Category */}
        <div className="dropdown-filters">
          <div>
            <select 
              id="select-axis"
              className="custom-select"
              value={filterAxis}
              onChange={(e) => { setFilterAxis(e.target.value); setCurrentPage(1); }}
            >
              <option value="all">Mọi trục tọa độ (Ax, Ay, Az)</option>
              <option value="A_x">Trục Ngang Ax (Horizontal)</option>
              <option value="A_y">Trục Sâu Ay (Depth)</option>
              <option value="A_z">Trục Đứng Az (Vertical)</option>
            </select>
          </div>

          <div>
            <select 
              id="select-perspective"
              className="custom-select"
              value={filterPersp}
              onChange={(e) => { setFilterPersp(e.target.value); setCurrentPage(1); }}
            >
              <option value="all">Mọi hệ quy chiếu góc nhìn</option>
              <option value="Q1_OutOfImage">Q1: Camera Ngoài Cảnh (Exocentric)</option>
              <option value="Q2_FirstPerson">Q2: Nhập Vai Thứ Nhất (Egocentric)</option>
              <option value="Q3_ThirdPerson">Q3: Góc Nhìn Thứ Ba trong ảnh</option>
            </select>
          </div>

          <div>
            <select 
              id="select-category"
              className="custom-select"
              value={filterCategory}
              onChange={(e) => { setFilterCategory(e.target.value); setCurrentPage(1); }}
            >
              <option value="all">Mọi phân loại cơ chế thất bại</option>
              {failureCategories.map((cat, i) => (
                <option key={i} value={cat}>{cat}</option>
              ))}
            </select>
          </div>
        </div>
      </section>

      {/* Results Header */}
      <div className="results-header">
        <span className="results-count">
          Đang hiển thị <strong>{filteredItems.length}</strong> mẫu 
          {filterStatus === 'fail' ? ' thất bại' : ''} (Trang {currentPage} / {totalPages})
        </span>

        {/* Quick pagination on top if items exist */}
        {totalPages > 1 && (
          <div className="pagination-row" style={{ margin: 0 }}>
            <button 
              className="page-btn" 
              disabled={currentPage === 1}
              onClick={() => setCurrentPage(p => p - 1)}
            >
              <ChevronLeft size={16} />
            </button>
            <span className="page-info">{currentPage} / {totalPages}</span>
            <button 
              className="page-btn" 
              disabled={currentPage === totalPages}
              onClick={() => setCurrentPage(p => p + 1)}
            >
              <ChevronRight size={16} />
            </button>
          </div>
        )}
      </div>

      {/* Gallery Grid */}
      <div className="gallery-grid">
        {paginatedItems.map((item) => (
          <div 
            key={item.id} 
            className={`failure-card ${item.is_correct ? 'card-correct' : 'card-fail'}`}
            onClick={() => setSelectedItem(item)}
            style={{ cursor: 'pointer' }}
          >
            {/* Image Thumbnail */}
            <div className="card-media-wrapper">
              <img 
                src={item.image_url} 
                alt={item.image}
                className="card-img"
                loading="lazy"
                onError={(e) => {
                  e.target.onerror = null;
                  e.target.src = "https://images.cocodataset.org/val2017/" + item.image;
                }}
              />
              <span className={`card-floating-badge ${item.is_correct ? 'badge-correct' : 'badge-fail'}`}>
                {item.is_correct ? 'ĐÚNG' : 'THẤT BẠI'}
              </span>
              <span className="card-id-tag">#{item.id}</span>
            </div>

            {/* Card Content */}
            <div className="card-body">
              <div className="card-meta-row">
                <span className="axis-badge">{item.axis}</span>
                <span className="persp-badge">{item.perspective_key}</span>
              </div>

              <p className="card-question" title={item.question}>
                {item.question}
              </p>

              {/* Comparison Box */}
              <div className="prediction-box">
                <div className="compare-row">
                  <span className="compare-label">Đáp án Ground Truth:</span>
                  <span className="val-gt">{item.answer}</span>
                </div>
                <div className="compare-row">
                  <span className="compare-label">LLaVA-1.5 Dự đoán:</span>
                  <span className={item.is_correct ? 'val-pred-ok' : 'val-pred-fail'}>
                    {item.output}
                  </span>
                </div>
              </div>

              {/* Mechanistic Diagnostic Callout if failed */}
              {!item.is_correct && (
                <div className="failure-reason-callout">
                  <div className="failure-title">
                    <AlertTriangle size={13} />
                    <span>{item.failure_type}</span>
                  </div>
                  <p className="failure-desc">{item.failure_desc}</p>
                </div>
              )}

              {/* Options Tags */}
              <div className="options-list">
                {item.options.map((opt, optIdx) => {
                  const isGt = opt.toLowerCase() === item.answer.toLowerCase();
                  const isPred = opt.toLowerCase() === item.output.toLowerCase() && !item.is_correct;
                  return (
                    <span 
                      key={optIdx} 
                      className={`option-tag ${isGt ? 'is-gt' : ''} ${isPred ? 'is-pred' : ''}`}
                    >
                      {opt} {isGt ? '✓' : ''} {isPred ? '✗' : ''}
                    </span>
                  );
                })}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Bottom Pagination */}
      {totalPages > 1 && (
        <div className="pagination-row">
          <button 
            id="prev-page-btn"
            className="page-btn" 
            disabled={currentPage === 1}
            onClick={() => {
              setCurrentPage(p => p - 1);
              window.scrollTo({ top: 300, behavior: 'smooth' });
            }}
          >
            <ChevronLeft size={16} /> Trang trước
          </button>
          <span className="page-info">Trang {currentPage} trên tổng số {totalPages}</span>
          <button 
            id="next-page-btn"
            className="page-btn" 
            disabled={currentPage === totalPages}
            onClick={() => {
              setCurrentPage(p => p + 1);
              window.scrollTo({ top: 300, behavior: 'smooth' });
            }}
          >
            Trang tiếp <ChevronRight size={16} />
          </button>
        </div>
      )}

      {/* Detail Inspection Modal */}
      {selectedItem && (
        <div className="modal-overlay" onClick={() => setSelectedItem(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div>
                <h3 style={{ fontSize: '18px', fontWeight: 700 }}>
                  Chẩn đoán Chi tiết Mẫu #{selectedItem.id} ({selectedItem.image})
                </h3>
                <span style={{ fontSize: '12px', color: 'var(--text-dim)' }}>
                  {selectedItem.perspective} • {selectedItem.axis}
                </span>
              </div>
              <button className="modal-close-btn" onClick={() => setSelectedItem(null)}>
                <X size={20} />
              </button>
            </div>

            <div className="modal-body">
              {/* Image Column */}
              <div>
                <img 
                  src={selectedItem.image_url} 
                  alt={selectedItem.image}
                  className="modal-img"
                  onError={(e) => {
                    e.target.onerror = null;
                    e.target.src = "https://images.cocodataset.org/val2017/" + selectedItem.image;
                  }}
                />
                <div style={{ marginTop: '12px', display: 'flex', gap: '8px' }}>
                  <a 
                    href={selectedItem.image_url} 
                    target="_blank" 
                    rel="noreferrer"
                    style={{ fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px' }}
                  >
                    <ExternalLink size={14} /> Mở ảnh gốc full-res
                  </a>
                  <button 
                    onClick={() => handleCopyJson(selectedItem)}
                    style={{ 
                      background: 'none', 
                      border: 'none', 
                      color: 'var(--primary-light)', 
                      fontSize: '12px', 
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px'
                    }}
                  >
                    {copiedId === selectedItem.id ? <Check size={14} color="#10b981" /> : <Copy size={14} />}
                    {copiedId === selectedItem.id ? 'Đã sao chép JSON!' : 'Copy JSON mẫu'}
                  </button>
                </div>
              </div>

              {/* Analysis Column */}
              <div>
                <h4 style={{ fontSize: '13px', color: 'var(--text-dim)', marginBottom: '6px' }}>CÂU HỎI BENCHMARK:</h4>
                <p style={{ fontSize: '15px', fontWeight: 600, marginBottom: '16px', lineHeight: 1.5 }}>
                  {selectedItem.question}
                </p>

                <div className="prediction-box" style={{ marginBottom: '16px' }}>
                  <div className="compare-row">
                    <span className="compare-label">Đáp án Ground Truth:</span>
                    <span className="val-gt" style={{ fontSize: '14px' }}>{selectedItem.answer}</span>
                  </div>
                  <div className="compare-row">
                    <span className="compare-label">Mô hình Dự đoán:</span>
                    <span className={selectedItem.is_correct ? 'val-pred-ok' : 'val-pred-fail'} style={{ fontSize: '14px' }}>
                      {selectedItem.output}
                    </span>
                  </div>
                </div>

                {!selectedItem.is_correct && (
                  <div className="failure-reason-callout" style={{ padding: '14px', marginBottom: '16px' }}>
                    <div className="failure-title" style={{ fontSize: '14px' }}>
                      <AlertTriangle size={16} />
                      <span>{selectedItem.failure_type}</span>
                    </div>
                    <p className="failure-desc" style={{ marginTop: '6px', fontSize: '13px' }}>
                      {selectedItem.failure_desc}
                    </p>
                  </div>
                )}

                <h4 style={{ fontSize: '13px', color: 'var(--text-dim)', marginBottom: '8px' }}>DANH SÁCH LỰA CHỌN (OPTIONS):</h4>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                  {selectedItem.options.map((opt, oIdx) => (
                    <span 
                      key={oIdx}
                      style={{
                        padding: '5px 12px',
                        borderRadius: '6px',
                        fontSize: '13px',
                        background: opt.toLowerCase() === selectedItem.answer.toLowerCase() ? 'var(--emerald-bg)' : 
                                   opt.toLowerCase() === selectedItem.output.toLowerCase() && !selectedItem.is_correct ? 'var(--rose-bg)' : 'rgba(255,255,255,0.05)',
                        color: opt.toLowerCase() === selectedItem.answer.toLowerCase() ? 'var(--emerald)' : 
                               opt.toLowerCase() === selectedItem.output.toLowerCase() && !selectedItem.is_correct ? 'var(--rose)' : 'var(--text-muted)',
                        border: opt.toLowerCase() === selectedItem.answer.toLowerCase() ? '1px solid var(--emerald)' : 
                                opt.toLowerCase() === selectedItem.output.toLowerCase() && !selectedItem.is_correct ? '1px solid var(--rose)' : '1px solid rgba(255,255,255,0.08)',
                        fontWeight: 600
                      }}
                    >
                      {opt}
                    </span>
                  ))}
                </div>

                <div style={{ marginTop: '20px', paddingTop: '16px', borderTop: '1px solid rgba(255,255,255,0.08)' }}>
                  <span style={{ fontSize: '12px', color: 'var(--text-dim)' }}>
                    Khuyến nghị nghiên cứu (Trụ cột 2): Bơm <strong>Embodiment Steering Prefix</strong> tại Layer chú ý để ép xoay hệ quy chiếu về nhân vật trước khi sinh từ.
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
