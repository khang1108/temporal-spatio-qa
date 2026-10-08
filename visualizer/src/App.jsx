import React, { useState, useEffect, useMemo } from 'react';
import predictionsData from './data/predictions_data.json';
import './App.css';

export default function App() {
  const { metrics, items, total, fails_count, correct_count } = predictionsData;

  // User ID state (persisted in localStorage)
  const [userId, setUserId] = useState(() => {
    return localStorage.getItem('annotator_user_id') || 'Khang';
  });

  const handleUserIdChange = (e) => {
    const val = e.target.value;
    setUserId(val);
    localStorage.setItem('annotator_user_id', val);
  };

  // Comments state from SQLite database
  const [commentsList, setCommentsList] = useState([]);
  const [commentInputs, setCommentInputs] = useState({});
  const [expandedComments, setExpandedComments] = useState({});
  const [isSubmitting, setIsSubmitting] = useState({});
  const [commentFilterOnly, setCommentFilterOnly] = useState(false);

  // Modal inspection
  const [selectedItem, setSelectedItem] = useState(null);

  // Filters & Pagination
  const [filterStatus, setFilterStatus] = useState('fail'); // 'all', 'fail', 'correct'
  const [filterAxis, setFilterAxis] = useState('all');
  const [filterPersp, setFilterPersp] = useState('all');
  const [filterCategory, setFilterCategory] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [pageSize, setPageSize] = useState(12);
  const [currentPage, setCurrentPage] = useState(1);

  // Load comments from backend SQLite
  const loadComments = () => {
    fetch('/api/comments')
      .then(res => res.json())
      .then(data => {
        if (data.success && Array.isArray(data.comments)) {
          setCommentsList(data.comments);
        }
      })
      .catch(err => {
        console.warn('[Comments] Could not connect to /api/comments SQLite backend:', err);
      });
  };

  useEffect(() => {
    loadComments();
  }, []);

  // Map comments by sample_id
  const commentsBySample = useMemo(() => {
    const map = {};
    commentsList.forEach(c => {
      if (!map[c.sample_id]) map[c.sample_id] = [];
      map[c.sample_id].push(c);
    });
    return map;
  }, [commentsList]);

  // Failure categories
  const failureCategories = useMemo(() => {
    const set = new Set();
    items.forEach(it => {
      if (!it.is_correct && it.failure_type) {
        set.add(it.failure_type);
      }
    });
    return Array.from(set);
  }, [items]);

  // Handle comment submit
  const handleAddComment = (sampleId) => {
    const text = (commentInputs[sampleId] || '').trim();
    if (!text) return;

    const trimmedUserId = userId.trim();
    if (!trimmedUserId) {
      alert('Vui lòng nhập Người đánh giá (User ID) ở góc trên bên phải trước khi gửi bình luận!');
      return;
    }

    setIsSubmitting(prev => ({ ...prev, [sampleId]: true }));

    fetch('/api/comments', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sample_id: sampleId,
        user_id: trimmedUserId,
        comment: text
      })
    })
      .then(res => res.json())
      .then(data => {
        if (data.success && data.comment) {
          setCommentsList(prev => [data.comment, ...prev]);
          setCommentInputs(prev => ({ ...prev, [sampleId]: '' }));
          setExpandedComments(prev => ({ ...prev, [sampleId]: true }));
        } else {
          alert('Lỗi lưu bình luận: ' + (data.error || 'Unknown error'));
        }
      })
      .catch(err => {
        alert('Lỗi kết nối máy chủ SQLite: ' + err.message);
      })
      .finally(() => {
        setIsSubmitting(prev => ({ ...prev, [sampleId]: false }));
      });
  };

  // Filter items
  const filteredItems = useMemo(() => {
    return items.filter(item => {
      // Filter by comments presence
      if (commentFilterOnly && (!commentsBySample[item.id] || commentsBySample[item.id].length === 0)) {
        return false;
      }

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
  }, [items, filterStatus, filterAxis, filterPersp, filterCategory, searchQuery, commentFilterOnly, commentsBySample]);

  // Pagination
  const totalPages = Math.ceil(filteredItems.length / pageSize) || 1;
  const paginatedItems = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredItems.slice(start, start + pageSize);
  }, [filteredItems, currentPage, pageSize]);

  return (
    <div className="container">
      {/* Top Navigation & User ID Bar */}
      <header className="top-nav">
        <div className="brand-section">
          <h1>SpatialMQA Failure Analysis & Diagnostic Portal</h1>
          <p className="brand-desc">
            Phân tích cơ chế thất bại mô hình LLaVA-1.5 trên SpatialMQA (ACL 2025 Long) | SQLite Annotation System
          </p>
        </div>

        <div className="user-id-bar">
          <label htmlFor="user-id-input" className="user-id-label">Người đánh giá (User ID):</label>
          <input
            id="user-id-input"
            type="text"
            className="user-id-input"
            value={userId}
            onChange={handleUserIdChange}
            placeholder="Nhập User ID (VD: Khang)"
          />
          <span className="user-id-badge">SQLite Active</span>
        </div>
      </header>

      {/* Benchmark Summary Table (Wide, high contrast) */}
      <section className="stats-table-wrapper">
        <table className="stats-table">
          <thead>
            <tr>
              <th>Chỉ số</th>
              <th>Tổng thể (Overall)</th>
              <th>Trục Ngang (Ax: left/right)</th>
              <th>Trục Sâu (Ay: front/behind)</th>
              <th>Trục Đứng (Az: above/below)</th>
              <th>Q1 (Camera ngoài cảnh)</th>
              <th>Q2 (Nhập vai thứ nhất - Ego)</th>
              <th>Q3 (Góc nhìn thứ 3)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Kết quả Run hiện tại</strong></td>
              <td className="stat-val-bold">{metrics.overall.accuracy}%</td>
              <td className="stat-val-bold" style={{ color: '#b91c1c' }}>{metrics.by_axis.A_x}%</td>
              <td className="stat-val-bold" style={{ color: '#15803d' }}>{metrics.by_axis.A_y}%</td>
              <td className="stat-val-bold" style={{ color: '#15803d' }}>{metrics.by_axis.A_z}%</td>
              <td className="stat-val-bold">{metrics.by_perspective.Q1_OutOfImage}%</td>
              <td className="stat-val-bold" style={{ color: '#b91c1c' }}>{metrics.by_perspective.Q2_FirstPerson}%</td>
              <td className="stat-val-bold">{metrics.by_perspective.Q3_ThirdPerson}%</td>
            </tr>
            <tr>
              <td><strong>Paper ACL 2025 (LoRA)</strong></td>
              <td>46.85%</td>
              <td>55.71%</td>
              <td>29.64%</td>
              <td>48.13%</td>
              <td>53.14%</td>
              <td>40.99%</td>
              <td>64.71%</td>
            </tr>
            <tr>
              <td><strong>Đánh giá chênh lệch</strong></td>
              <td><span className="tag-gap">-10.79%</span></td>
              <td><span className="tag-gap">-19.36% (Nút thắt FRS)</span></td>
              <td><span className="tag-matched">✓ Khớp (-0.47%)</span></td>
              <td><span className="tag-matched">✓ Khớp (-1.57%)</span></td>
              <td><span className="tag-gap">-11.33%</span></td>
              <td><span className="tag-gap">-10.31% (Sập bẫy camera)</span></td>
              <td><span className="tag-gap">-11.77%</span></td>
            </tr>
          </tbody>
        </table>
      </section>

      {/* Filter Strip */}
      <section className="filter-panel">
        <div className="filter-row">
          <input
            id="search-input"
            type="text"
            className="search-field"
            placeholder="Tìm kiếm theo câu hỏi, tên file ảnh, thực thể (giraffe, car, clock...), đáp án..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setCurrentPage(1);
            }}
          />

          <div className="filter-btn-group">
            <button
              className={`filter-btn ${filterStatus === 'all' && !commentFilterOnly ? 'active' : ''}`}
              onClick={() => { setFilterStatus('all'); setCommentFilterOnly(false); setCurrentPage(1); }}
            >
              Tất cả [{total}]
            </button>
            <button
              className={`filter-btn ${filterStatus === 'fail' && !commentFilterOnly ? 'active' : ''}`}
              onClick={() => { setFilterStatus('fail'); setCommentFilterOnly(false); setCurrentPage(1); }}
            >
              Ca Thất Bại [{fails_count}]
            </button>
            <button
              className={`filter-btn ${filterStatus === 'correct' && !commentFilterOnly ? 'active' : ''}`}
              onClick={() => { setFilterStatus('correct'); setCommentFilterOnly(false); setCurrentPage(1); }}
            >
              Ca Đúng [{correct_count}]
            </button>
            <button
              className={`filter-btn ${commentFilterOnly ? 'active' : ''}`}
              onClick={() => { setCommentFilterOnly(!commentFilterOnly); setCurrentPage(1); }}
            >
              Đã Có Bình Luận [{Object.keys(commentsBySample).length}]
            </button>
          </div>
        </div>

        <div className="filter-row">
          <select
            id="filter-axis-select"
            className="filter-select"
            value={filterAxis}
            onChange={(e) => { setFilterAxis(e.target.value); setCurrentPage(1); }}
          >
            <option value="all">Mọi trục tọa độ (Ax, Ay, Az)</option>
            <option value="A_x">Trục Ngang Ax (Horizontal: left/right - 575 mẫu)</option>
            <option value="A_y">Trục Sâu Ay (Depth: front/behind - 312 mẫu)</option>
            <option value="A_z">Trục Đứng Az (Vertical: above/below - 189 mẫu)</option>
          </select>

          <select
            id="filter-persp-select"
            className="filter-select"
            value={filterPersp}
            onChange={(e) => { setFilterPersp(e.target.value); setCurrentPage(1); }}
          >
            <option value="all">Mọi góc nhìn (Q1, Q2, Q3)</option>
            <option value="Q1_OutOfImage">Q1: Camera Ngoài Cảnh (Exocentric - 452 mẫu)</option>
            <option value="Q2_FirstPerson">Q2: Nhập Vai Thứ Nhất (Egocentric - 590 mẫu)</option>
            <option value="Q3_ThirdPerson">Q3: Góc Nhìn Thứ Ba trong ảnh (34 mẫu)</option>
          </select>

          <select
            id="filter-cat-select"
            className="filter-select"
            value={filterCategory}
            onChange={(e) => { setFilterCategory(e.target.value); setCurrentPage(1); }}
          >
            <option value="all">Mọi phân loại nguyên nhân thất bại</option>
            {failureCategories.map((c, i) => (
              <option key={i} value={c}>{c}</option>
            ))}
          </select>

          <select
            className="filter-select"
            value={pageSize}
            onChange={(e) => { setPageSize(Number(e.target.value)); setCurrentPage(1); }}
          >
            <option value={12}>12 mẫu / trang</option>
            <option value={24}>24 mẫu / trang</option>
            <option value={48}>48 mẫu / trang</option>
            <option value={96}>96 mẫu / trang</option>
          </select>
        </div>
      </section>

      {/* Results Header */}
      <div className="results-summary-bar">
        <div className="results-count-text">
          Hiển thị <strong>{filteredItems.length}</strong> mẫu 
          {filterStatus === 'fail' ? ' [Lỗi]' : filterStatus === 'correct' ? ' [Đúng]' : ''} 
          (Trang {currentPage} / {totalPages})
        </div>

        {totalPages > 1 && (
          <div className="pagination-controls">
            <button
              className="btn-nav"
              disabled={currentPage === 1}
              onClick={() => setCurrentPage(p => p - 1)}
            >
              Trang trước
            </button>
            <span style={{ fontSize: '13px', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
              {currentPage} / {totalPages}
            </span>
            <button
              className="btn-nav"
              disabled={currentPage === totalPages}
              onClick={() => setCurrentPage(p => p + 1)}
            >
              Trang tiếp
            </button>
          </div>
        )}
      </div>

      {/* Wide Card Grid */}
      <div className="wide-grid">
        {paginatedItems.map((item) => {
          const sampleComments = commentsBySample[item.id] || [];
          const isExpanded = !!expandedComments[item.id];

          return (
            <div
              key={item.id}
              className={`case-card ${item.is_correct ? 'card-ok-border' : 'card-fail-border'}`}
            >
              {/* Card Head */}
              <div className="case-card-header">
                <span className="case-card-id">#{item.id} [{item.image}]</span>
                <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                  <span className={item.is_correct ? 'badge-tag-ok' : 'badge-tag-fail'}>
                    {item.is_correct ? '[ĐÚNG]' : '[THẤT BẠI]'}
                  </span>
                  <button
                    className="btn-nav"
                    style={{ padding: '2px 6px', fontSize: '11px' }}
                    onClick={() => setSelectedItem(item)}
                  >
                    Phóng to
                  </button>
                </div>
              </div>

              {/* Card Image */}
              <div className="case-img-container">
                <img
                  src={item.image_url}
                  alt={item.image}
                  className="case-img"
                  loading="lazy"
                  onError={(e) => {
                    e.target.onerror = null;
                    e.target.src = "https://images.cocodataset.org/val2017/" + item.image;
                  }}
                />
              </div>

              {/* Card Body */}
              <div className="case-card-body">
                <div className="meta-tags-line">
                  <span className="mini-tag">{item.axis}</span>
                  <span className="mini-tag">{item.perspective_key}</span>
                </div>

                <p className="question-text">
                  {item.question}
                </p>

                {/* Ground Truth vs Output */}
                <div className="compare-box">
                  <div className="compare-item">
                    <span className="ans-label">Đáp án đúng (Ground Truth):</span>
                    <span className="ans-gt">{item.answer}</span>
                  </div>
                  <div className="compare-item">
                    <span className="ans-label">LLaVA-1.5 dự đoán:</span>
                    <span className={item.is_correct ? 'ans-pred-ok' : 'ans-pred-wrong'}>
                      {item.output}
                    </span>
                  </div>
                </div>

                {/* Failure explanation box */}
                {!item.is_correct && (
                  <div className="reason-box">
                    <div className="reason-title">Nguyên nhân: {item.failure_type}</div>
                    <div className="reason-desc">{item.failure_desc}</div>
                  </div>
                )}

                {/* Options List */}
                <div className="options-wrap">
                  {item.options.map((opt, oIdx) => {
                    const isGt = opt.toLowerCase() === item.answer.toLowerCase();
                    const isPred = opt.toLowerCase() === item.output.toLowerCase() && !item.is_correct;
                    return (
                      <span
                        key={oIdx}
                        className={`opt-pill ${isGt ? 'gt-match' : ''} ${isPred ? 'pred-match' : ''}`}
                      >
                        {opt} {isGt ? '[Đúng]' : ''} {isPred ? '[Model chọn]' : ''}
                      </span>
                    );
                  })}
                </div>

                {/* ==============================================================
                    SQLite Comments Section
                    ============================================================== */}
                <div className="comments-section">
                  <div className="comments-header-row">
                    <span className="comments-count-badge">
                      Nhận xét ({sampleComments.length})
                    </span>
                    <button
                      className="btn-toggle-comments"
                      onClick={() => setExpandedComments(prev => ({ ...prev, [item.id]: !isExpanded }))}
                    >
                      {isExpanded ? 'Thu gọn' : 'Xem / Thêm'}
                    </button>
                  </div>

                  {isExpanded && (
                    <>
                      {/* Comments List */}
                      {sampleComments.length > 0 && (
                        <div className="comments-list">
                          {sampleComments.map((c) => (
                            <div key={c.id} className="comment-bubble">
                              <div className="comment-meta">
                                <span className="comment-author">[{c.user_id}]</span>
                                <span>{c.created_at ? new Date(c.created_at).toLocaleString('vi-VN') : ''}</span>
                              </div>
                              <p className="comment-text">{c.comment}</p>
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Add Comment Form */}
                      <div className="comment-form">
                        <textarea
                          className="comment-textarea"
                          placeholder={`Viết nhận xét của ${userId || 'User'} về mẫu #${item.id}...`}
                          value={commentInputs[item.id] || ''}
                          onChange={(e) => setCommentInputs(prev => ({ ...prev, [item.id]: e.target.value }))}
                        />
                        <button
                          className="btn-submit-comment"
                          disabled={isSubmitting[item.id] || !(commentInputs[item.id] || '').trim()}
                          onClick={() => handleAddComment(item.id)}
                        >
                          {isSubmitting[item.id] ? 'Đang lưu...' : 'Gửi nhận xét vào SQLite'}
                        </button>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Bottom Pagination */}
      {totalPages > 1 && (
        <div style={{ display: 'flex', justifyContent: 'center', marginTop: '30px' }}>
          <div className="pagination-controls">
            <button
              className="btn-nav"
              disabled={currentPage === 1}
              onClick={() => { setCurrentPage(p => p - 1); window.scrollTo({ top: 300, behavior: 'smooth' }); }}
            >
              Trang trước
            </button>
            <span style={{ fontSize: '13px', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
              Trang {currentPage} / {totalPages}
            </span>
            <button
              className="btn-nav"
              disabled={currentPage === totalPages}
              onClick={() => { setCurrentPage(p => p + 1); window.scrollTo({ top: 300, behavior: 'smooth' }); }}
            >
              Trang tiếp
            </button>
          </div>
        </div>
      )}

      {/* Modal Zoom View */}
      {selectedItem && (
        <div className="modal-backdrop" onClick={() => setSelectedItem(null)}>
          <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <h3>Chi tiết Mẫu #{selectedItem.id} ({selectedItem.image})</h3>
              <button className="btn-close" onClick={() => setSelectedItem(null)}>Đóng [X]</button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px' }}>
              <div>
                <img
                  src={selectedItem.image_url}
                  alt={selectedItem.image}
                  style={{ width: '100%', borderRadius: '6px', border: '1px solid var(--border-color)', display: 'block' }}
                />
                <div style={{ marginTop: '8px', fontSize: '12px' }}>
                  <a href={selectedItem.image_url} target="_blank" rel="noreferrer">
                    Mở ảnh gốc full-res (Hugging Face CDN)
                  </a>
                </div>
              </div>

              <div>
                <div style={{ marginBottom: '12px' }}>
                  <span className="mini-tag" style={{ marginRight: '6px' }}>{selectedItem.axis}</span>
                  <span className="mini-tag">{selectedItem.perspective}</span>
                </div>

                <p style={{ fontSize: '15px', fontWeight: 700, marginBottom: '12px', lineHeight: 1.5 }}>
                  {selectedItem.question}
                </p>

                <div className="compare-box" style={{ marginBottom: '14px' }}>
                  <div className="compare-item">
                    <span className="ans-label">Đáp án đúng (Ground Truth):</span>
                    <span className="ans-gt" style={{ fontSize: '15px' }}>{selectedItem.answer}</span>
                  </div>
                  <div className="compare-item">
                    <span className="ans-label">Mô hình dự đoán:</span>
                    <span className={selectedItem.is_correct ? 'ans-pred-ok' : 'ans-pred-wrong'} style={{ fontSize: '15px' }}>
                      {selectedItem.output}
                    </span>
                  </div>
                </div>

                {!selectedItem.is_correct && (
                  <div className="reason-box" style={{ marginBottom: '14px' }}>
                    <div className="reason-title">Phân tích cơ chế: {selectedItem.failure_type}</div>
                    <div className="reason-desc">{selectedItem.failure_desc}</div>
                  </div>
                )}

                <div style={{ marginBottom: '16px' }}>
                  <div style={{ fontSize: '12px', fontWeight: 700, marginBottom: '6px' }}>Các lựa chọn (Options):</div>
                  <div className="options-wrap">
                    {selectedItem.options.map((opt, idx) => (
                      <span
                        key={idx}
                        className={`opt-pill ${opt.toLowerCase() === selectedItem.answer.toLowerCase() ? 'gt-match' : ''}`}
                      >
                        {opt}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Comments in Modal */}
                <div className="comments-section" style={{ borderTop: '1px solid var(--border-color)', paddingTop: '12px' }}>
                  <div style={{ fontSize: '13px', fontWeight: 700, marginBottom: '8px' }}>
                    Nhận xét cho mẫu này ({(commentsBySample[selectedItem.id] || []).length}):
                  </div>

                  <div className="comments-list" style={{ maxHeight: '160px' }}>
                    {(commentsBySample[selectedItem.id] || []).map(c => (
                      <div key={c.id} className="comment-bubble">
                        <div className="comment-meta">
                          <span className="comment-author">[{c.user_id}]</span>
                          <span>{new Date(c.created_at).toLocaleString('vi-VN')}</span>
                        </div>
                        <p className="comment-text">{c.comment}</p>
                      </div>
                    ))}
                  </div>

                  <div className="comment-form" style={{ marginTop: '8px' }}>
                    <textarea
                      className="comment-textarea"
                      placeholder={`Viết nhận xét của ${userId}...`}
                      value={commentInputs[selectedItem.id] || ''}
                      onChange={(e) => setCommentInputs(prev => ({ ...prev, [selectedItem.id]: e.target.value }))}
                    />
                    <button
                      className="btn-submit-comment"
                      onClick={() => handleAddComment(selectedItem.id)}
                    >
                      Gửi nhận xét
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
