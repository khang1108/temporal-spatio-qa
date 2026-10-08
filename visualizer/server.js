import express from 'express';
import cors from 'cors';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { DatabaseSync } from 'node:sqlite';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// Initialize SQLite database
const dbDir = path.join(__dirname, 'database');
if (!fs.existsSync(dbDir)) {
  fs.mkdirSync(dbDir, { recursive: true });
}

const dbPath = path.join(dbDir, 'comments.db');
const db = new DatabaseSync(dbPath);

// Create table if not exists
db.exec(`
  CREATE TABLE IF NOT EXISTS comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id INTEGER NOT NULL,
    user_id TEXT NOT NULL,
    comment TEXT NOT NULL,
    failure_tag TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  CREATE INDEX IF NOT EXISTS idx_sample_id ON comments(sample_id);
`);

console.log(`[SQLite] Database initialized at ${dbPath}`);

// API: Get all comments (or by sample_id)
app.get('/api/comments', (req, res) => {
  try {
    const { sample_id } = req.query;
    let query = 'SELECT id, sample_id, user_id, comment, failure_tag, created_at FROM comments';
    let params = [];
    
    if (sample_id) {
      query += ' WHERE sample_id = ? ORDER BY created_at DESC';
      params.push(Number(sample_id));
    } else {
      query += ' ORDER BY created_at DESC';
    }
    
    const stmt = db.prepare(query);
    const rows = stmt.all(...params);
    res.json({ success: true, count: rows.length, comments: rows });
  } catch (err) {
    console.error('Error fetching comments:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// API: Post a new comment
app.post('/api/comments', (req, res) => {
  try {
    const { sample_id, user_id, comment, failure_tag } = req.body;
    if (!sample_id || !user_id || !comment) {
      return res.status(400).json({ success: false, error: 'sample_id, user_id, and comment are required' });
    }

    const cleanUserId = String(user_id).trim();
    const cleanComment = String(comment).trim();
    const cleanTag = failure_tag ? String(failure_tag).trim() : null;

    if (!cleanUserId || !cleanComment) {
      return res.status(400).json({ success: false, error: 'user_id and comment cannot be empty' });
    }

    const stmt = db.prepare(`
      INSERT INTO comments (sample_id, user_id, comment, failure_tag)
      VALUES (?, ?, ?, ?)
    `);
    
    stmt.run(Number(sample_id), cleanUserId, cleanComment, cleanTag);
    
    // Fetch inserted comment
    const lastRow = db.prepare('SELECT * FROM comments WHERE id = last_insert_rowid()').get();
    res.json({ success: true, comment: lastRow });
  } catch (err) {
    console.error('Error adding comment:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// API: Delete a comment (optional)
app.delete('/api/comments/:id', (req, res) => {
  try {
    const { id } = req.params;
    const stmt = db.prepare('DELETE FROM comments WHERE id = ?');
    stmt.run(Number(id));
    res.json({ success: true, message: `Deleted comment #${id}` });
  } catch (err) {
    console.error('Error deleting comment:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// API: Export all comments as JSON or summary
app.get('/api/export', (req, res) => {
  try {
    const rows = db.prepare('SELECT * FROM comments ORDER BY sample_id ASC, created_at ASC').all();
    res.json({
      export_time: new Date().toISOString(),
      total_comments: rows.length,
      data: rows
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// Serve static frontend build if dist exists
const distPath = path.join(__dirname, 'dist');
if (fs.existsSync(distPath)) {
  app.use(express.static(distPath));
  app.use((req, res, next) => {
    if (req.method === 'GET' && !req.path.startsWith('/api')) {
      return res.sendFile(path.join(distPath, 'index.html'));
    }
    next();
  });
}

app.listen(PORT, '0.0.0.0', () => {
  console.log(`====================================================`);
  console.log(`SpatialMQA Failure Visualizer & Annotation Server`);
  console.log(`Listening on http://0.0.0.0:${PORT}`);
  console.log(`SQLite database: ${dbPath}`);
  console.log(`====================================================`);
});
