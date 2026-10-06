import os
import re
import tinycss2
from bs4 import BeautifulSoup

def prefix_rule_prelude(prelude_tokens, prefix):
    selectors = []
    current = []
    for token in prelude_tokens:
        if token.type == "literal" and token.value == ",":
            selectors.append(current)
            current = []
        else:
            current.append(token)
    if current:
        selectors.append(current)
    
    new_tokens = []
    for i, sel in enumerate(selectors):
        sel_text = tinycss2.serialize(sel).strip()
        if sel_text in [":root", "html"]:
            new_sel_text = sel_text
        elif sel_text == "body":
            new_sel_text = prefix
        elif sel_text == "*":
            new_sel_text = f"{prefix} *"
        elif sel_text.startswith("@"):
            new_sel_text = sel_text
        elif sel_text.startswith(":"):
            new_sel_text = f"{prefix}{sel_text}"
        else:
            new_sel_text = f"{prefix} {sel_text}"
        
        parsed = tinycss2.parse_component_value_list(new_sel_text)
        new_tokens.extend(parsed)
        if i < len(selectors) - 1:
            new_tokens.append(tinycss2.ast.LiteralToken(1, 1, ","))
            new_tokens.append(tinycss2.ast.WhitespaceToken(1, 1, " "))
    return new_tokens

def scope_css(css_str, prefix):
    rules = tinycss2.parse_stylesheet(css_str, skip_whitespace=True)
    for r in rules:
        if r.type == "qualified-rule":
            r.prelude = prefix_rule_prelude(r.prelude, prefix)
        elif r.type == "at-rule" and r.lower_at_keyword == "media" and r.content:
            inner_rules = tinycss2.parse_rule_list(r.content, skip_whitespace=True)
            for ir in inner_rules:
                if ir.type == "qualified-rule":
                    ir.prelude = prefix_rule_prelude(ir.prelude, prefix)
            r.content = tinycss2.parse_component_value_list(tinycss2.serialize(inner_rules))
    return tinycss2.serialize(rules)

def extract_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    styles = "\n".join([s.get_text() for s in soup.find_all("style")])
    
    body_nodes = []
    if soup.body:
        for ch in soup.body.children:
            if getattr(ch, "name", None) == "script":
                continue
            body_nodes.append(str(ch))
    body_html = "".join(body_nodes)
    return styles, body_html

print("Extracting 4 files...")
s_survey, b_survey = extract_file("visualizations/spatial_mqa_27_papers_survey.html")
s_review, b_review = extract_file("visualizations/spatial_mqa_peer_review_dashboard.html")
s_depth, b_depth = extract_file("visualizations/spatial_mllm_survey_limitations.html")
s_base, b_base = extract_file("visualizations/baselines_workflow.html")

print("Scoping styles...")
scoped_survey = scope_css(s_survey, "#mod-survey")
scoped_review = scope_css(s_review, "#mod-review")
scoped_depth = scope_css(s_depth, "#mod-depth")
scoped_base = scope_css(s_base, "#mod-baselines")

# Adjust in-page links inside b_survey and b_review
b_survey = b_survey.replace(
    'href="spatial_mqa_peer_review_dashboard.html"',
    'href="javascript:void(0)" onclick="switchMasterModule(\'mod-review\')"'
).replace(
    'href="spatial_mllm_survey_limitations.html"',
    'href="javascript:void(0)" onclick="switchMasterModule(\'mod-depth\')"'
)

b_review = b_review.replace(
    'href="spatial_mqa_27_papers_survey.html"',
    'href="javascript:void(0)" onclick="switchMasterModule(\'mod-survey\')"'
).replace(
    'href="spatial_mllm_survey_limitations.html"',
    'href="javascript:void(0)" onclick="switchMasterModule(\'mod-depth\')"'
).replace(
    'href="baselines_workflow.html"',
    'href="javascript:void(0)" onclick="switchMasterModule(\'mod-baselines\')"'
)

# In Module 4 (baselines), ensure header doesn't hide behind sticky master header
scoped_base += """
#mod-baselines header {
  position: relative !important;
  top: auto !important;
  box-shadow: none !important;
}
"""

master_css = f"""
/* ========================================================
   MASTER UNIFIED DASHBOARD STYLES
   ======================================================== */
:root {{
  --bg-master-dark: #07090e;
  --bg-master-nav: rgba(10, 13, 20, 0.88);
  --border-master: rgba(255, 255, 255, 0.1);
  --border-master-glow: rgba(56, 189, 248, 0.4);
  --accent-cyan: #38bdf8;
  --accent-purple: #a855f7;
  --accent-emerald: #10b981;
  --accent-amber: #f59e0b;
  --accent-rose: #f43f5e;
  --text-primary: #f8fafc;
  --text-muted: #94a3b8;
}}

* {{
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}}

html {{
  scroll-behavior: smooth;
  scroll-padding-top: 130px;
}}

body {{
  background-color: var(--bg-master-dark);
  color: var(--text-primary);
  font-family: 'Outfit', 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  min-height: 100vh;
  line-height: 1.6;
  overflow-x: hidden;
}}

/* Sticky Master Navigation Header */
#master-portal-header {{
  position: sticky;
  top: 0;
  z-index: 9999;
  background: var(--bg-master-nav);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border-master);
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
  padding: 0.75rem 1.5rem 0.5rem 1.5rem;
}}

.master-header-container {{
  max-width: 1440px;
  margin: 0 auto;
}}

.master-brand-row {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1.5rem;
  margin-bottom: 0.75rem;
  flex-wrap: wrap;
}}

.master-brand-left {{
  display: flex;
  align-items: center;
  gap: 0.85rem;
}}

.master-logo-badge {{
  background: linear-gradient(135deg, #38bdf8, #818cf8);
  color: #07090e;
  font-weight: 900;
  font-size: 0.95rem;
  letter-spacing: 0.05em;
  padding: 0.45rem 0.75rem;
  border-radius: 10px;
  box-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
  font-family: 'JetBrains Mono', monospace;
  white-space: nowrap;
}}

.master-portal-title {{
  font-size: 1.25rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  background: linear-gradient(to right, #ffffff, #cbd5e1);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}}

.master-portal-subtitle {{
  font-size: 0.8rem;
  color: var(--text-muted);
  font-weight: 400;
}}

.master-stats-strip {{
  display: flex;
  align-items: center;
  gap: 0.6rem;
  flex-wrap: wrap;
}}

.master-stat-chip {{
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.3rem 0.65rem;
  border-radius: 8px;
  font-size: 0.75rem;
  font-family: 'JetBrains Mono', monospace;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid var(--border-master);
}}

.master-stat-chip .chip-num {{
  font-weight: 700;
}}

.master-stat-chip.cyan .chip-num {{ color: var(--accent-cyan); }}
.master-stat-chip.rose .chip-num {{ color: var(--accent-rose); }}
.master-stat-chip.amber .chip-num {{ color: var(--accent-amber); }}
.master-stat-chip.emerald .chip-num {{ color: var(--accent-emerald); }}
.master-stat-chip .chip-lbl {{ color: #94a3b8; font-size: 0.72rem; }}

/* Master Tabs Bar */
.master-nav-tabs {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.6rem;
  background: rgba(15, 23, 42, 0.65);
  padding: 0.35rem;
  border-radius: 12px;
  border: 1px solid var(--border-master);
}}

@media (max-width: 992px) {{
  .master-nav-tabs {{
    grid-template-columns: repeat(2, 1fr);
  }}
}}

@media (max-width: 576px) {{
  .master-nav-tabs {{
    grid-template-columns: 1fr;
  }}
}}

.master-tab-btn {{
  background: transparent;
  border: 1px solid transparent;
  border-radius: 9px;
  padding: 0.55rem 0.85rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  text-align: left;
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
  color: #94a3b8;
  font-family: inherit;
}}

.master-tab-btn:hover {{
  background: rgba(255, 255, 255, 0.05);
  color: #f1f5f9;
  border-color: rgba(255, 255, 255, 0.1);
}}

.master-tab-btn.active {{
  background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.95));
  border-color: var(--accent-cyan);
  box-shadow: 0 4px 20px rgba(56, 189, 248, 0.25), inset 0 0 12px rgba(56, 189, 248, 0.1);
  color: #ffffff;
}}

.master-tab-btn .mtab-num {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  font-weight: 700;
  color: #64748b;
  padding: 0.2rem 0.4rem;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 6px;
}}

.master-tab-btn.active .mtab-num {{
  color: var(--accent-cyan);
  background: rgba(56, 189, 248, 0.15);
}}

.master-tab-btn .mtab-icon {{
  font-size: 1.25rem;
  line-height: 1;
}}

.master-tab-btn .mtab-content {{
  display: flex;
  flex-direction: column;
  min-width: 0;
}}

.master-tab-btn .mtab-title {{
  font-size: 0.85rem;
  font-weight: 700;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}

.master-tab-btn .mtab-desc {{
  font-size: 0.7rem;
  color: #64748b;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}

.master-tab-btn.active .mtab-desc {{
  color: #94a3b8;
}}

/* Master Module Containers */
.master-module {{
  display: none;
  animation: modFadeIn 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}}

.master-module.active {{
  display: block;
}}

@keyframes modFadeIn {{
  from {{
    opacity: 0;
    transform: translateY(8px);
  }}
  to {{
    opacity: 1;
    transform: translateY(0);
  }}
}}

/* Quick Floating Bar */
.master-floating-nav {{
  position: fixed;
  bottom: 1.5rem;
  right: 1.5rem;
  z-index: 999;
  display: flex;
  gap: 0.5rem;
  background: rgba(10, 13, 20, 0.85);
  backdrop-filter: blur(12px);
  padding: 0.4rem;
  border-radius: 30px;
  border: 1px solid var(--border-master);
  box-shadow: 0 10px 25px rgba(0,0,0,0.5);
}}

.master-floating-btn {{
  background: transparent;
  border: none;
  color: #94a3b8;
  padding: 0.45rem 0.75rem;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  gap: 0.35rem;
}}

.master-floating-btn:hover {{
  color: #fff;
  background: rgba(255,255,255,0.1);
}}

.master-floating-btn.to-top {{
  background: rgba(56, 189, 248, 0.15);
  color: var(--accent-cyan);
  border: 1px solid rgba(56, 189, 248, 0.3);
}}

/* Scoped Module CSS */
{scoped_survey}
{scoped_review}
{scoped_depth}
{scoped_base}
"""

master_js = """
// ========================================================
// MASTER UNIFIED DASHBOARD JAVASCRIPT
// ========================================================

// Master Module Switcher
function switchMasterModule(targetModId, clickedBtn) {
  const modules = document.querySelectorAll('.master-module');
  const buttons = document.querySelectorAll('.master-nav-tabs .master-tab-btn');

  modules.forEach(m => {
    m.classList.remove('active');
  });

  const targetMod = document.getElementById(targetModId);
  if (targetMod) {
    targetMod.classList.add('active');
  }

  buttons.forEach(b => {
    if (b.getAttribute('data-mod') === targetModId || b === clickedBtn) {
      b.classList.add('active');
    } else {
      b.classList.remove('active');
    }
  });

  // Update URL Hash without scroll jump
  const hashKey = targetModId.replace('mod-', '');
  if (window.history && window.history.replaceState) {
    window.history.replaceState(null, null, '#' + hashKey);
  } else {
    window.location.hash = hashKey;
  }

  // Module 3: Refresh SVG simulation
  if (targetModId === 'mod-depth' && typeof window.updateSimulation === 'function') {
    window.updateSimulation();
  }

  // Module 4: Refresh Simulator
  if (targetModId === 'mod-baselines' && typeof window.updateSimulator === 'function') {
    window.updateSimulator();
  }

  // Render KaTeX for newly visible module
  if (window.renderMathInElement && targetMod) {
    renderMathInElement(targetMod, {
      delimiters: [
        {left: '$$', right: '$$', display: true},
        {left: '$', right: '$', display: false}
      ]
    });
  }

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Router initialization
function initMasterRouter() {
  const hash = window.location.hash.replace('#', '').toLowerCase();
  const map = {
    'survey': 'mod-survey',
    'papers': 'mod-survey',
    'review': 'mod-review',
    'rebuttal': 'mod-review',
    'depth': 'mod-depth',
    'limitations': 'mod-depth',
    'baselines': 'mod-baselines',
    'workflow': 'mod-baselines'
  };

  const targetModId = map[hash] || 'mod-survey';
  const targetBtn = document.querySelector(`.master-tab-btn[data-mod="${targetModId}"]`);
  switchMasterModule(targetModId, targetBtn);
}

// Module 1 Script
(function() {
  const mod = document.getElementById('mod-survey');
  if (!mod) return;
  const navBtns = mod.querySelectorAll('.nav-btn');
  const tabContents = mod.querySelectorAll('.tab-content');

  navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      navBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(tc => tc.classList.remove('active'));

      btn.classList.add('active');
      const targetTab = mod.querySelector('#' + btn.dataset.tab);
      if (targetTab) {
        targetTab.classList.add('active');
        if (window.renderMathInElement) {
          renderMathInElement(targetTab, {
            delimiters: [{left: '$$', right: '$$', display: true}, {left: '$', right: '$', display: false}]
          });
        }
      }
    });
  });

  const filterChips = mod.querySelectorAll('.filter-chip');
  const searchInput = mod.querySelector('#pSearch');
  const tableRows = mod.querySelectorAll('#tbl27 tbody tr');

  function filter27Table() {
    const activeChip = mod.querySelector('.filter-chip.active');
    const activeFilter = activeChip ? activeChip.dataset.f : 'all';
    const query = searchInput ? searchInput.value.toLowerCase() : '';

    tableRows.forEach(row => {
      const cat = row.dataset.c || '';
      const text = row.textContent.toLowerCase();
      let matchesCat = false;
      
      if (activeFilter === 'all') {
        matchesCat = true;
      } else if (activeFilter === 'key') {
        matchesCat = cat.includes('key');
      } else {
        matchesCat = cat.includes(activeFilter);
      }

      const matchesSearch = text.includes(query);

      if (matchesCat && matchesSearch) {
        row.style.display = '';
      } else {
        row.style.display = 'none';
      }
    });
  }

  filterChips.forEach(chip => {
    chip.addEventListener('click', () => {
      filterChips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      filter27Table();
    });
  });

  if (searchInput) {
    searchInput.addEventListener('input', filter27Table);
  }
})();

// Module 2 Script
(function() {
  const mod = document.getElementById('mod-review');
  if (!mod) return;
  const navBtns = mod.querySelectorAll('.nav-btn');
  const tabContents = mod.querySelectorAll('.tab-content');

  navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      navBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(tc => tc.classList.remove('active'));

      btn.classList.add('active');
      const targetTab = mod.querySelector('#' + btn.dataset.tab);
      if (targetTab) {
        targetTab.classList.add('active');
        if (window.renderMathInElement) {
          renderMathInElement(targetTab, {
            delimiters: [{left: '$$', right: '$$', display: true}, {left: '$', right: '$', display: false}]
          });
        }
      }
    });
  });
})();

// Module 3 Script
(function() {
  const mod = document.getElementById('mod-depth');
  if (!mod) return;

  const rotSlider = mod.querySelector('#rotSlider');
  const degLabel = mod.querySelector('#degLabel');
  const headingArrow = mod.querySelector('#headingArrow');
  const objAns = mod.querySelector('#objAns');
  const camAns = mod.querySelector('#camAns');

  window.updateSimulation = function() {
    if (!rotSlider || !degLabel || !headingArrow || !objAns || !camAns) return;
    const deg = parseInt(rotSlider.value);
    degLabel.textContent = `${deg}°`;
    headingArrow.setAttribute('transform', `rotate(${deg})`);

    const targetAngle = 55; 
    let relAngle = (targetAngle - deg) % 360;
    if (relAngle < 0) relAngle += 360;

    let ans = "";
    let color = "var(--rose)";

    if (relAngle >= 315 || relAngle < 45) {
      ans = "IN FRONT OF (Ở ĐẰNG TRƯỚC)";
      color = "var(--emerald)";
    } else if (relAngle >= 45 && relAngle < 135) {
      ans = "RIGHT OF (Ở BÊN PHẢI)";
      color = "var(--cyan)";
    } else if (relAngle >= 135 && relAngle < 225) {
      ans = "BEHIND (Ở ĐẰNG SAU)";
      color = "var(--amber)";
    } else {
      ans = "LEFT OF (Ở BÊN TRÁI)";
      color = "var(--purple)";
    }

    objAns.textContent = ans;
    objAns.style.color = color;
    camAns.textContent = "RIGHT OF (BÊN PHẢI)";
  };

  if (rotSlider) {
    rotSlider.addEventListener('input', window.updateSimulation);
    window.updateSimulation();
  }

  const filterBtns = mod.querySelectorAll('.filter-btn');
  const tableRows = mod.querySelectorAll('#papersTable tbody tr');
  const searchInput = mod.querySelector('#paperSearch');

  function filterTable() {
    const activeBtn = mod.querySelector('.filter-btn.active');
    const activeFilter = activeBtn ? activeBtn.dataset.filter : 'all';
    const query = searchInput ? searchInput.value.toLowerCase() : '';

    tableRows.forEach(row => {
      const cat = row.dataset.cat;
      const text = row.textContent.toLowerCase();
      const matchesCat = (activeFilter === 'all') || (cat === activeFilter);
      const matchesSearch = text.includes(query);

      if (matchesCat && matchesSearch) {
        row.style.display = '';
      } else {
        row.style.display = 'none';
      }
    });
  }

  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      filterTable();
    });
  });

  if (searchInput) {
    searchInput.addEventListener('input', filterTable);
  }
})();

// Module 4 Script
(function() {
  const mod = document.getElementById('mod-baselines');
  if (!mod) return;

  const scenarios = [
    {
      id: "sample_0",
      type: "Q2 (First-person / Ego-centric)",
      question: "If you were the giraffe in the picture, where is the sun relative to you?",
      options: ["in front of", "behind", "left of", "right of", "above", "below"],
      answer: "right of",
      llava_zero: "left of (Wrong - Frame Inversion)",
      space_zero: "behind (Wrong - Projection Ambiguity)",
      llava_lora: "left of (Wrong - Frame Inversion)",
      space_lora: "right of (Correct)"
    },
    {
      id: "sample_1",
      type: "Q1 (Camera-centric / Depth Axis Ay)",
      question: "Is the chair closer to or farther from the camera than the dining table?",
      options: ["closer to", "farther from"],
      answer: "closer to",
      llava_zero: "farther from (Wrong - Ay Depth Collapse)",
      space_zero: "closer to (Correct)",
      llava_lora: "farther from (Wrong - Ay Depth Collapse)",
      space_lora: "closer to (Correct)"
    },
    {
      id: "sample_2",
      type: "Q3 (Third-person in-image landmark)",
      question: "Where is the bicycle relative to the fire hydrant?",
      options: ["to the left of", "to the right of", "in front of", "behind"],
      answer: "to the left of",
      llava_zero: "to the right of (Wrong)",
      space_zero: "to the left of (Correct)",
      llava_lora: "to the left of (Correct)",
      space_lora: "to the left of (Correct)"
    },
    {
      id: "sample_3",
      type: "Q2 (First-person / Metric Distance)",
      question: "Relative to the driver in the red car, where is the stop sign?",
      options: ["in front of", "behind", "left of", "right of"],
      answer: "in front of",
      llava_zero: "right of (Wrong - Viewpoint Inversion)",
      space_zero: "in front of (Correct)",
      llava_lora: "behind (Wrong)",
      space_lora: "in front of (Correct)"
    }
  ];

  window.updateSimulator = function() {
    const sel = document.getElementById("sampleSelect");
    if (!sel) return;
    const idx = sel.value;
    const s = scenarios[idx];
    if (!s) return;
    
    const formattedPrompt = 
`[SYSTEM]
You are currently a senior expert in spatial relation reasoning.
Given an Image, a Question and Options, your task is to answer the correct spatial relation. Note that you only need to choose one option from the all options without explaining any reason.

[USER]
Image: <image>
Question: ${s.question}
Options: ${s.options.join("; ")}

[ASSISTANT]
Output:`;

    const pDisplay = document.getElementById("promptDisplay");
    const predDisplay = document.getElementById("predictionDisplay");
    if (pDisplay) pDisplay.innerText = formattedPrompt;
    if (predDisplay) {
      predDisplay.innerHTML = 
`Target Perspective: <span style="color: #a5b4fc; font-weight: bold;">${s.type}</span>
Target Ground Truth Answer: <span style="color: #34d399; font-weight: bold;">>> ${s.answer} <<</span>

(Click "Run Simulated Comparison" to view model responses)`;
    }
  };

  window.simulateInference = function() {
    const sel = document.getElementById("sampleSelect");
    if (!sel) return;
    const idx = sel.value;
    const s = scenarios[idx];
    if (!s) return;

    const html = 
`Target Perspective: <span style="color: #a5b4fc; font-weight: bold;">${s.type}</span>
Target Ground Truth Answer: <span style="color: #34d399; font-weight: bold;">>> ${s.answer} <<</span>

--------------------------------------------------
1. LLaVA-1.5 (Zero-Shot Direct):
   Prediction: ${s.llava_zero.includes("Correct") ? '<span style="color:#34d399">✓</span>' : '<span style="color:#f87171">✗</span>'} ${s.llava_zero}

2. SpaceLLaVA (Zero-Shot Direct):
   Prediction: ${s.space_zero.includes("Correct") ? '<span style="color:#34d399">✓</span>' : '<span style="color:#f87171">✗</span>'} ${s.space_zero}

3. LLaVA-1.5 (LoRA Fine-Tuned):
   Prediction: ${s.llava_lora.includes("Correct") ? '<span style="color:#34d399">✓</span>' : '<span style="color:#f87171">✗</span>'} ${s.llava_lora}

4. SpaceLLaVA (LoRA Fine-Tuned):
   Prediction: ${s.space_lora.includes("Correct") ? '<span style="color:#34d399">✓</span>' : '<span style="color:#f87171">✗</span>'} ${s.space_lora}
--------------------------------------------------
Diagnostic Note: SpaceLLaVA leverages SpatialVLM 3D depth priors to resolve depth & perspective more reliably than pure 2D LLaVA.`;

    const predDisplay = document.getElementById("predictionDisplay");
    if (predDisplay) predDisplay.innerHTML = html;
  };

  const sampleSel = document.getElementById("sampleSelect");
  if (sampleSel) {
    sampleSel.addEventListener("change", window.updateSimulator);
  }
  window.updateSimulator();
})();

window.addEventListener('DOMContentLoaded', initMasterRouter);
window.addEventListener('hashchange', initMasterRouter);
"""

full_html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Spatial Intelligence in MLLMs — Unified Research & Survey Portal (SpatialMQA ACL 2025)</title>
  
  <!-- Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  
  <!-- KaTeX CSS & JS for academic math rendering -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}]}});"></script>

  <style>
{master_css}
  </style>
</head>
<body>

  <!-- Sticky Master Portal Header -->
  <header id="master-portal-header">
    <div class="master-header-container">
      <div class="master-brand-row">
        <div class="master-brand-left">
          <div class="master-logo-badge">3D-MLLM</div>
          <div>
            <h1 class="master-portal-title">Khảo cứu Toàn diện & Hệ thống Nghiên cứu SpatialMQA</h1>
            <p class="master-portal-subtitle">Unified Research Portal · 27+ Papers Literature · Academic Peer Review · Depth Fallacy Proof · Baselines Architecture</p>
          </div>
        </div>
        <div class="master-stats-strip">
          <div class="master-stat-chip cyan">
            <span class="chip-num">27+</span>
            <span class="chip-lbl">Papers Khảo cứu</span>
          </div>
          <div class="master-stat-chip rose">
            <span class="chip-num">38.2%</span>
            <span class="chip-lbl">Lỗi FRS Shift</span>
          </div>
          <div class="master-stat-chip amber">
            <span class="chip-num">29.6%</span>
            <span class="chip-lbl">Depth LLaVA (Ay)</span>
          </div>
          <div class="master-stat-chip emerald">
            <span class="chip-num">≥ 56%</span>
            <span class="chip-lbl">Mục tiêu Đề tài</span>
          </div>
        </div>
      </div>
      <nav class="master-nav-tabs">
        <button class="master-tab-btn active" data-mod="mod-survey" onclick="switchMasterModule('mod-survey', this)">
          <span class="mtab-num">01</span>
          <span class="mtab-icon">📚</span>
          <span class="mtab-content">
            <span class="mtab-title">Khảo cứu 27+ Papers</span>
            <span class="mtab-desc">Phân loại 4 Trường phái & Ma trận Literature</span>
          </span>
        </button>
        <button class="master-tab-btn" data-mod="mod-review" onclick="switchMasterModule('mod-review', this)">
          <span class="mtab-num">02</span>
          <span class="mtab-icon">⚖️</span>
          <span class="mtab-content">
            <span class="mtab-title">Phản biện & Hiệu chỉnh</span>
            <span class="mtab-desc">Đánh giá R1-R3, Diagnostic Interventions & 15 Lỗi</span>
          </span>
        </button>
        <button class="master-tab-btn" data-mod="mod-depth" onclick="switchMasterModule('mod-depth', this)">
          <span class="mtab-num">03</span>
          <span class="mtab-icon">📐</span>
          <span class="mtab-content">
            <span class="mtab-title">Giới hạn Depth & Mô phỏng</span>
            <span class="mtab-desc">Bằng chứng Depth Fallacy & Giraffe Heading Slider</span>
          </span>
        </button>
        <button class="master-tab-btn" data-mod="mod-baselines" onclick="switchMasterModule('mod-baselines', this)">
          <span class="mtab-num">04</span>
          <span class="mtab-icon">🔬</span>
          <span class="mtab-content">
            <span class="mtab-title">Baselines LLaVA vs SpaceLLaVA</span>
            <span class="mtab-desc">Quy trình LoRA & Simulator Suy luận Mẫu</span>
          </span>
        </button>
      </nav>
    </div>
  </header>

  <!-- MODULE 1: Khảo cứu 27+ Papers -->
  <section id="mod-survey" class="master-module active">
{b_survey}
  </section>

  <!-- MODULE 2: Phản biện Học thuật & Hiệu chỉnh -->
  <section id="mod-review" class="master-module">
{b_review}
  </section>

  <!-- MODULE 3: Giới hạn Depth & Mô phỏng 3D -->
  <section id="mod-depth" class="master-module">
{b_depth}
  </section>

  <!-- MODULE 4: Baselines Workflow & Simulator -->
  <section id="mod-baselines" class="master-module">
{b_base}
  </section>

  <!-- Quick Floating Action Bar -->
  <div class="master-floating-nav">
    <button class="master-floating-btn" onclick="switchMasterModule('mod-survey')">📚 Khảo cứu</button>
    <button class="master-floating-btn" onclick="switchMasterModule('mod-review')">⚖️ Phản biện</button>
    <button class="master-floating-btn" onclick="switchMasterModule('mod-depth')">📐 Depth Fallacy</button>
    <button class="master-floating-btn" onclick="switchMasterModule('mod-baselines')">🔬 Baselines</button>
    <button class="master-floating-btn to-top" onclick="window.scrollTo({{top: 0, behavior: 'smooth'}})">↑ Đầu trang</button>
  </div>

  <script>
{master_js}
  </script>
</body>
</html>
"""

# Write to visualizations/survey.html
out_vis = "visualizations/survey.html"
with open(out_vis, "w", encoding="utf-8") as f:
    f.write(full_html)
print(f"Generated {out_vis} ({len(full_html):,} bytes)")

# Write to survey.html at root as well
out_root = "survey.html"
with open(out_root, "w", encoding="utf-8") as f:
    f.write(full_html)
print(f"Generated {out_root} ({len(full_html):,} bytes)")
