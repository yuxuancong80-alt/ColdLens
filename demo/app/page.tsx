"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  simulate,
  simulateFromHistory,
  type HistoryCatalogItem,
  type MatchedTerm,
  type Sandbox,
  type SandboxPreset,
  type SimulationResult,
} from "./simulation";

type Recommendation = {
  rank: number;
  title: string;
  score: number;
  is_validation_positive: boolean;
  matched_terms: MatchedTerm[];
};

type DemoSample = {
  id: string;
  history_bucket: string;
  history_count: number;
  sample_outcome: "hit" | "miss";
  history_titles: string[];
  validation_positive_titles: string[];
  recommendations: Recommendation[];
};

type DemoData = {
  demo: string;
  evaluated_split: string;
  test_split_read: boolean;
  model: string;
  overall_validation: {
    users: number;
    candidate_items: number;
    positive_pairs: number;
    recall_at_10: number;
    hit_rate_at_10: number;
  };
  selection_note: string;
  interpretation_boundary: string;
  samples: DemoSample[];
  sandbox: Sandbox;
};

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function policyFor(historyCount: number) {
  if (historyCount >= 4) {
    return {
      title: "Text 冷启动首批验证人群",
      description: "历史信号相对充分，可进入 Text 冷启动分支；仍需与现有策略做线上小流量对照。",
      level: "standard",
    };
  }
  return {
    title: "低置信 Text + 探索兜底",
    description: "历史只有 1–3 条，离线表现明显更弱。建议降低 Text 权重并保留探索来源。",
    level: "caution",
  };
}

function ExperimentRecommendations({ sample }: { sample: DemoSample }) {
  return (
    <ol className="recommendation-list">
      {sample.recommendations.map((item) => (
        <li
          key={`${sample.id}-recommendation-${item.rank}`}
          className={item.is_validation_positive ? "is-positive" : ""}
        >
          <span className="rank">{String(item.rank).padStart(2, "0")}</span>
          <div className="recommendation-copy">
            <div className="title-row">
              <h3>{item.title}</h3>
              {item.is_validation_positive && (
                <span className="hit-label">离线正例</span>
              )}
            </div>
            <div className="term-row">
              {item.matched_terms.length ? (
                item.matched_terms.map((term) => (
                  <span key={term.term}>{term.term}</span>
                ))
              ) : (
                <span className="no-term">无有效共享词</span>
              )}
            </div>
          </div>
          <div className="score-block">
            <span>相似度</span>
            <strong>{item.score.toFixed(3)}</strong>
          </div>
        </li>
      ))}
    </ol>
  );
}

function SimulationRecommendations({ results }: { results: SimulationResult[] }) {
  if (!results.length) {
    return (
      <div className="empty-results">
        <strong>还不能生成推荐</strong>
        <p>请输入模型词表能够识别的英文兴趣词，或先选择一个预设兴趣。</p>
      </div>
    );
  }
  return (
    <ol className="recommendation-list simulation-list">
      {results.map((item) => (
        <li key={`simulation-${item.rank}-${item.title}`}>
          <span className="rank">{String(item.rank).padStart(2, "0")}</span>
          <div className="recommendation-copy">
            <div className="title-row">
              <h3>{item.title}</h3>
            </div>
            <div className="term-row">
              {item.matched_terms.map((term) => (
                <span key={term.term}>{term.term}</span>
              ))}
            </div>
            {item.evidence && (
              <p className="evidence-line">
                <span>主要相关历史</span>
                {item.evidence.title}
              </p>
            )}
          </div>
          <div className="score-block">
            <span>相似度</span>
            <strong>{item.score.toFixed(3)}</strong>
          </div>
        </li>
      ))}
    </ol>
  );
}

const FROZEN_COMPARISON = [
  { label: "Recall@10", text: 0.12551, visual: 0.04543 },
  { label: "NDCG@10", text: 0.07969, visual: 0.028 },
  { label: "HitRate@10", text: 0.15214, visual: 0.058 },
];

function ComparisonPanel() {
  return (
    <section className="comparison-shell">
      <div className="comparison-hero">
        <div>
          <p className="section-kicker">Frozen Test · 只读汇总</p>
          <h2>不是多模态越多越好，<br />而是证据决定去留。</h2>
          <p>
            标题文本在全部冻结核心指标上优于封面视觉；三种融合又没有同时改善事先约定的双指标，因此当前产品方案选择轻量Text兜底。
          </p>
        </div>
        <div className="decision-stamp">
          <span>最终决策</span>
          <strong>Text-only</strong>
          <small>保留视觉证据，不进入主链路</small>
        </div>
      </div>

      <div className="modality-explainer">
        <div className="comparison-heading">
          <div>
            <p className="section-kicker">先理解两种方案</p>
            <h3>它们推荐的都是视频，看的信息不同</h3>
          </div>
          <span className="plain-language-tag">同一个例子</span>
        </div>
        <div className="modality-paths">
          <article className="text-path">
            <div className="modality-title">
              <span className="modality-mark">T</span>
              <div>
                <small>Text</small>
                <h4>标题文本方案</h4>
              </div>
            </div>
            <p><strong>它看什么：</strong>用户过去评论过的视频标题中出现了哪些人物、主题和关键词。</p>
            <div className="example-box">
              <span>历史标题</span>
              <q>Naruto final battle · Anime hero story</q>
            </div>
            <div className="simple-flow" aria-label="文本方案处理路径">
              <span>找出 naruto / battle</span><i>→</i><span>匹配新视频标题</span><i>→</i><strong>生成排序</strong>
            </div>
          </article>
          <article className="visual-path">
            <div className="modality-title">
              <span className="modality-mark">V</span>
              <div>
                <small>Visual</small>
                <h4>封面视觉方案</h4>
              </div>
            </div>
            <p><strong>它看什么：</strong>用户过去评论过的视频封面中有哪些人物、颜色、物体和画面风格。</p>
            <div className="example-box visual-example">
              <span>历史封面</span>
              <q>动漫人物 · 战斗场景 · 相似构图</q>
            </div>
            <div className="simple-flow" aria-label="视觉方案处理路径">
              <span>提取封面特征</span><i>→</i><span>匹配新视频封面</span><i>→</i><strong>生成排序</strong>
            </div>
          </article>
        </div>
        <div className="shared-output">
          <strong>两种方案最终都输出：最值得推荐的新视频Top-10</strong>
          <span>当前Demo的交互推荐使用Text，也就是根据标题进行匹配。</span>
        </div>
      </div>

      <div className="comparison-layout">
        <article className="metric-comparison-card">
          <div className="comparison-heading">
            <div>
              <p className="section-kicker">核心结果</p>
              <h3>Text 与 Visual 的冻结 Test 对比</h3>
            </div>
            <div className="chart-legend" aria-label="图例">
              <span className="text-legend">Text 标题</span>
              <span className="visual-legend">Visual 封面</span>
            </div>
          </div>
          <div className="metric-bars">
            {FROZEN_COMPARISON.map((metric) => {
              const maximum = Math.max(metric.text, metric.visual);
              return (
                <div className="metric-bar-group" key={metric.label}>
                  <div className="metric-label-row">
                    <strong>{metric.label}</strong>
                    <span>绝对差 +{(metric.text - metric.visual).toFixed(5)}</span>
                  </div>
                  <div className="bar-row">
                    <span>标题</span>
                    <div className="bar-track">
                      <i className="text-bar" style={{ width: `${(metric.text / maximum) * 100}%` }} />
                    </div>
                    <strong>{metric.text.toFixed(5)}</strong>
                  </div>
                  <div className="bar-row">
                    <span>封面</span>
                    <div className="bar-track">
                      <i className="visual-bar" style={{ width: `${(metric.visual / maximum) * 100}%` }} />
                    </div>
                    <strong>{metric.visual.toFixed(5)}</strong>
                  </div>
                </div>
              );
            })}
          </div>
          <p className="metric-source">
            来源：已锁定的 final Test 汇总报告。页面不读取 Test 标签，也不根据这些数字继续调参。
          </p>
        </article>

        <aside className="decision-logic-card">
          <p className="section-kicker">为什么不是“视觉无用”</p>
          <h3>有信号，不等于值得上线</h3>
          <p>
            Visual明显高于随机量级，也可能补充部分用户；但现有方法无法稳定识别“什么时候该相信视觉”。若直接上线，会增加特征提取、监控和回退成本，却没有通过效果门槛。
          </p>
          <dl>
            <div>
              <dt>接受门槛</dt>
              <dd>Recall@10与NDCG@10必须同时严格提升</dd>
            </div>
            <div>
              <dt>实际结果</dt>
              <dd>没有一种融合同时满足</dd>
            </div>
            <div>
              <dt>产品动作</dt>
              <dd>Text先做低成本冷启动兜底</dd>
            </div>
          </dl>
        </aside>
      </div>

      <div className="fusion-section">
        <div className="comparison-heading">
          <div>
            <p className="section-kicker">Validation 决策漏斗</p>
            <h3>三种融合为什么都被拒绝</h3>
          </div>
          <span className="rejected-count">3 / 3 rejected</span>
        </div>
        <div className="fusion-card-grid">
          <article>
            <span className="fusion-number">01</span>
            <h4>全局排名融合</h4>
            <p>测试五档Text权重，最优点退化为纯Text；加入任何正视觉权重都降低Top-10核心指标。</p>
            <strong>拒绝：没有增量价值</strong>
          </article>
          <article>
            <span className="fusion-number">02</span>
            <h4>低文本置信门控</h4>
            <p>最低文本相似度没有成功识别“视觉更可靠”的用户，最优切换比例为0%。</p>
            <strong>拒绝：门控信号失效</strong>
          </article>
          <article>
            <span className="fusion-number">03</span>
            <h4>学习式晚融合</h4>
            <p>NDCG@10由0.08211升至0.08283，但Recall@10由0.12797降至0.12717。</p>
            <strong>拒绝：未通过双指标门槛</strong>
          </article>
        </div>
      </div>

      <div className="product-tradeoff">
        <div>
          <p className="section-kicker">产品取舍</p>
          <h3>为什么当前先做 Text</h3>
        </div>
        <div className="tradeoff-grid">
          <div><span>效果证据</span><strong>冻结指标全面领先</strong></div>
          <div><span>接入成本</span><strong>复用标题，CPU即可运行</strong></div>
          <div><span>解释与排障</span><strong>共享词和相似度可追溯</strong></div>
          <div><span>上线策略</span><strong>小流量兜底，不替代成熟主链路</strong></div>
        </div>
        <div className="comparison-boundary">
          <strong>结论边界</strong>
          <p>这些数字验证的是公开评论正反馈下的离线排序能力，不证明CTR、播放时长、留存或线上因果提升。第一评论时间也不等于真实视频发布时间。</p>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const [data, setData] = useState<DemoData | null>(null);
  const [error, setError] = useState("");
  const [selectedId, setSelectedId] = useState("");
  const [mode, setMode] = useState<"experiment" | "simulate" | "comparison">(
    "experiment",
  );
  const [interestInput, setInterestInput] = useState("anime game naruto");
  const [submittedInterest, setSubmittedInterest] = useState("anime game naruto");
  const [activePreset, setActivePreset] = useState("anime-game");
  const [simulationSource, setSimulationSource] = useState<"history" | "text">(
    "history",
  );
  const [selectedHistoryIds, setSelectedHistoryIds] = useState<string[]>([]);
  const [historyTopic, setHistoryTopic] = useState("anime-game");
  const [historySearch, setHistorySearch] = useState("");
  const [interestTermTopic, setInterestTermTopic] = useState("anime-game");

  useEffect(() => {
    let active = true;
    fetch("/demo-data.json")
      .then((response) => {
        if (!response.ok) throw new Error("本地样例尚未生成");
        return response.json() as Promise<DemoData>;
      })
      .then((value) => {
        if (!active) return;
        setData(value);
        setSelectedId(value.samples[0]?.id ?? "");
        const initial = value.sandbox.presets[0];
        if (initial) {
          setInterestInput(initial.keywords);
          setSubmittedInterest(initial.keywords);
          setActivePreset(initial.id);
        }
        setSelectedHistoryIds(
          value.sandbox.history_catalog
            .filter((item, index, items) =>
              index === items.findIndex((other) => other.topic_id === item.topic_id),
            )
            .slice(0, 3)
            .map((item) => item.id),
        );
      })
      .catch(() => {
        if (active) setError("找不到本地 Demo 数据。请先运行样例导出步骤。");
      });
    return () => {
      active = false;
    };
  }, []);

  const sample = useMemo(
    () => data?.samples.find((item) => item.id === selectedId) ?? data?.samples[0],
    [data, selectedId],
  );
  const simulation = useMemo(
    () => (data ? simulate(data.sandbox, submittedInterest) : null),
    [data, submittedInterest],
  );
  const selectedHistory = useMemo(
    () =>
      data?.sandbox.history_catalog.filter((item) =>
        selectedHistoryIds.includes(item.id),
      ) ?? [],
    [data, selectedHistoryIds],
  );
  const historySimulation = useMemo(
    () => (data ? simulateFromHistory(data.sandbox, selectedHistory) : null),
    [data, selectedHistory],
  );
  const visibleHistory = useMemo(() => {
    const query = historySearch.trim().toLowerCase();
    return data?.sandbox.history_catalog.filter((item) =>
      query
        ? item.title.toLowerCase().includes(query)
        : item.topic_id === historyTopic,
    ) ?? [];
  }, [data, historySearch, historyTopic]);

  if (error) {
    return (
      <main className="missing-shell">
        <p className="eyebrow">ColdLens · Local demo</p>
        <h1>需要先生成匿名样例</h1>
        <p>{error}</p>
        <code>.\.venv\Scripts\python.exe scripts\export_demo_data.py</code>
      </main>
    );
  }

  if (!data || !sample || !simulation || !historySimulation) {
    return (
      <main className="loading-shell" aria-live="polite">
        <span className="loading-dot" />
        正在读取本地匿名样例…
      </main>
    );
  }

  const policy = policyFor(sample.history_count);
  const hitCount = sample.recommendations.filter(
    (item) => item.is_validation_positive,
  ).length;

  function choosePreset(preset: SandboxPreset) {
    setInterestInput(preset.keywords);
    setSubmittedInterest(preset.keywords);
    setActivePreset(preset.id);
  }

  function submitInterest(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmittedInterest(interestInput.trim());
    setActivePreset("");
  }

  function toggleHistory(item: HistoryCatalogItem) {
    setSelectedHistoryIds((current) => {
      if (current.includes(item.id)) {
        return current.filter((id) => id !== item.id);
      }
      if (current.length >= 6) return current;
      return [...current, item.id];
    });
  }

  function toggleInterestTerm(term: string) {
    const terms = interestInput.toLowerCase().match(/[a-z0-9]+/g) ?? [];
    const nextTerms = terms.includes(term)
      ? terms.filter((value) => value !== term)
      : [...terms, term];
    const nextInput = nextTerms.join(" ");
    setInterestInput(nextInput);
    setSubmittedInterest(nextInput);
    setActivePreset("");
  }

  const activeSimulation =
    simulationSource === "history" ? historySimulation : simulation;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <span className="brand-mark">CL</span>
          <div>
            <p className="eyebrow">Cold-start recommendation lab</p>
            <h1>ColdLens</h1>
          </div>
        </div>
        <div className="topbar-status">
          <span className="status-dot" />
          Validation候选池 · Test未读取
        </div>
      </header>

      <section className="intro-grid">
        <div className="intro-copy">
          <p className="section-kicker">可解释的冷启动推荐</p>
          <h2>从真实实验，到模拟兴趣，再到方案决策。</h2>
          <p>
            查看匿名Validation用户的离线结果，创建模拟画像，或用冻结证据理解为什么当前产品方案最终选择Text-only。
          </p>
        </div>
        <dl className="metric-strip" aria-label="Validation总体数据">
          <div>
            <dt>冷候选</dt>
            <dd>{data.overall_validation.candidate_items.toLocaleString()}</dd>
          </div>
          <div>
            <dt>实验用户</dt>
            <dd>{data.overall_validation.users.toLocaleString()}</dd>
          </div>
          <div>
            <dt>Recall@10</dt>
            <dd>{percent(data.overall_validation.recall_at_10)}</dd>
          </div>
        </dl>
      </section>

      <nav className="mode-switch" aria-label="Demo模式">
        <button
          type="button"
          aria-pressed={mode === "experiment"}
          className={mode === "experiment" ? "is-active" : ""}
          onClick={() => setMode("experiment")}
        >
          <strong>实验样例</strong>
          <span>有真实Validation正例</span>
        </button>
        <button
          type="button"
          aria-pressed={mode === "simulate"}
          className={mode === "simulate" ? "is-active" : ""}
          onClick={() => setMode("simulate")}
        >
          <strong>创建模拟用户</strong>
          <span>选择历史，实时重新排序</span>
        </button>
        <button
          type="button"
          aria-pressed={mode === "comparison"}
          className={mode === "comparison" ? "is-active" : ""}
          onClick={() => setMode("comparison")}
        >
          <strong>方案对比</strong>
          <span>看懂模型去留决策</span>
        </button>
      </nav>

      {mode === "experiment" ? (
        <>
          <p className="sample-warning">
            8个样例按四个历史分层平衡选择：每层一个命中、一个未命中。它们用于解释模型，不代表总体命中率。
          </p>
          <div className="workspace-grid">
            <aside className="user-panel" aria-label="匿名用户样例">
              <div className="panel-heading">
                <p className="section-kicker">选择样例</p>
                <span>8 users</span>
              </div>
              <div className="user-list">
                {data.samples.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    className={`user-option ${item.id === sample.id ? "is-active" : ""}`}
                    aria-pressed={item.id === sample.id}
                    onClick={() => setSelectedId(item.id)}
                  >
                    <span className="user-avatar">{item.id}</span>
                    <span className="user-copy">
                      <strong>{item.history_bucket}条历史</strong>
                      <small>
                        {item.sample_outcome === "hit"
                          ? "Top-10命中样例"
                          : "Top-10未命中样例"}
                      </small>
                    </span>
                    <span
                      className={`result-mark ${item.sample_outcome}`}
                      aria-hidden="true"
                    >
                      {item.sample_outcome === "hit" ? "✓" : "–"}
                    </span>
                  </button>
                ))}
              </div>
            </aside>

            <section className="profile-panel">
              <div className="profile-header">
                <div>
                  <p className="section-kicker">匿名用户 {sample.id}</p>
                  <h2>{sample.history_count}条训练历史</h2>
                </div>
                <span className={`outcome-badge ${sample.sample_outcome}`}>
                  {sample.sample_outcome === "hit"
                    ? `Top-10命中 ${hitCount} 条`
                    : "Top-10未命中"}
                </span>
              </div>

              <div className="context-grid">
                <article className="context-block">
                  <div className="block-heading">
                    <h3>最近历史</h3>
                    <span>最多展示6条</span>
                  </div>
                  <ol className="history-list">
                    {sample.history_titles.map((title, index) => (
                      <li key={`${sample.id}-history-${index}`}>{title}</li>
                    ))}
                  </ol>
                </article>

                <article className="context-block positive-block">
                  <div className="block-heading">
                    <h3>Validation正例</h3>
                    <span>真实公开评论行为</span>
                  </div>
                  <ul className="positive-list">
                    {sample.validation_positive_titles.map((title, index) => (
                      <li key={`${sample.id}-positive-${index}`}>{title}</li>
                    ))}
                  </ul>
                </article>
              </div>

              <div className="recommendation-heading">
                <div>
                  <p className="section-kicker">Text-only TF-IDF</p>
                  <h2>完整冷候选池中的Top-10</h2>
                </div>
                <p>分数是余弦相似度，不是概率</p>
              </div>
              <ExperimentRecommendations sample={sample} />
            </section>

            <aside className="decision-panel">
              <p className="section-kicker">产品决策</p>
              <span className={`policy-level ${policy.level}`}>
                {sample.history_count >= 4 ? "首批验证" : "谨慎覆盖"}
              </span>
              <h2>{policy.title}</h2>
              <p>{policy.description}</p>

              <div className="decision-rule">
                <span>当前分层</span>
                <strong>{sample.history_bucket}条历史</strong>
              </div>
              <div className="decision-rule">
                <span>离线结果</span>
                <strong>
                  {sample.sample_outcome === "hit" ? "至少命中1条" : "Top-10未命中"}
                </strong>
              </div>
              <div className="decision-rule">
                <span>降级原则</span>
                <strong>不阻塞主推荐链路</strong>
              </div>

              <div className="boundary-note">
                <strong>不要过度解释</strong>
                <p>公开评论不是曝光、点击或观看标签；一个样例命中也不等于线上业务提升。</p>
              </div>
            </aside>
          </div>
        </>
      ) : mode === "simulate" ? (
        <>
          <p className="sample-warning simulation-warning">
            模拟用户没有未来真实互动标签，因此这里只生成推荐和解释，不计算命中率。历史目录是固定演示样本，不代表真实内容分布。
          </p>
          <div className="simulation-grid">
            <section className="profile-panel simulation-panel">
              <div className="simulation-builder">
                <div className="builder-copy">
                  <p className="section-kicker">临时兴趣画像</p>
                  <h2>创建一个模拟用户</h2>
                  <p>推荐方式一：选择1–6条看过的视频，按照正式实验的画像公式生成推荐。也可切换到兴趣词快速体验。</p>
                </div>
                <div className="builder-mode-switch" aria-label="模拟画像方式">
                  <button
                    type="button"
                    className={simulationSource === "history" ? "is-active" : ""}
                    aria-pressed={simulationSource === "history"}
                    onClick={() => setSimulationSource("history")}
                  >
                    选择历史视频 <span>更接近正式实验</span>
                  </button>
                  <button
                    type="button"
                    className={simulationSource === "text" ? "is-active" : ""}
                    aria-pressed={simulationSource === "text"}
                    onClick={() => setSimulationSource("text")}
                  >
                    输入兴趣词 <span>快速近似体验</span>
                  </button>
                </div>

                {simulationSource === "history" ? (
                  <div className="history-builder">
                    <div className="history-toolbar">
                      <div className="topic-tabs" aria-label="历史视频主题">
                        {data.sandbox.presets.map((preset) => (
                          <button
                            key={preset.id}
                            type="button"
                            className={historyTopic === preset.id ? "is-active" : ""}
                            onClick={() => setHistoryTopic(preset.id)}
                          >
                            {preset.label}
                          </button>
                        ))}
                      </div>
                      <input
                        aria-label="搜索历史视频"
                        value={historySearch}
                        placeholder={`搜索全部${data.sandbox.history_catalog.length}条历史`}
                        onChange={(event) => setHistorySearch(event.target.value)}
                      />
                    </div>
                    <div className="history-catalog">
                      {visibleHistory.map((item) => {
                        const selected = selectedHistoryIds.includes(item.id);
                        return (
                          <button
                            key={item.id}
                            type="button"
                            className={selected ? "is-selected" : ""}
                            aria-pressed={selected}
                            disabled={!selected && selectedHistoryIds.length >= 6}
                            onClick={() => toggleHistory(item)}
                          >
                            <span>{selected ? "已选择" : item.topic_label}</span>
                            <strong>{item.title}</strong>
                          </button>
                        );
                      })}
                    </div>
                    <div className="selected-history">
                      <div>
                        <strong>已选 {selectedHistory.length}/6 条</strong>
                        <span>排序会随选择立即更新</span>
                      </div>
                      <ul>
                        {selectedHistory.map((item) => (
                          <li key={item.id}>
                            <span>{item.title}</span>
                            <button type="button" onClick={() => toggleHistory(item)} aria-label={`移除 ${item.title}`}>
                              ×
                            </button>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="preset-list" aria-label="预设兴趣">
                      {data.sandbox.presets.map((preset) => (
                    <button
                      key={preset.id}
                      type="button"
                      className={preset.id === activePreset ? "is-active" : ""}
                      aria-pressed={preset.id === activePreset}
                      onClick={() => choosePreset(preset)}
                    >
                      <strong>{preset.label}</strong>
                      <span>{preset.keywords}</span>
                    </button>
                  ))}
                    </div>
                    <div className="interest-library">
                      <div className="interest-library-heading">
                        <strong>模型可识别兴趣词</strong>
                        <span>点击添加或移除，无需猜英文</span>
                      </div>
                      <div className="interest-topic-tabs" aria-label="兴趣词分类">
                        {data.sandbox.interest_term_groups.map((group) => (
                          <button
                            key={group.id}
                            type="button"
                            className={interestTermTopic === group.id ? "is-active" : ""}
                            onClick={() => setInterestTermTopic(group.id)}
                          >
                            {group.label}
                          </button>
                        ))}
                      </div>
                      <div className="interest-term-list">
                        {data.sandbox.interest_term_groups
                          .find((group) => group.id === interestTermTopic)
                          ?.terms.map((term) => {
                            const selectedTerms = interestInput.toLowerCase().match(/[a-z0-9]+/g) ?? [];
                            const selected = selectedTerms.includes(term);
                            return (
                              <button
                                key={term}
                                type="button"
                                className={selected ? "is-selected" : ""}
                                aria-pressed={selected}
                                onClick={() => toggleInterestTerm(term)}
                              >
                                {term}
                              </button>
                            );
                          })}
                      </div>
                    </div>
                    <form className="interest-form" onSubmit={submitInterest}>
                      <label htmlFor="interest-input">英文兴趣词</label>
                      <div>
                        <input
                          id="interest-input"
                          value={interestInput}
                          maxLength={120}
                          placeholder="例如 anime game naruto"
                          onChange={(event) => setInterestInput(event.target.value)}
                        />
                        <button type="submit">生成Top-10</button>
                      </div>
                    </form>
                  </>
                )}
              </div>

              {simulationSource === "text" && <div className="recognition-row" aria-live="polite">
                <div>
                  <span>已识别</span>
                  <strong>
                    {activeSimulation.recognized.length
                      ? activeSimulation.recognized.join(" · ")
                      : "没有有效词"}
                  </strong>
                </div>
                <div>
                  <span>未识别</span>
                  <strong>
                    {[
                      ...activeSimulation.ignored,
                      ...(activeSimulation.hasUnsupportedCharacters ? ["中文/非ASCII字符"] : []),
                    ].join(" · ") || "无"}
                  </strong>
                </div>
              </div>}

              <div className="recommendation-heading">
                <div>
                  <p className="section-kicker">{simulationSource === "history" ? "正式画像公式" : "兴趣词近似画像"}</p>
                  <h2>{data.sandbox.candidate_items.toLocaleString()}个冷候选中的Top-10</h2>
                </div>
                <p>无真实正例 · 不计算准确率</p>
              </div>
              <SimulationRecommendations results={activeSimulation.results} />
            </section>

            <aside className="decision-panel simulation-aside">
              <p className="section-kicker">模拟模式边界</p>
              <span className="policy-level caution">交互演示</span>
              <h2>这是新的推理，不是新的实验</h2>
              <p>
                你定义了一个临时兴趣画像，冻结模型据此重新排序；因为没有用户之后的真实行为，无法判断推荐是否正确。
              </p>
              <div className="decision-rule">
                <span>画像来源</span>
                <strong>{simulationSource === "history" ? `${selectedHistory.length}条训练历史` : "兴趣词近似"}</strong>
              </div>
              <div className="decision-rule">
                <span>候选范围</span>
                <strong>完整1,305个冷视频</strong>
              </div>
              <div className="decision-rule">
                <span>模型状态</span>
                <strong>冻结Text-only</strong>
              </div>
              <div className="decision-rule">
                <span>数据发送</span>
                <strong>仅浏览器本地</strong>
              </div>
              <div className="boundary-note">
                <strong>{simulationSource === "history" ? "主要相关历史是什么意思？" : "为什么中文不工作？"}</strong>
                <p>{simulationSource === "history" ? "它是与该候选余弦相似度最高的一条已选历史，只解释当前文本匹配，不代表真实因果。" : "原实验标题主要是英文翻译/转写，冻结tokenizer不具备中文语义理解。支持中文需要一个新的编码器与独立评测。"}</p>
              </div>
            </aside>
          </div>
        </>
      ) : (
        <ComparisonPanel />
      )}

      <footer>
        <p>Model-side Item Cold-Start · Validation Cold · Full candidate ranking</p>
        <p>模型与最终Test已锁定；模拟模式不写入实验结果。</p>
      </footer>
    </main>
  );
}
