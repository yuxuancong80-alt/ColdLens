import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { PUBLIC_DEMO_DATA } from "../app/public-demo-data.ts";
import { simulate, simulateFromHistory } from "../app/simulation.ts";

const root = new URL("../", import.meta.url);
const localDemoUrl = new URL("public/demo-data.json", root);

test("public build uses only the synthetic data path and keeps boundaries visible", async () => {
  const [page, layout, packageJson, viteConfig] = await Promise.all([
    readFile(new URL("app/page.tsx", root), "utf8"),
    readFile(new URL("app/layout.tsx", root), "utf8"),
    readFile(new URL("package.json", root), "utf8"),
    readFile(new URL("vite.config.ts", root), "utf8"),
  ]);

  assert.match(layout, /ColdLens · 冷启动推荐实验室/);
  assert.match(layout, /以合成样例解释 Text-only 推荐机制/);
  assert.match(layout, /lang="zh-CN"/);
  assert.match(page, /公开安全版 · 合成机制数据/);
  assert.match(page, /不是证明多模态更强/);
  assert.match(page, /60 秒看懂项目/);
  assert.match(page, /产品决策负责，AI 辅助实现/);
  assert.match(page, /进入交互 Demo/);
  assert.match(page, /查看 GitHub 证据链/);
  assert.match(page, /所有可交互标题、历史、目标与排序均为完全自写的合成示例/);
  assert.match(page, /创建模拟用户/);
  assert.match(page, /选择历史视频/);
  assert.match(page, /主要相关历史/);
  assert.match(page, /方案对比/);
  assert.match(page, /标题文本方案/);
  assert.match(page, /封面视觉方案/);
  assert.match(page, /证据决定去留/);
  assert.match(page, /0\.12551/);
  assert.match(page, /0\.04543/);
  assert.match(page, /未通过双指标门槛/);
  assert.match(page, /分数是余弦相似度，不是概率/);
  assert.match(page, /这是新的推理，不是新的实验/);
  assert.match(page, /get\("local"\) === "1"/);
  assert.match(viteConfig, /command === "serve" \? "public" : "public-safe"/);
  assert.doesNotMatch(page, /SkeletonPreview|codex-preview/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
});

test("safe public dataset is fully synthetic and offers a broader interest library", () => {
  const data = PUBLIC_DEMO_DATA;
  assert.equal(data.demo, "safe_public_synthetic_v1");
  assert.equal(data.test_split_read, false);
  assert.equal(data.samples.length, 8);
  assert.equal(data.sandbox.candidate_items, 72);
  assert.equal(data.sandbox.candidates.length, 72);
  assert.equal(data.sandbox.history_catalog.length, 48);
  assert.equal(data.sandbox.presets.length, 12);
  assert.equal(data.sandbox.interest_term_groups.length, 12);
  assert.equal(
    data.sandbox.interest_term_groups.reduce(
      (total, group) => total + group.terms.length,
      0,
    ),
    96,
  );
  assert.ok(data.samples.every((sample) => sample.id.startsWith("S")));
  assert.ok(data.sandbox.candidates.every((candidate) => candidate.title.startsWith("合成候选｜")));
  assert.ok(data.sandbox.history_catalog.every((item) => item.id.startsWith("synthetic-history-")));
});

test("safe history and interest simulations produce deterministic explanations", () => {
  const data = PUBLIC_DEMO_DATA;
  const selected = [data.sandbox.history_catalog[0], data.sandbox.history_catalog[5]];
  const first = simulateFromHistory(data.sandbox, selected);
  const second = simulateFromHistory(data.sandbox, selected);
  assert.deepEqual(first, second);
  assert.ok(first.results.length > 0);
  assert.ok(first.results.every((item) => item.evidence?.history_id.startsWith("synthetic-history-")));

  for (const preset of data.sandbox.presets) {
    const result = simulate(data.sandbox, preset.keywords);
    assert.ok(result.results.length > 0, preset.id);
  }
  const unsupported = simulate(data.sandbox, "火影忍者");
  assert.equal(unsupported.hasUnsupportedCharacters, true);
  assert.deepEqual(unsupported.results, []);
});

test("optional local full dataset still matches its frozen references", async (t) => {
  let data;
  try {
    data = JSON.parse(await readFile(localDemoUrl, "utf8"));
  } catch (error) {
    if (error && typeof error === "object" && "code" in error && error.code === "ENOENT") {
      t.skip("local ignored demo data is intentionally absent");
      return;
    }
    throw error;
  }

  assert.equal(data.test_split_read, false);
  assert.ok(data.samples.every((sample) => !("user_id" in sample)));
  const reference = data.sandbox.reference_history_scenario;
  const selected = reference.selected_history_ids.map((id) =>
    data.sandbox.history_catalog.find((item) => item.id === id),
  );
  const result = simulateFromHistory(data.sandbox, selected);
  assert.deepEqual(
    result.results.map((item) => item.candidate_index),
    reference.expected_top_candidate_indices,
  );
  for (const query of data.sandbox.reference_queries) {
    assert.deepEqual(
      simulate(data.sandbox, query.keywords).results.map((item) => item.candidate_index),
      query.expected_top_candidate_indices,
      query.id,
    );
  }
});
