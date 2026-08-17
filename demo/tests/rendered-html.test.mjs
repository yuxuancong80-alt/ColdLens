import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { simulate, simulateFromHistory } from "../app/simulation.ts";

const root = new URL("../", import.meta.url);

test("ColdLens demo replaces the starter and keeps the data boundary visible", async () => {
  const [page, layout, packageJson, demoData] = await Promise.all([
    readFile(new URL("app/page.tsx", root), "utf8"),
    readFile(new URL("app/layout.tsx", root), "utf8"),
    readFile(new URL("package.json", root), "utf8"),
    readFile(new URL("public/demo-data.json", root), "utf8"),
  ]);

  assert.match(layout, /ColdLens · 冷启动推荐实验室/);
  assert.match(layout, /lang="zh-CN"/);
  assert.match(page, /Validation候选池 · Test未读取/);
  assert.match(page, /创建模拟用户/);
  assert.match(page, /选择历史视频/);
  assert.match(page, /主要相关历史/);
  assert.match(page, /方案对比/);
  assert.match(page, /标题文本方案/);
  assert.match(page, /封面视觉方案/);
  assert.match(page, /当前Demo的交互推荐使用Text/);
  assert.match(page, /证据决定去留/);
  assert.match(page, /0\.12551/);
  assert.match(page, /0\.04543/);
  assert.match(page, /未通过双指标门槛/);
  assert.match(page, /页面不读取 Test 标签/);
  assert.match(page, /分数是余弦相似度，不是概率/);
  assert.match(page, /这是新的推理，不是新的实验/);
  assert.doesNotMatch(page, /SkeletonPreview|codex-preview/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);

  const data = JSON.parse(demoData);
  assert.equal(data.test_split_read, false);
  assert.equal(data.samples.length, 8);
  assert.deepEqual(
    data.samples.map((sample) => `${sample.history_bucket}:${sample.sample_outcome}`),
    [
      "1-3:hit",
      "1-3:miss",
      "4-6:hit",
      "4-6:miss",
      "7-10:hit",
      "7-10:miss",
      "11+:hit",
      "11+:miss",
    ],
  );
  assert.ok(data.samples.every((sample) => !("user_id" in sample)));
  assert.equal(data.sandbox.candidate_items, 1305);
  assert.equal(data.sandbox.candidates.length, 1305);
  assert.equal(data.sandbox.presets.length, 6);
  assert.equal(data.sandbox.history_catalog.length, 360);
  assert.equal(data.sandbox.interest_term_groups.length, 6);
  assert.equal(
    data.sandbox.interest_term_groups.reduce(
      (total, group) => total + group.terms.length,
      0,
    ),
    58,
  );
});

test("selected-history simulation matches the formal Python profile reference", async () => {
  const data = JSON.parse(
    await readFile(new URL("public/demo-data.json", root), "utf8"),
  );
  const reference = data.sandbox.reference_history_scenario;
  const selected = reference.selected_history_ids.map((id) =>
    data.sandbox.history_catalog.find((item) => item.id === id),
  );
  assert.ok(selected.every(Boolean));

  const result = simulateFromHistory(data.sandbox, selected);
  assert.deepEqual(
    result.results.map((item) => item.candidate_index),
    reference.expected_top_candidate_indices,
  );
  assert.deepEqual(
    result.results.map((item) => item.evidence?.history_id),
    reference.expected_evidence_history_ids,
  );
  assert.deepEqual(simulateFromHistory(data.sandbox, []).results, []);
});

test("browser simulation reproduces every frozen Python reference query", async () => {
  const data = JSON.parse(
    await readFile(new URL("public/demo-data.json", root), "utf8"),
  );
  for (const reference of data.sandbox.reference_queries) {
    const result = simulate(data.sandbox, reference.keywords);
    assert.deepEqual(
      result.results.map((item) => item.candidate_index),
      reference.expected_top_candidate_indices,
      reference.id,
    );
    assert.ok(result.results.length > 0);
  }

  const unsupported = simulate(data.sandbox, "火影忍者");
  assert.equal(unsupported.hasUnsupportedCharacters, true);
  assert.deepEqual(unsupported.results, []);
});
