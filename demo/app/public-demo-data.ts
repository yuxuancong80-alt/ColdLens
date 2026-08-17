import type { Sandbox } from "./simulation";

type Topic = {
  id: string;
  label: string;
  terms: string[];
};

const TOPICS: Topic[] = [
  { id: "anime", label: "动漫", terms: ["anime", "manga", "hero", "battle", "ninja", "fantasy", "cosplay", "mecha"] },
  { id: "gaming", label: "游戏", terms: ["game", "rpg", "strategy", "esports", "console", "puzzle", "adventure", "speedrun"] },
  { id: "food", label: "美食", terms: ["food", "recipe", "cooking", "baking", "dessert", "noodles", "coffee", "spicy"] },
  { id: "sports", label: "运动", terms: ["sports", "football", "basketball", "running", "fitness", "tennis", "cycling", "swimming"] },
  { id: "music", label: "音乐", terms: ["music", "guitar", "piano", "concert", "jazz", "rock", "vocal", "remix"] },
  { id: "film", label: "影视", terms: ["movie", "cinema", "comedy", "drama", "mystery", "scifi", "documentary", "thriller"] },
  { id: "travel", label: "旅行", terms: ["travel", "beach", "mountain", "city", "hotel", "hiking", "camping", "roadtrip"] },
  { id: "pets", label: "萌宠", terms: ["cat", "dog", "pet", "puppy", "kitten", "animal", "rescue", "training"] },
  { id: "tech", label: "科技", terms: ["tech", "ai", "robot", "coding", "phone", "camera", "gadget", "science"] },
  { id: "fashion", label: "穿搭", terms: ["fashion", "outfit", "makeup", "skincare", "style", "shoes", "vintage", "design"] },
  { id: "home", label: "生活", terms: ["home", "garden", "diy", "decor", "cleaning", "plant", "furniture", "organization"] },
  { id: "learning", label: "学习", terms: ["learning", "language", "history", "book", "math", "study", "career", "productivity"] },
];

const vectorFor = (terms: string[]): [string, number][] => {
  const weight = 1 / Math.sqrt(terms.length);
  return terms.map((term) => [term, weight]);
};

const historyCatalog = TOPICS.flatMap((topic) =>
  Array.from({ length: 4 }, (_, index) => {
    const terms = [
      topic.terms[index % topic.terms.length],
      topic.terms[(index + 1) % topic.terms.length],
      topic.terms[(index + 3) % topic.terms.length],
    ];
    return {
      id: `synthetic-history-${topic.id}-${index + 1}`,
      topic_id: topic.id,
      topic_label: topic.label,
      title: `${topic.label}灵感：${terms.join(" · ")} 入门清单`,
      vector: vectorFor(terms),
    };
  }),
);

const candidates = TOPICS.flatMap((topic) =>
  Array.from({ length: 6 }, (_, index) => {
    const terms = [
      topic.terms[index % topic.terms.length],
      topic.terms[(index + 2) % topic.terms.length],
      topic.terms[(index + 5) % topic.terms.length],
    ];
    return {
      title: `合成候选｜${topic.label}：${terms.join(" / ")} 今日精选`,
      vector: vectorFor(terms),
    };
  }),
);

const sandbox: Sandbox = {
  mode: "public_synthetic_text_sandbox_v1",
  candidate_items: candidates.length,
  tokenization: "ASCII word tokens for mechanism demonstration only",
  input_language_boundary: "English selectable terms only; Chinese UI labels are not model tokens",
  tie_breaking: "synthetic candidate order",
  presets: TOPICS.map((topic) => ({
    id: topic.id,
    label: topic.label,
    keywords: topic.terms.slice(0, 3).join(" "),
  })),
  interest_term_groups: TOPICS.map((topic) => ({
    id: topic.id,
    label: topic.label,
    terms: topic.terms,
  })),
  history_catalog_note: "48 self-authored synthetic titles; not MicroLens samples",
  history_catalog: historyCatalog,
  idf: Object.fromEntries(
    TOPICS.flatMap((topic, topicIndex) =>
      topic.terms.map((term, termIndex) => [term, 1.2 + topicIndex * 0.03 + termIndex * 0.02]),
    ),
  ),
  candidates,
};

const sampleTopics = TOPICS.slice(0, 4);
const samples = sampleTopics.flatMap((topic, topicIndex) =>
  (["hit", "miss"] as const).map((outcome, outcomeIndex) => {
    const topicCandidates = candidates.slice(topicIndex * 6, topicIndex * 6 + 6);
    const recommendations = topicCandidates.slice(0, 5).map((candidate, index) => ({
      rank: index + 1,
      title: candidate.title,
      score: Number((0.82 - index * 0.09 - outcomeIndex * 0.02).toFixed(3)),
      is_validation_positive: outcome === "hit" && index === 2,
      matched_terms: candidate.vector.slice(0, 2).map(([term], termIndex) => ({
        term,
        contribution: Number((0.31 - termIndex * 0.07).toFixed(3)),
      })),
    }));
    return {
      id: `S${topicIndex * 2 + outcomeIndex + 1}`,
      history_bucket: topicIndex < 2 ? "1-3" : "4-6",
      history_count: topicIndex < 2 ? 3 : 5,
      sample_outcome: outcome,
      history_titles: historyCatalog
        .filter((item) => item.topic_id === topic.id)
        .slice(0, topicIndex < 2 ? 3 : 4)
        .map((item) => item.title),
      validation_positive_titles: [
        outcome === "hit"
          ? topicCandidates[2].title
          : `合成目标｜${topic.label}：未进入展示列表的假设目标`,
      ],
      recommendations,
    };
  }),
);

export const PUBLIC_DEMO_DATA = {
  demo: "safe_public_synthetic_v1",
  evaluated_split: "aggregate_validation_metrics_plus_synthetic_examples",
  test_split_read: false,
  model: "text_tfidf_mechanism_replica",
  overall_validation: {
    users: 13_750,
    candidate_items: 1_305,
    positive_pairs: 0,
    recall_at_10: 0.12797,
    hit_rate_at_10: 0.15636,
  },
  selection_note: "Samples are fully self-authored and balanced for explanation only.",
  interpretation_boundary: "Aggregate metrics are real; titles, histories, targets, vectors, and rankings are synthetic.",
  samples,
  sandbox,
};
