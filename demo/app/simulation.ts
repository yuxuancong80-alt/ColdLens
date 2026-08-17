export type MatchedTerm = {
  term: string;
  contribution: number;
};

export type SandboxCandidate = {
  title: string;
  vector: [string, number][];
};

export type SandboxPreset = {
  id: string;
  label: string;
  keywords: string;
};

export type InterestTermGroup = {
  id: string;
  label: string;
  terms: string[];
};

export type HistoryCatalogItem = {
  id: string;
  topic_id: string;
  topic_label: string;
  title: string;
  vector: [string, number][];
};

export type Sandbox = {
  mode: string;
  candidate_items: number;
  tokenization: string;
  input_language_boundary: string;
  tie_breaking: string;
  presets: SandboxPreset[];
  interest_term_groups: InterestTermGroup[];
  history_catalog_note: string;
  history_catalog: HistoryCatalogItem[];
  idf: Record<string, number>;
  candidates: SandboxCandidate[];
  reference_queries?: {
    id: string;
    keywords: string;
    expected_top_candidate_indices: number[];
  }[];
  reference_history_scenario?: {
    selected_history_ids: string[];
    expected_top_candidate_indices: number[];
    expected_evidence_history_ids: string[];
  };
};

export type SimulationResult = {
  rank: number;
  candidate_index: number;
  title: string;
  score: number;
  matched_terms: MatchedTerm[];
  evidence?: {
    history_id: string;
    title: string;
    score: number;
  };
};

function rankCandidates(
  sandbox: Sandbox,
  profile: Map<string, number>,
  selectedHistory: HistoryCatalogItem[] = [],
): SimulationResult[] {
  const scored = sandbox.candidates.map((candidate, index) => {
    const contributions = candidate.vector
      .filter(([term]) => profile.has(term))
      .map(([term, weight]) => ({
        term,
        contribution: (profile.get(term) ?? 0) * weight,
      }))
      .sort(
        (left, right) =>
          right.contribution - left.contribution ||
          left.term.localeCompare(right.term),
      );
    const evidenceScores = selectedHistory.map((history) => ({
      history,
      score: history.vector.reduce(
        (total, [term, weight]) =>
          total + weight * (candidate.vector.find(([candidateTerm]) => candidateTerm === term)?.[1] ?? 0),
        0,
      ),
    }));
    evidenceScores.sort(
      (left, right) =>
        right.score - left.score ||
        selectedHistory.indexOf(left.history) - selectedHistory.indexOf(right.history),
    );
    const evidence = evidenceScores[0];
    return {
      index,
      title: candidate.title,
      score: contributions.reduce(
        (total, item) => total + item.contribution,
        0,
      ),
      matched_terms: contributions.slice(0, 3),
      evidence: evidence
        ? {
            history_id: evidence.history.id,
            title: evidence.history.title,
            score: evidence.score,
          }
        : undefined,
    };
  });

  return scored
    .filter((item) => item.score > 0)
    .sort((left, right) => right.score - left.score || left.index - right.index)
    .slice(0, 10)
    .map((item, index) => ({
      rank: index + 1,
      candidate_index: item.index,
      title: item.title,
      score: item.score,
      matched_terms: item.matched_terms,
      evidence: item.evidence,
    }));
}

export type Simulation = {
  recognized: string[];
  ignored: string[];
  hasUnsupportedCharacters: boolean;
  results: SimulationResult[];
};

export function simulate(sandbox: Sandbox, query: string): Simulation {
  const tokens = query.toLowerCase().match(/[a-z0-9]+/g) ?? [];
  const counts = new Map<string, number>();
  for (const token of tokens) counts.set(token, (counts.get(token) ?? 0) + 1);

  const recognized = [...counts.keys()].filter((token) => token in sandbox.idf);
  const ignored = [...counts.keys()].filter((token) => !(token in sandbox.idf));
  const rawWeights = new Map<string, number>();
  for (const term of recognized) {
    rawWeights.set(
      term,
      (1 + Math.log(counts.get(term) ?? 1)) * sandbox.idf[term],
    );
  }
  const norm = Math.sqrt(
    [...rawWeights.values()].reduce((total, value) => total + value * value, 0),
  );
  const profile = new Map<string, number>();
  if (norm > 0) {
    for (const [term, value] of rawWeights) profile.set(term, value / norm);
  }

  return {
    recognized,
    ignored,
    hasUnsupportedCharacters: [...query].some(
      (character) => character.charCodeAt(0) > 127,
    ),
    results: norm ? rankCandidates(sandbox, profile) : [],
  };
}

export function simulateFromHistory(
  sandbox: Sandbox,
  selectedHistory: HistoryCatalogItem[],
): Simulation {
  const rawProfile = new Map<string, number>();
  for (const history of selectedHistory) {
    for (const [term, weight] of history.vector) {
      rawProfile.set(term, (rawProfile.get(term) ?? 0) + weight);
    }
  }
  const norm = Math.sqrt(
    [...rawProfile.values()].reduce(
      (total, value) => total + value * value,
      0,
    ),
  );
  const profile = new Map<string, number>();
  if (norm > 0) {
    for (const [term, weight] of rawProfile) profile.set(term, weight / norm);
  }

  return {
    recognized: [...profile.keys()].sort(),
    ignored: [],
    hasUnsupportedCharacters: false,
    results: norm ? rankCandidates(sandbox, profile, selectedHistory) : [],
  };
}
