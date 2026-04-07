"""Convenience Neo4j queries used by the application.

This module exposes small helpers that return graph facts for a list of
`arxiv_id`s. The functions intentionally return simple Python structures so
callers can decide how to render them into prompts.
"""

from __future__ import annotations

from typing import Dict, List, Any


def build_papers_relations_query() -> str:
	"""Return a parameterized Cypher query that collects relations for papers.

	Expects parameter `ids` as a list of arxiv_id strings.
	Returns rows with fields: arxiv_id, title, abstract, relations (list of maps).
	"""
	return (
		"""
		MATCH (p:Paper)
		WHERE p.arxiv_id IN $ids
		OPTIONAL MATCH (p)-[rel]->(o)
		WITH p, rel, o
		RETURN p.arxiv_id AS arxiv_id,
			   p.title AS title,
			   p.abstract AS abstract,
			   collect(CASE WHEN rel IS NULL THEN null ELSE {rel_type: type(rel), obj_labels: labels(o), obj_props: properties(o)} END) AS relations
		"""
	)


def simplify_relations_row(row: Dict[str, Any]) -> Dict[str, Any]:
	"""Normalize a row returned by execute_read into a simpler dict.

	The Neo4j driver may return nested types; this helper returns a mapping
	with `arxiv_id` -> list of relation dicts (filtering out null placeholders).
	"""
	arxiv_id = row.get("arxiv_id")
	relations = row.get("relations") or []
	cleaned = []
	for r in relations:
		if not r:
			continue
		# r is expected to be a map with keys rel_type, obj_labels, obj_props
		cleaned.append({
			"rel_type": r.get("rel_type"),
			"obj_labels": r.get("obj_labels") or [],
			"obj_props": r.get("obj_props") or {},
		})

	return {"arxiv_id": arxiv_id, "relations": cleaned, "title": row.get("title"), "abstract": row.get("abstract")}

