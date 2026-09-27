from chatbot_eval.graph import (
    GraphNode,
    KnowledgeGraph,
    build_edges,
    cluster_evidence_ids,
    cluster_importance,
    derive_theme_clusters,
    derive_topic_clusters,
    entity_match_keys,
    normalize_entity,
)


def test_normalization_is_lossless_and_does_not_corrupt_base_words():
    # Gershayim/niqqud stripped, casefolded, but leading letters of real words are preserved.
    assert normalize_entity('צה"ל') == normalize_entity("צה״ל") == "צהל"
    assert normalize_entity("שכר") == "שכר"          # must NOT become כר
    assert normalize_entity("מעטפת") == "מעטפת"      # must NOT become עטפת
    assert normalize_entity("Benefits") == "benefits"
    assert normalize_entity("   מעטפת   רווחה  ") == "מעטפת רווחה"


def test_entity_match_keys_tolerate_attached_prefixes_only_where_safe():
    # An attached conjunction/preposition on a gershayim acronym should match the bare acronym.
    assert "צהל" in entity_match_keys('ולצה"ל')
    assert entity_match_keys('צה"ל') & entity_match_keys('ולצה"ל')
    # A plain single base word gets no risky prefix-stripped variant (avoids false merges).
    assert entity_match_keys("ושכר") == {"ושכר"}
    assert not (entity_match_keys("שכר") & entity_match_keys("ושכר"))


def test_shared_entity_edges_use_inverted_index_and_prefix_tolerant_matching():
    nodes = [
        GraphNode("a#1", "a.md", "d", "x", entities=['צה"ל']),
        GraphNode("a#2", "a.md", "d", "x", entities=['ולצה"ל']),
        GraphNode("b#1", "b.md", "d", "x", entities=["תמריץ"]),
    ]
    edges = build_edges(nodes)
    shared = [e for e in edges if e.type == "shared_entity"]
    assert any({e.source_id, e.target_id} == {"a#1", "a#2"} for e in shared)
    # b#1 shares nothing, so it gets no shared_entity edge.
    assert not any("b#1" in {e.source_id, e.target_id} for e in shared)


def test_keyphrase_overlap_and_same_document_edges():
    nodes = [
        GraphNode("a#1", "a.md", "d", "x", keyphrases=["מבנה שכר", "תמריץ"]),
        GraphNode("a#2", "a.md", "d", "x", keyphrases=["מבנה שכר", "תמריץ"]),
    ]
    edges = build_edges(nodes, keyphrase_overlap_threshold=0.3)
    assert any(e.type == "keyphrase_overlap" for e in edges)
    assert any(e.type == "same_document" for e in edges)


def test_clusters_group_by_meaning_not_mere_adjacency():
    # a#1/a#2 share TWO entities (strong enough to cluster under the default weight threshold);
    # a#3 is unrelated and stays separate even though same_document edges exist, because clustering
    # ignores same_document.
    nodes = [
        GraphNode("a#1", "a.md", "d", "x", entities=["חופשה", "אישור"]),
        GraphNode("a#2", "a.md", "d", "x", entities=["חופשה", "אישור"]),
        GraphNode("a#3", "a.md", "d", "x", entities=["דבר אחר לגמרי"]),
    ]
    graph = KnowledgeGraph(nodes, build_edges(nodes))
    clusters = derive_topic_clusters(graph, max_topics=12)
    joined = next(c for c in clusters if "a#1" in c)
    assert "a#2" in joined and "a#3" not in joined


def test_weak_single_hub_link_does_not_fuse_clusters():
    # A single shared "hub" entity (weight 1) must NOT merge two otherwise-distinct nodes into one
    # topic (avoids the knowledge-graph hairball). They remain separate clusters.
    nodes = [
        GraphNode("a#1", "a.md", "d", "x", entities=["צבא", "חופשה"]),
        GraphNode("b#1", "b.md", "d", "x", entities=["צבא", "שכר"]),
    ]
    graph = KnowledgeGraph(nodes, build_edges(nodes))
    clusters = derive_topic_clusters(graph, max_topics=12, min_cluster_edge_weight=2.0)
    assert not any(len(c) == 2 for c in clusters)  # not fused by the single shared "צבא"


def test_dense_corpus_neither_collapses_nor_explodes():
    """A densely interconnected corpus (many nodes glued by weak hub links, with cohesive strong
    sub-groups) must split into bounded topics -- not one giant 'other' and not a swarm of
    singletons. Nodes 0-5 form a strong sub-cluster (share 3 entities); the rest are weakly linked
    singletons via a common hub entity."""
    nodes = []
    for i in range(6):  # strong cohesive core
        nodes.append(GraphNode(f"core{i}", "f.md", "d", "x", entities=["hub", "e1", "e2", "e3"]))
    for i in range(20):  # weakly-linked periphery (only the hub entity in common)
        nodes.append(GraphNode(f"p{i}", "f.md", "d", "x", entities=["hub", f"u{i}"]))
    graph = KnowledgeGraph(nodes, build_edges(nodes))
    clusters = derive_topic_clusters(graph, max_topics=5, min_cluster_edge_weight=2.0, max_cluster_size=8)
    sizes = sorted((len(c) for c in clusters), reverse=True)
    assert sizes[0] <= 8                       # no giant cluster
    assert len(clusters) <= 10                 # not exploded into ~26 singletons
    assert sum(sizes) == 26                    # every node covered exactly once
    # The strong core stays together in one cluster.
    core = next(c for c in clusters if "core0" in c)
    assert all(f"core{i}" in core for i in range(6))


def test_oversized_cluster_is_split_by_weakest_edge_removal():
    # A chain a-b-c-d fully connected by 2-entity links is one component; capping size forces a split.
    nodes = [
        GraphNode("n1", "f.md", "d", "x", entities=["e", "f"]),
        GraphNode("n2", "f.md", "d", "x", entities=["e", "f"]),
        GraphNode("n3", "f.md", "d", "x", entities=["e", "f"]),
        GraphNode("n4", "f.md", "d", "x", entities=["e", "f"]),
    ]
    graph = KnowledgeGraph(nodes, build_edges(nodes))
    clusters = derive_topic_clusters(graph, max_topics=12, min_cluster_edge_weight=2.0, max_cluster_size=2)
    assert all(len(c) <= 2 for c in clusters)
    covered = sorted(n for c in clusters for n in c)
    assert covered == ["n1", "n2", "n3", "n4"]


def test_clusters_are_capped_and_tail_merged():
    nodes = [GraphNode(f"n{i}", f"f{i}.md", "d", "x", entities=[f"e{i}"]) for i in range(10)]
    graph = KnowledgeGraph(nodes, build_edges(nodes))
    clusters = derive_topic_clusters(graph, max_topics=3)
    assert len(clusters) <= 3
    # Every node is covered exactly once.
    covered = sorted(n for cluster in clusters for n in cluster)
    assert covered == sorted(node.chunk_id for node in nodes)


def test_cluster_importance_reflects_prevalence():
    assert cluster_importance(10, 10) == 5
    assert cluster_importance(1, 100) == 1
    assert cluster_importance(0, 0) == 1
    assert cluster_importance(5, 10) >= cluster_importance(1, 10)


def test_cluster_evidence_growth_is_bounded_and_connected():
    nodes = [
        GraphNode("a#1", "a.md", "d", "x", entities=["חופשה"]),
        GraphNode("a#2", "a.md", "d", "x", entities=["חופשה", "אישור"]),
        GraphNode("a#3", "a.md", "d", "x", entities=["אישור"]),
        GraphNode("far#1", "b.md", "d", "x", entities=["לא קשור"]),
    ]
    graph = KnowledgeGraph(nodes, build_edges(nodes))
    evidence = cluster_evidence_ids(graph, ["a#1"], max_nodes=3)
    assert "a#1" in evidence
    assert len(evidence) <= 3
    assert "far#1" not in evidence  # unreachable by semantic edges from a#1


def test_cluster_evidence_empty_seeds_returns_empty():
    graph = KnowledgeGraph([GraphNode("a#1", "a.md", "d", "x")], [])
    assert cluster_evidence_ids(graph, ["missing"], max_nodes=5) == []
    assert cluster_evidence_ids(graph, ["a#1"], max_nodes=0) == []


# --- theme-first clustering (opt-in theme layer) ------------------------------------------------


def test_theme_clusters_group_by_theme_even_with_low_entity_overlap():
    # Three pay chunks that share almost no entities still form ONE topic because they share a theme.
    # Entity mode would scatter them; theme mode keeps them together.
    nodes = [
        GraphNode("a#1", "a.md", "d", "x", entities=["קצין", "טופס 101"], theme="שכר"),
        GraphNode("b#1", "b.md", "d", "x", entities=["מילואים", "החזר"], theme="שכר"),
        GraphNode("c#1", "c.md", "d", "x", entities=["דרגה", "ותק"], theme="שכר"),
        GraphNode("d#1", "d.md", "d", "x", entities=["חופשה"], theme="חופשות"),
    ]
    graph = KnowledgeGraph(nodes, build_edges(nodes))
    clusters = derive_theme_clusters(graph, max_topics=40)
    pay = next(c for c in clusters if "a#1" in c)
    assert {"a#1", "b#1", "c#1"} <= set(pay)
    assert "d#1" not in pay


def test_theme_clusters_fall_back_to_entity_clustering_when_no_themes():
    # No node carries a theme -> behave exactly like entity clustering (nothing lost).
    nodes = [
        GraphNode("a#1", "a.md", "d", "x", entities=["חופשה", "אישור"]),
        GraphNode("a#2", "a.md", "d", "x", entities=["חופשה", "אישור"]),
        GraphNode("a#3", "a.md", "d", "x", entities=["דבר אחר"]),
    ]
    graph = KnowledgeGraph(nodes, build_edges(nodes))
    theme = derive_theme_clusters(graph, max_topics=40, min_cluster_edge_weight=2.0)
    entity = derive_topic_clusters(graph, max_topics=40, min_cluster_edge_weight=2.0)
    assert sorted(sorted(c) for c in theme) == sorted(sorted(c) for c in entity)


def test_theme_clusters_split_oversized_theme_by_entity_cohesion():
    # One theme with two internally-cohesive sub-groups (each shares 2 entities) is split by entity
    # cohesion when it exceeds the cap, so the theme does not dominate; every node stays covered once.
    nodes = [
        GraphNode("g1a", "f.md", "d", "x", entities=["p", "q"], theme="שכר"),
        GraphNode("g1b", "f.md", "d", "x", entities=["p", "q"], theme="שכר"),
        GraphNode("g2a", "f.md", "d", "x", entities=["r", "s"], theme="שכר"),
        GraphNode("g2b", "f.md", "d", "x", entities=["r", "s"], theme="שכר"),
    ]
    graph = KnowledgeGraph(nodes, build_edges(nodes))
    clusters = derive_theme_clusters(graph, max_topics=40, max_cluster_size=2)
    assert all(len(c) <= 2 for c in clusters)
    covered = sorted(n for c in clusters for n in c)
    assert covered == sorted(node.chunk_id for node in nodes)


def test_theme_clusters_route_unthemed_and_other_nodes_without_dropping_them():
    # Themed nodes group by theme; unthemed / "אחר" nodes still surface (entity-clustered), never lost.
    nodes = [
        GraphNode("a#1", "a.md", "d", "x", entities=["x"], theme="שכר"),
        GraphNode("b#1", "b.md", "d", "x", entities=["y"], theme=""),
        GraphNode("c#1", "c.md", "d", "x", entities=["z"], theme="אחר"),
    ]
    graph = KnowledgeGraph(nodes, build_edges(nodes))
    clusters = derive_theme_clusters(graph, max_topics=40)
    covered = sorted(n for c in clusters for n in c)
    assert covered == ["a#1", "b#1", "c#1"]
    pay = next(c for c in clusters if "a#1" in c)
    assert "b#1" not in pay and "c#1" not in pay


def test_theme_clusters_are_capped_at_max_topics():
    nodes = [GraphNode(f"n{i}", f"f{i}.md", "d", "x", entities=[f"e{i}"], theme=f"t{i}") for i in range(10)]
    graph = KnowledgeGraph(nodes, build_edges(nodes))
    clusters = derive_theme_clusters(graph, max_topics=3)
    assert len(clusters) <= 3
    covered = sorted(n for c in clusters for n in c)
    assert covered == sorted(node.chunk_id for node in nodes)


def test_oversized_theme_splits_by_existing_components_not_into_singletons():
    nodes = [
        GraphNode("g1a", "f.md", "d", "x", entities=["p", "q"], theme="שכר"),
        GraphNode("g1b", "f.md", "d", "x", entities=["p", "q"], theme="שכר"),
        GraphNode("g2a", "f.md", "d", "x", entities=["r", "s"], theme="שכר"),
        GraphNode("g2b", "f.md", "d", "x", entities=["r", "s"], theme="שכר"),
    ]
    graph = KnowledgeGraph(nodes, build_edges(nodes))
    clusters = derive_theme_clusters(graph, max_topics=40, max_cluster_size=2)
    assert sorted(sorted(c) for c in clusters) == [["g1a", "g1b"], ["g2a", "g2b"]]


def test_oversized_theme_is_repacked_into_balanced_groups_within_the_theme():
    # Ten weakly connected pay chunks and one leave chunk: the pay theme must become two balanced
    # pay-only topics rather than singletons that get mixed with other themes.
    nodes = [GraphNode(f"pay#{i}", f"p{i}.md", "d", "x", entities=[f"e{i}"], theme="שכר") for i in range(10)]
    nodes.append(GraphNode("leave#1", "l.md", "d", "x", entities=["חופשה"], theme="חופשות"))
    nodes[0].entities, nodes[1].entities = ["a", "b"], ["a", "b"]
    graph = KnowledgeGraph(nodes, build_edges(nodes))
    clusters = derive_theme_clusters(graph, max_topics=40, max_cluster_size=6)
    pay = [c for c in clusters if any(n.startswith("pay") for n in c)]
    assert sorted(len(c) for c in pay) == [5, 5]
    assert all(all(n.startswith("pay") for n in c) for c in pay)
    assert ["leave#1"] in clusters
