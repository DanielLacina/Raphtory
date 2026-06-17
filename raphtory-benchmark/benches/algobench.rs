use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use rand::{rngs::SmallRng, SeedableRng};
use raphtory::{
    algorithms::{
        alternating_mask::alternating_mask,
        bipartite::max_weight_matching::max_weight_matching,
        centrality::{
            betweenness::betweenness_centrality, degree_centrality::degree_centrality,
            hits::hits, pagerank::page_rank,
        },
        community_detection::{
            label_propagation::label_propagation,
            louvain::louvain,
            modularity::ModularityUnDir,
        },
        components::{
            in_component, in_components, out_component, out_components,
            strongly_connected_components, weakly_connected_components,
        },
        cores::k_core::{k_core, k_core_set},
        dynamics::temporal::epidemics::{temporal_SEIR, Number},
        embeddings::fast_rp::fast_rp,
        layout::{
            cohesive_fruchterman_reingold::cohesive_fruchterman_reingold,
            fruchterman_reingold::fruchterman_reingold_unbounded,
        },
        metrics::{
            balance::balance,
            clustering_coefficient::{
                global_clustering_coefficient::global_clustering_coefficient,
                local_clustering_coefficient::local_clustering_coefficient,
                local_clustering_coefficient_batch::local_clustering_coefficient_batch,
            },
            degree::{
                average_degree, max_degree, max_in_degree, max_out_degree, min_degree,
                min_in_degree, min_out_degree,
            },
            directed_graph_density::directed_graph_density,
            reciprocity::{all_local_reciprocity, global_reciprocity},
        },
        motifs::{
            global_temporal_three_node_motifs::{
                global_temporal_three_node_motif, temporal_three_node_motif_multi,
            },
            local_temporal_three_node_motifs::temporal_three_node_motif as local_temporal_three_node_motif,
            local_triangle_count::local_triangle_count,
            temporal_rich_club_coefficient::temporal_rich_club_coefficient,
            triangle_count::triangle_count,
            triplet_count::triplet_count,
        },
        pathing::{
            dijkstra::dijkstra_single_source_shortest_paths,
            single_source_shortest_path::single_source_shortest_path,
            temporal_reachability::temporally_reachable_nodes,
        },
        projections::temporal_bipartite_projection::temporal_bipartite_projection,
    },
    graphgen::random_attachment::random_attachment,
    prelude::*,
};
use raphtory_api::core::Direction;
use raphtory_benchmark::common::bench;
use std::hint::black_box;

fn large_random_attachment_graph() -> Graph {
    let graph = Graph::new();
    let seed: [u8; 32] = [1; 32];
    random_attachment(&graph, 500000, 4, Some(seed));
    graph
}

fn first_node_id(graph: &Graph) -> GID {
    graph
        .nodes()
        .id()
        .iter_values()
        .next()
        .expect("graph has nodes")
}

fn large_weighted_random_attachment_graph() -> Graph {
    let graph = large_random_attachment_graph();
    let ids = graph.nodes().id().iter_values().collect::<Vec<_>>();
    if let (Some(src), Some(dst)) = (ids.first(), ids.get(1)) {
        graph
            .add_edge(0, src.clone(), dst.clone(), [("weight", 1.0f64)], None)
            .expect("unable to add weighted edge");
    }
    graph
}

fn large_typed_random_attachment_graph() -> Graph {
    let graph = large_random_attachment_graph();
    for id in graph.nodes().id().iter_values() {
        graph
            .add_node(0, id, NO_PROPS, Some("Right"), None)
            .expect("unable to set node type");
    }
    graph
}

pub fn local_triangle_count_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("local_triangle_count");
    group.sample_size(10);
    bench(&mut group, "local_triangle_count", None, |b| {
        let graph = large_random_attachment_graph();
        let node_id = graph.nodes().id().iter_values().next().expect("graph has nodes");

        b.iter(|| black_box(local_triangle_count(&graph, node_id.clone()).unwrap()))
    });

    group.finish();
}

pub fn local_clustering_coefficient_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("local_clustering_coefficient");

    bench(&mut group, "local_clustering_coefficient", None, |b| {
        let graph = large_random_attachment_graph();
        let node_id = graph.nodes().id().iter_values().next().expect("graph has nodes");

        b.iter(|| black_box(local_clustering_coefficient(&graph, node_id.clone())))
    });

    group.finish();
}

pub fn graphgen_large_clustering_coeff(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_clustering_coeff");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(60));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_clustering_coeff", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = global_clustering_coefficient(graph);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_pagerank(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_pagerank");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_pagerank", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = page_rank(graph, None, Some(100), None, None, true, None);
                black_box(result);
            });
        },
    );
    group.finish()
}


pub fn graphgen_large_concomp(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_concomp");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(60));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_concomp", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = weakly_connected_components(graph);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_hits(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_hits");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_hits", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = hits(graph, 100, None);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_degree_centrality(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_degree_centrality");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_degree_centrality", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = degree_centrality(graph);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_betweenness(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_betweenness");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_betweenness", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = betweenness_centrality(graph, None, false);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_triangle_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_triangle_count");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_triangle_count", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = triangle_count(graph, None);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_triplet_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_triplet_count");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_triplet_count", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = triplet_count(graph, None);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_directed_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_directed_density");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_directed_density", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = directed_graph_density(graph);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_reciprocity(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_reciprocity");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_reciprocity", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = global_reciprocity(graph);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_scc(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_scc");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_scc", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = strongly_connected_components(graph);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_in_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_in_components");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_in_components", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = in_components(graph, None);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_out_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_out_components");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_out_components", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = out_components(graph, None);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_label_propagation(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_label_propagation");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_label_propagation", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = label_propagation(graph, 20, Some([1; 32]), None);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_louvain(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_louvain");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_louvain", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = louvain::<ModularityUnDir, _>(graph, 1.0, None, None);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_alternating_mask(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_alternating_mask");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_alternating_mask", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = alternating_mask(graph);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_all_local_reciprocity(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_all_local_reciprocity");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_all_local_reciprocity", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = all_local_reciprocity(graph);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_balance(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_balance");
    let graph = large_weighted_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_balance", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = balance(graph, "weight".to_string(), Direction::BOTH).unwrap();
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_max_degree(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_max_degree");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_max_degree", &graph),
        &graph,
        |b, graph| {
            b.iter(|| black_box(max_degree(graph)));
        },
    );
    group.finish()
}

pub fn graphgen_large_min_degree(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_min_degree");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_min_degree", &graph),
        &graph,
        |b, graph| {
            b.iter(|| black_box(min_degree(graph)));
        },
    );
    group.finish()
}

pub fn graphgen_large_max_out_degree(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_max_out_degree");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_max_out_degree", &graph),
        &graph,
        |b, graph| {
            b.iter(|| black_box(max_out_degree(graph)));
        },
    );
    group.finish()
}

pub fn graphgen_large_max_in_degree(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_max_in_degree");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_max_in_degree", &graph),
        &graph,
        |b, graph| {
            b.iter(|| black_box(max_in_degree(graph)));
        },
    );
    group.finish()
}

pub fn graphgen_large_min_out_degree(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_min_out_degree");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_min_out_degree", &graph),
        &graph,
        |b, graph| {
            b.iter(|| black_box(min_out_degree(graph)));
        },
    );
    group.finish()
}

pub fn graphgen_large_min_in_degree(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_min_in_degree");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_min_in_degree", &graph),
        &graph,
        |b, graph| {
            b.iter(|| black_box(min_in_degree(graph)));
        },
    );
    group.finish()
}

pub fn graphgen_large_average_degree(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_average_degree");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_average_degree", &graph),
        &graph,
        |b, graph| {
            b.iter(|| black_box(average_degree(graph)));
        },
    );
    group.finish()
}

pub fn graphgen_large_local_clustering_coefficient_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_local_clustering_coefficient_batch");
    let graph = large_random_attachment_graph();
    let node_id = first_node_id(&graph);

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_local_clustering_coefficient_batch", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = local_clustering_coefficient_batch(graph, vec![node_id.clone()]);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_temporal_rich_club(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_temporal_rich_club");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_temporal_rich_club", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let rolling = graph.rolling(1, Some(1)).unwrap();
                let result = temporal_rich_club_coefficient(graph, rolling, 3, 3);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_temporal_motif_multi(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_temporal_motif_multi");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_temporal_motif_multi", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = temporal_three_node_motif_multi(graph, vec![100], None);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_local_temporal_motif(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_local_temporal_motif");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_local_temporal_motif", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = local_temporal_three_node_motif(graph, 100, None);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_dijkstra(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_dijkstra");
    let graph = large_random_attachment_graph();
    let source = first_node_id(&graph);

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_dijkstra", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = dijkstra_single_source_shortest_paths(
                    graph,
                    source.clone(),
                    vec![source.clone()],
                    None,
                    Direction::BOTH,
                )
                .unwrap();
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_single_source_shortest_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_single_source_shortest_path");
    let graph = large_random_attachment_graph();
    let source = first_node_id(&graph);

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_single_source_shortest_path", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = single_source_shortest_path(graph, source.clone(), None);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_temporally_reachable_nodes(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_temporally_reachable_nodes");
    let graph = large_random_attachment_graph();
    let source = first_node_id(&graph);

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_temporally_reachable_nodes", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result =
                    temporally_reachable_nodes(graph, None, 20, 0, vec![source.clone()], None);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_in_component(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_in_component");
    let graph = large_random_attachment_graph();
    let source = first_node_id(&graph);

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_in_component", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let node = graph.node(source.clone()).expect("source node exists");
                let result = in_component(node);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_out_component(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_out_component");
    let graph = large_random_attachment_graph();
    let source = first_node_id(&graph);

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_out_component", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let node = graph.node(source.clone()).expect("source node exists");
                let result = out_component(node);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_k_core_set(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_k_core_set");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_k_core_set", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = k_core_set(graph, 2, usize::MAX, None);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_k_core(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_k_core");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_k_core", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = k_core(graph, 2, usize::MAX, None);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_fruchterman_reingold(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_fruchterman_reingold");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_fruchterman_reingold", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = fruchterman_reingold_unbounded(graph, 5, 1.0, 1.0, 0.9, 0.1);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_cohesive_fruchterman_reingold(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_cohesive_fruchterman_reingold");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_cohesive_fruchterman_reingold", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = cohesive_fruchterman_reingold(graph, 5, 1.0, 1.0, 0.9, 0.1);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_fast_rp(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_fast_rp");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_fast_rp", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = fast_rp(graph, 32, 0.5, vec![1.0, 1.0, 1.0], Some(1), None);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_max_weight_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_max_weight_matching");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_max_weight_matching", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = max_weight_matching(graph, None, false, false);
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_temporal_seir(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_temporal_seir");
    let graph = large_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_temporal_seir", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let mut rng = SmallRng::seed_from_u64(1);
                let result = temporal_SEIR(graph, Some(0.1), None, 0.5f64, 0, Number(1), &mut rng)
                    .unwrap();
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn graphgen_large_temporal_bipartite_projection(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphgen_large_temporal_bipartite_projection");
    let graph = large_typed_random_attachment_graph();

    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::new("graphgen_large_temporal_bipartite_projection", &graph),
        &graph,
        |b, graph| {
            b.iter(|| {
                let result = temporal_bipartite_projection(graph, 1, "Right".to_string());
                black_box(result);
            });
        },
    );
    group.finish()
}

pub fn temporal_motifs(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_motifs");

    bench(&mut group, "temporal_motifs", None, |b| {
        let graph = large_random_attachment_graph();

        b.iter(|| black_box(global_temporal_three_node_motif(&graph, 100, None)))
    });

    group.finish();
}

criterion_group!(
    benches,
    local_triangle_count_analysis,
    local_clustering_coefficient_analysis,
    graphgen_large_clustering_coeff,
    graphgen_large_pagerank,
    graphgen_large_concomp,
    graphgen_large_hits,
    graphgen_large_degree_centrality,
    graphgen_large_betweenness,
    graphgen_large_triangle_count,
    graphgen_large_triplet_count,
    graphgen_large_directed_density,
    graphgen_large_reciprocity,
    graphgen_large_scc,
    graphgen_large_in_components,
    graphgen_large_out_components,
    graphgen_large_label_propagation,
    graphgen_large_louvain,
    graphgen_large_alternating_mask,
    graphgen_large_all_local_reciprocity,
    graphgen_large_balance,
    graphgen_large_max_degree,
    graphgen_large_min_degree,
    graphgen_large_max_out_degree,
    graphgen_large_max_in_degree,
    graphgen_large_min_out_degree,
    graphgen_large_min_in_degree,
    graphgen_large_average_degree,
    graphgen_large_local_clustering_coefficient_batch,
    graphgen_large_temporal_rich_club,
    graphgen_large_temporal_motif_multi,
    graphgen_large_local_temporal_motif,
    graphgen_large_dijkstra,
    graphgen_large_single_source_shortest_path,
    graphgen_large_temporally_reachable_nodes,
    graphgen_large_in_component,
    graphgen_large_out_component,
    graphgen_large_k_core_set,
    graphgen_large_k_core,
    graphgen_large_fruchterman_reingold,
    graphgen_large_cohesive_fruchterman_reingold,
    graphgen_large_fast_rp,
    graphgen_large_max_weight_matching,
    graphgen_large_temporal_seir,
    graphgen_large_temporal_bipartite_projection,
    temporal_motifs,
);
criterion_main!(benches);
