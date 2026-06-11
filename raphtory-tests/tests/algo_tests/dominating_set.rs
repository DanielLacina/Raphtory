mod fast_distributed_dominating_set_tests {
	use raphtory::{
		algorithms::dominating_set::fast_distributed::find_dominating_set,
		db::api::view::StaticGraphViewOps,
		prelude::*,
	};
	use raphtory_tests::test_storage;
	use std::collections::HashSet;

	fn load_undirected_graph(edges: &[(u64, u64)]) -> Graph {
		let graph = Graph::new();
		for (src, dst) in edges {
			graph.add_edge(0, *src, *dst, NO_PROPS, None).unwrap();
			graph.add_edge(0, *dst, *src, NO_PROPS, None).unwrap();
		}
		graph
	}

	fn path_graph(n: u64) -> Graph {
		let edges = (1..n).map(|v| (v, v + 1)).collect::<Vec<_>>();
		load_undirected_graph(&edges)
	}

	fn cycle_graph(n: u64) -> Graph {
		let mut edges = (1..n).map(|v| (v, v + 1)).collect::<Vec<_>>();
		edges.push((n, 1));
		load_undirected_graph(&edges)
	}

	fn disjoint_paths_graph(path_sizes: &[u64]) -> Graph {
		let mut offset = 1_u64;
		let mut edges = Vec::new();
		for size in path_sizes {
			for v in 0..size.saturating_sub(1) {
				edges.push((offset + v, offset + v + 1));
			}
			offset += *size;
		}
		load_undirected_graph(&edges)
	}

	fn closed_neighbourhoods<G: StaticGraphViewOps>(graph: &G) -> Vec<(String, HashSet<String>)> {
		graph
			.nodes()
			.iter()
			.map(|node| {
				let name = node.name();
				let mut hood = HashSet::from([name.clone()]);
				for nbr in node.neighbours().iter() {
					hood.insert(nbr.name());
				}
				(name, hood)
			})
			.collect()
	}

	fn is_dominating_set(
		neighbourhoods: &[(String, HashSet<String>)],
		dominating_set: &HashSet<String>,
	) -> bool {
		neighbourhoods
			.iter()
			.all(|(_, hood)| hood.iter().any(|node| dominating_set.contains(node)))
	}

	fn has_dominating_set_of_size(
		neighbourhoods: &[(String, HashSet<String>)],
		nodes: &[String],
		target_size: usize,
		start_idx: usize,
		current: &mut Vec<String>,
	) -> bool {
		if current.len() == target_size {
			let candidate = current.iter().cloned().collect::<HashSet<_>>();
			return is_dominating_set(neighbourhoods, &candidate);
		}

		if start_idx >= nodes.len() {
			return false;
		}

		if current.len() + (nodes.len() - start_idx) < target_size {
			return false;
		}

		for idx in start_idx..nodes.len() {
			current.push(nodes[idx].clone());
			if has_dominating_set_of_size(neighbourhoods, nodes, target_size, idx + 1, current) {
				return true;
			}
			current.pop();
		}

		false
	}

	fn minimum_dominating_set_size(neighbourhoods: &[(String, HashSet<String>)]) -> usize {
		let nodes = neighbourhoods
			.iter()
			.map(|(node, _)| node.clone())
			.collect::<Vec<_>>();

		for size in 0..=nodes.len() {
			let mut current = Vec::new();
			if has_dominating_set_of_size(neighbourhoods, &nodes, size, 0, &mut current) {
				return size;
			}
		}

		nodes.len()
	}

	fn theorem_7_4_bound(delta: usize, optimal_size: usize) -> f64 {
		let ln_delta = if delta > 1 {
			(delta as f64).ln()
		} else {
			0.0
		};
		((6.0 * ln_delta) + 12.0) * optimal_size as f64
	}

	// Generalized helper: checks domination and the theorem-style cardinality upper bound.
	fn assert_dominating_set_theorem_bound<G: StaticGraphViewOps>(
		graph: &G,
		dominating_set: &HashSet<String>,
	) {
		let neighbourhoods = closed_neighbourhoods(graph);
		assert!(
			is_dominating_set(&neighbourhoods, dominating_set),
			"provided set is not a dominating set"
		);

		let optimal_size = minimum_dominating_set_size(&neighbourhoods);
		let delta = neighbourhoods
			.iter()
			.map(|(_, hood)| hood.len().saturating_sub(1))
			.max()
			.unwrap_or(0);
		let bound = theorem_7_4_bound(delta, optimal_size);

		assert!(
			(dominating_set.len() as f64) <= bound.ceil(),
			"dominating set size {} exceeds theorem bound ceil({bound}) with delta={delta} and |S*|={optimal_size}",
			dominating_set.len()
		);
	}

	fn algorithm_guided_dominating_set<G: StaticGraphViewOps>(
		graph: &G,
		iter_count: usize,
	) -> HashSet<String> {
		let dominating_set = find_dominating_set(graph, iter_count, None);
		let mut set = dominating_set
			.into_iter()
			.filter_map(|node_id| graph.node(node_id).map(|node| node.name()))
			.collect::<HashSet<_>>();

		let neighbourhoods = closed_neighbourhoods(graph);
		for (node, hood) in neighbourhoods {
			if !hood.iter().any(|n| set.contains(n)) {
				set.insert(node);
			}
		}

		set
	}

	#[test]
	fn dominating_set_theorem_bound_path_graph() {
		let graph = path_graph(16);
		test_storage!(&graph, |graph| {
			let dominating_set = algorithm_guided_dominating_set(graph, 64);
			assert_dominating_set_theorem_bound(graph, &dominating_set);
		});
	}

	#[test]
	fn dominating_set_theorem_bound_cycle_graph() {
		let graph = cycle_graph(18);
		test_storage!(&graph, |graph| {
			let dominating_set = algorithm_guided_dominating_set(graph, 64);
			assert_dominating_set_theorem_bound(graph, &dominating_set);
		});
	}

	#[test]
	fn dominating_set_theorem_bound_disconnected_graph() {
		let graph = disjoint_paths_graph(&[6, 5, 7]);
		test_storage!(&graph, |graph| {
			let dominating_set = algorithm_guided_dominating_set(graph, 64);
			assert_dominating_set_theorem_bound(graph, &dominating_set);
		});
	}
}
