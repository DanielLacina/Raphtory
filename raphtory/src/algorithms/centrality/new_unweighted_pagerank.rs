use std::{collections::HashMap, sync::{Arc, atomic::AtomicU64}};

use crate::{
    core::state::{accumulator_id::accumulators, compute_state::ComputeStateVec},
    db::{
        api::{
            state::NodeState,
            view::{NodeViewOps, StaticGraphViewOps, filter_ops::NodeSelect},
        }, graph::views::{filter::model::{degree_filter::DegreeFilterFactory, property_filter::ops::PropertyFilterOps}, node_subgraph::NodeSubgraph}, task::{
            context::Context,
            task::{ATask, Job, Step},
            task_runner::TaskRunner,
        }
    },
    prelude::GraphViewOps
};
use num_traits::abs;
use crate::prelude::NodeFilter;

#[derive(Clone, Debug, Default)]
struct PageRankState {
    score: f64,
    out_degree: usize,
    one_degree_out_neighbor_score: f64 
}

impl PageRankState {
    fn new(num_nodes: usize) -> Self {
        Self {
            score: 1f64 / num_nodes as f64,
            one_degree_out_neighbor_score: 1f64 / num_nodes as f64,
            out_degree: 0,
        }
    }

    fn reset(&mut self) {
        self.score = 0f64;
        self.one_degree_out_neighbor_score = 0f64;
    }
}

/// PageRank Algorithm:
/// PageRank shows how important a node is in a graph.
///
/// # Arguments
///
/// - `g`: A GraphView object
/// - `iter_count`: Number of iterations to run the algorithm for
/// - `threads`: Number of threads to use for parallel execution
/// - `tol`: The tolerance value for convergence
/// - `use_l2_norm`: Whether to use L2 norm for convergence
/// - `damping_factor`: Probability of likelihood the spread will continue
///
/// # Returns
///
/// An [AlgorithmResult] object containing the mapping from node ID to the PageRank score of the node
///
pub fn unweighted_page_rank<G: StaticGraphViewOps>(
    g: &G,
    iter_count: Option<usize>,
    threads: Option<usize>,
    tol: Option<f64>,
    use_l2_norm: bool,
    damping_factor: Option<f64>,
) -> NodeState<'static, f64, NodeSubgraph<G>> {
    let n = g.count_nodes();

    let mut ctx: Context<G, ComputeStateVec> = g.into();

    let mut one_degree_neighbors_map = HashMap::new();

    let mut zero_degree_sink_contributor_count = 0;

    for node in g.nodes() {
        if node.degree() == 0 {
            zero_degree_sink_contributor_count += 1;
            continue;
        }
        let one_degree_in_neighbor_count = node.in_neighbours().iter().filter(|n| n.degree() == 1).count();
        let one_degree_out_neighbor_count = node.out_neighbours().iter().filter(|n| n.degree() == 1).count();
        one_degree_neighbors_map.insert(node.node, (one_degree_in_neighbor_count, one_degree_out_neighbor_count));
    }

    let one_degree_neighbors = Arc::new(one_degree_neighbors_map);

    let subgraph = g.subgraph(g.nodes().select(NodeFilter.degree().gt(1)).unwrap());

    let tol: f64 = tol.unwrap_or(0.000001f64);
    let damp = damping_factor.unwrap_or(0.85);
    let iter_count = iter_count.unwrap_or(20);
    let teleport_prob = (1f64 - damp) / n as f64;
    let factor = damp / n as f64;

    let in_degree_zero_score = Arc::new(atomic_float::AtomicF64::new(0.0));
    let zero_degree_sink_contributor_count = Arc::new(AtomicU64::new(zero_degree_sink_contributor_count as u64));

    let max_diff = accumulators::sum::<f64>(2);

    let total_sink_contribution = accumulators::sum::<f64>(4);

    ctx.global_agg_reset(max_diff);

    ctx.global_agg_reset(total_sink_contribution);

    let step1 = ATask::new(move |s| {
        let out_degree = s.out_degree();
        let state: &mut PageRankState = s.get_mut();
        state.out_degree = out_degree;
        Step::Continue
    });

    let in_degree_zero_score_step2 = Arc::clone(&in_degree_zero_score);
    let one_degree_neighbors_step2 = Arc::clone(&one_degree_neighbors);
    let step2: ATask<G, ComputeStateVec, PageRankState, _> = ATask::new(move |s| {
        // reset score
        {
            let state: &mut PageRankState = s.get_mut();
            state.reset();
        }

        for t in s.in_neighbours() {
            let prev = t.prev();

            s.get_mut().score += prev.score / prev.out_degree as f64;
        }

        let (one_degree_in_neighbors, _) = one_degree_neighbors_step2.get(&s.node).unwrap_or(&(0, 0));

        let in_degree_zero_score = in_degree_zero_score_step2.load(std::sync::atomic::Ordering::Relaxed);

        s.get_mut().score += (*one_degree_in_neighbors as f64) * in_degree_zero_score;   

        s.get_mut().score *= damp;

        s.get_mut().score += teleport_prob;

        let one_degree_out_neighbor_score = (s.prev().score / s.prev().out_degree as f64 * damp) + teleport_prob; 
        s.get_mut().one_degree_out_neighbor_score = one_degree_out_neighbor_score;
        Step::Continue
    });

    let one_degree_neighbors_step3 = Arc::clone(&one_degree_neighbors);
    let step3 = ATask::new(move |s| {
        let state: &mut PageRankState = s.get_mut();

        if state.out_degree == 0 {
            let curr = s.prev().score;

            let ts_contrib = factor * curr;
            s.global_update(&total_sink_contribution, ts_contrib);
        } else {
            let (_, one_degree_out_neighbors) = one_degree_neighbors_step3.get(&s.node).unwrap_or(&(0, 0));
            let one_degree_out_neighbor_score = s.prev().one_degree_out_neighbor_score; 
            let contrib = (*one_degree_out_neighbors as f64) * factor * one_degree_out_neighbor_score;
            s.global_update(&total_sink_contribution, contrib);
        }

        Step::Continue
    });

    let zero_degree_sink_contributor_count_step4 = Arc::clone(&zero_degree_sink_contributor_count);
    let in_degree_zero_score_step4 = Arc::clone(&in_degree_zero_score);
    let step4 = ATask::new(move |s| {
        //read total sink contribution
        let total_sink_contribution = s
            .read_global_state(&total_sink_contribution)
            .unwrap_or_default();
        let filtered_out_contributor_count = zero_degree_sink_contributor_count_step4.load(std::sync::atomic::Ordering::Relaxed) as f64;
        let in_degree_zero_score = in_degree_zero_score_step4.load(std::sync::atomic::Ordering::Relaxed);
        let (curr_score, curr_one_degree_out_neighbor_score) = {
            // update local score with total sink contribution
            let state: &mut PageRankState = s.get_mut();
            state.score += total_sink_contribution + filtered_out_contributor_count * factor * in_degree_zero_score;
            state.one_degree_out_neighbor_score += total_sink_contribution + filtered_out_contributor_count * factor * in_degree_zero_score;

            (state.score, state.one_degree_out_neighbor_score)
        };

        // update global max diff
        let prev_score = s.prev().score;
        let md = if use_l2_norm {
            f64::powi(abs(prev_score - curr_score), 2)
        } else {
            abs(prev_score - curr_score)
        };

        s.global_update(&max_diff, md);

        let prev_one_degree_out_neighbor_score = s.prev().one_degree_out_neighbor_score;
        let md = if use_l2_norm {
            f64::powi(abs(prev_one_degree_out_neighbor_score - curr_one_degree_out_neighbor_score), 2)
        } else {
            abs(prev_one_degree_out_neighbor_score - curr_one_degree_out_neighbor_score)
        };

        s.global_update(&max_diff, md);

        Step::Continue
    });


    let zero_degree_sink_contributor_count_step5 = Arc::clone(&zero_degree_sink_contributor_count);
    let prev_in_degree_zero_score_step5 = Arc::clone(&in_degree_zero_score);
    let step5 = Job::Check(Box::new(move |state| {
        let total_sink_contribution = state.read(&total_sink_contribution);
        let zero_degree_sink_contributor_count = zero_degree_sink_contributor_count_step5.load(std::sync::atomic::Ordering::Relaxed) as f64;
        let prev_in_degree_zero_score = prev_in_degree_zero_score_step5.load(std::sync::atomic::Ordering::Relaxed);
        let cur_in_degree_zero_score = teleport_prob + total_sink_contribution + zero_degree_sink_contributor_count * factor * prev_in_degree_zero_score;
        prev_in_degree_zero_score_step5.store(cur_in_degree_zero_score, std::sync::atomic::Ordering::Relaxed);
        let diff_in_degree_zero_score = abs(prev_in_degree_zero_score - cur_in_degree_zero_score);
        let max_diff_val = state.read(&max_diff).max(diff_in_degree_zero_score);
        let cont = if use_l2_norm {
            let sum_d = f64::sqrt(max_diff_val);
            (sum_d) > tol * n as f64
        } else {
            (max_diff_val) > tol * n as f64
        };
        if cont {
            Step::Continue
        } else {
            Step::Done
        }
    }));

    let mut runner: TaskRunner<G, _> = TaskRunner::new(ctx);

    let num_nodes = g.count_nodes();

    runner.run(
        vec![Job::new(step1)],
        vec![Job::new(step2), Job::new(step3), Job::new(step4), step5],
        Some(vec![PageRankState::new(num_nodes); num_nodes]),
        |_, _, _, local, _| NodeState::new_from_eval_mapped(subgraph.clone(), local, |v| v.score),
        threads,
        iter_count,
        None,
        None,
    )
}