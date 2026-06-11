use crate::{
    core::{
        entities::VID,
        state::{accumulator_id::accumulators::hash_set, compute_state::ComputeStateVec},
    },
    db::{
        api::{
            state::{GenericNodeState, TypedNodeState},
            view::{NodeViewOps, StaticGraphViewOps},
        },
        task::{
            context::Context,
            node::eval_node::EvalNodeView,
            task::{ATask, Job, Step},
            task_runner::TaskRunner,
        },
    },
};
use serde::{Deserialize, Serialize};

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug, Default)]
pub struct Fdds {
    pub is_active: bool,
    pub is_covered: bool,
    pub num_undominated_neighbours: usize,
    pub num_undominated_neighbours_rounded_down: usize,
    pub is_candidate: bool,
    pub candidate_count: usize,
    pub support_count: usize,
}

pub fn find_dominating_set<G: StaticGraphViewOps>(
    g: &G,
    iter_count: usize,
    threads: Option<usize>,
) -> TypedNodeState<'static, Fdds, G> {
    let mut ctx: Context<G, ComputeStateVec> = g.into();
    let dominating_set = hash_set::<VID>(0);
    ctx.global_agg(dominating_set);
    let covered_set = hash_set::<VID>(1);
    ctx.global_agg(covered_set);


    let step1: ATask<G, ComputeStateVec, Fdds, _> = ATask::new(
        move |s: &mut EvalNodeView<'_, '_, &'_ G, Fdds, ComputeStateVec>| {
            let undominated = s.neighbours().iter().count();
            let s_state = s.get_mut();
            s_state.is_active = true;
            s_state.is_covered = false;
            s_state.num_undominated_neighbours = undominated;
            s_state.num_undominated_neighbours_rounded_down = 0;
            s_state.support_count = 0;
            s_state.is_candidate = false;
            s_state.candidate_count = 0;
            Step::Continue
        },
    );

    let step2: ATask<G, ComputeStateVec, Fdds, _> = ATask::new(
        move |s: &mut EvalNodeView<'_, '_, &'_ G, Fdds, ComputeStateVec>| {
            let num_undominated = s.prev().num_undominated_neighbours;
            if num_undominated == 0 {
                return Step::Done;
            }
            let num_undominated_neighbours_rounded_down = 1_usize << num_undominated.ilog2();
            let s_state = s.get_mut();
            s_state.num_undominated_neighbours_rounded_down = num_undominated_neighbours_rounded_down;
            Step::Continue
        },
    );

    let step3: ATask<G, ComputeStateVec, Fdds, _> = ATask::new(
        move |s: &mut EvalNodeView<'_, '_, &'_ G, Fdds, ComputeStateVec>| {
            let num_undominated = s.prev().num_undominated_neighbours;
            if num_undominated == 0 {
                return Step::Done;
            }
            let my_rounded = s.prev().num_undominated_neighbours_rounded_down;
            let max_undominated_neighbours_rounded_down = s
                .neighbours()
                .neighbours()
                .iter()
                .map(|n| n.prev().num_undominated_neighbours_rounded_down)
                .max()
                .unwrap();
            let s_state = s.get_mut();
            s_state.is_active = my_rounded == max_undominated_neighbours_rounded_down;
            Step::Continue
        },
    );

    let step4: ATask<G, ComputeStateVec, Fdds, _> = ATask::new(
        move |s: &mut EvalNodeView<'_, '_, &'_ G, Fdds, ComputeStateVec>| {
            if s.prev().num_undominated_neighbours == 0 {
                return Step::Done;
            }
            let support_count = s.neighbours().iter().filter(|n| n.prev().is_active).count();
            let s_state = s.get_mut();
            s_state.support_count = support_count;
            Step::Continue
        },
    );

    let step5: ATask<G, ComputeStateVec, Fdds, _> = ATask::new(
        move |s: &mut EvalNodeView<'_, '_, &'_ G, Fdds, ComputeStateVec>| {
            if s.prev().num_undominated_neighbours == 0 {
                return Step::Done;
            }
            let is_active = s.prev().is_active;
            let is_candidate = if is_active {
                let max_support_count = s
                    .neighbours()
                    .iter()
                    .filter(|n| !n.prev().is_covered)
                    .map(|n| n.prev().support_count)
                    .max()
                    .unwrap();
                let p = 1.0 / max_support_count as f64;
                rand::random::<f64>() < p
            } else {
                false
            };
            let s_state = s.get_mut();
            s_state.is_candidate = is_candidate;
            Step::Continue
        },
    );

    let step6: ATask<G, ComputeStateVec, Fdds, _> = ATask::new(
        move |s: &mut EvalNodeView<'_, '_, &'_ G, Fdds, ComputeStateVec>| {
            if s.prev().num_undominated_neighbours == 0 {
                return Step::Done;
            }
            let candidate_count = s.neighbours().iter().filter(|n| n.prev().is_candidate).count();
            let s_state = s.get_mut();
            s_state.candidate_count = candidate_count;
            Step::Continue
        },
    );

    let step7: ATask<G, ComputeStateVec, Fdds, _> = ATask::new(
        move |s: &mut EvalNodeView<'_, '_, &'_ G, Fdds, ComputeStateVec>| {
            let num_undominated = s.prev().num_undominated_neighbours;
            if num_undominated == 0 {
                return Step::Done;
            }
            let is_candidate = s.prev().is_candidate;
            let should_dominate = if is_candidate {
                let candidate_sum = s
                    .neighbours()
                    .iter()
                    .filter(|n| !n.prev().is_covered)
                    .map(|n| n.prev().candidate_count)
                    .sum::<usize>();
                candidate_sum <= 3 * num_undominated
            } else {
                false
            };
            if should_dominate {
                let node = s.node;
                s.global_update(&dominating_set, node);
                s.global_update(&covered_set, node);
                for n in s.neighbours().iter() {
                    s.global_update(&covered_set, n.node);
                }
            }
            Step::Continue
        },
    );

    let step8: ATask<G, ComputeStateVec, Fdds, _> = ATask::new(
        move |s: &mut EvalNodeView<'_, '_, &'_ G, Fdds, ComputeStateVec>| {
            let num_undominated = s.prev().num_undominated_neighbours;
            if num_undominated == 0 {
                return Step::Done;
            }
            let is_covered = s.read(&covered_set).contains(&s.node);
            let s_state = s.get_mut();
            s_state.is_covered = is_covered;
            Step::Continue 
        },
    );


    let step9: ATask<G, ComputeStateVec, Fdds, _> = ATask::new(
        move |s: &mut EvalNodeView<'_, '_, &'_ G, Fdds, ComputeStateVec>| {
            if s.prev().num_undominated_neighbours == 0 {
                return Step::Done;
            }
            let is_covered = s
                .read_global_state(&covered_set)
                .map(|covered| covered.contains(&s.node))
                .unwrap_or(false);
            let num_undominated_neighbours = s
                .neighbours()
                .iter()
                .filter(|n| !n.prev().is_covered)
                .count();
            let s_state = s.get_mut();
            s_state.is_covered = is_covered;
            s_state.num_undominated_neighbours = num_undominated_neighbours;
            Step::Continue
        },
    );


    let mut runner: TaskRunner<G, _> = TaskRunner::new(ctx);
    runner.run(
        vec![Job::new(step1)],
        vec![
            Job::new(step2),
            Job::new(step3),
            Job::new(step4),
            Job::new(step5),
            Job::new(step6),
            Job::new(step7),
            Job::new(step8),
            Job::new(step9),
        ],
        None,
        |_, _, _, local, index| {
            TypedNodeState::new(GenericNodeState::new_from_eval_with_index(
                g.clone(),
                local,
                index,
                None,
            ))
        },
        threads,
        iter_count,
        None,
        None,
    )
}