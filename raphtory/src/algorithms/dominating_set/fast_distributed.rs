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
    pub undominated_count: usize,
    pub undominated_count_rounded_down: usize,
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
            let undominated_count = s.neighbours().iter().count() + 1;
            let s_state = s.get_mut();
            s_state.is_active = true;
            s_state.is_covered = false;
            s_state.undominated_count = undominated_count;
            s_state.undominated_count_rounded_down = 0;
            s_state.support_count = 0;
            s_state.is_candidate = false;
            s_state.candidate_count = 0;
            Step::Continue
        },
    );

    let step2: ATask<G, ComputeStateVec, Fdds, _> = ATask::new(
        move |s: &mut EvalNodeView<'_, '_, &'_ G, Fdds, ComputeStateVec>| {
            let num_undominated = s.prev().undominated_count;
            if num_undominated == 0 {
                return Step::Done;
            }
            let undominated_count_rounded_down = 1_usize << num_undominated.ilog2();
            let s_state = s.get_mut();
            s_state.undominated_count_rounded_down = undominated_count_rounded_down;
            Step::Continue
        },
    );

    let step3: ATask<G, ComputeStateVec, Fdds, _> = ATask::new(
        move |s: &mut EvalNodeView<'_, '_, &'_ G, Fdds, ComputeStateVec>| {
            if s.prev().undominated_count == 0 {
                return Step::Done;
            }
            let my_rounded = s.prev().undominated_count_rounded_down;
            let mut max_undominated_count_rounded_down = s.prev().undominated_count_rounded_down;
            for n in s.neighbours() {
                max_undominated_count_rounded_down = max_undominated_count_rounded_down.max(n.prev().undominated_count_rounded_down);  
                for nn in n.neighbours() {
                    max_undominated_count_rounded_down = max_undominated_count_rounded_down.max(nn.prev().undominated_count_rounded_down);
                }
            }
            let s_state = s.get_mut();
            s_state.is_active = my_rounded == max_undominated_count_rounded_down;
            Step::Continue
        },
    );

    let step4: ATask<G, ComputeStateVec, Fdds, _> = ATask::new(
        move |s: &mut EvalNodeView<'_, '_, &'_ G, Fdds, ComputeStateVec>| {
            if s.prev().undominated_count== 0 {
                return Step::Done;
            }
            let mut support_count = s.neighbours().iter().filter(|n| n.prev().is_active).count();
            if s.prev().is_active {
                support_count += 1;
            } 
            let s_state = s.get_mut();
            s_state.support_count = support_count;
            Step::Continue
        },
    );

    let step5: ATask<G, ComputeStateVec, Fdds, _> = ATask::new(
        move |s: &mut EvalNodeView<'_, '_, &'_ G, Fdds, ComputeStateVec>| {
            if s.prev().undominated_count == 0 {
                return Step::Done;
            }
            let is_active = s.prev().is_active;
            let is_candidate = if is_active {
                let s_prev = s.prev();
                let max_support_count_exclusive = s.neighbours().iter().filter(|n| !n.prev().is_covered).map(|n| n.prev().support_count).max().unwrap_or(0); 
                let max_support_count = if !s_prev.is_covered { max_support_count_exclusive.max(s_prev.support_count) } else { max_support_count_exclusive };
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
            if s.prev().undominated_count == 0 {
                return Step::Done;
            }
            let mut candidate_count = s.neighbours().iter().filter(|n| n.prev().is_candidate).count();
            if s.prev().is_candidate {
                candidate_count += 1;
            }
            let s_state = s.get_mut();
            s_state.candidate_count = candidate_count;
            Step::Continue
        },
    );

    let step7: ATask<G, ComputeStateVec, Fdds, _> = ATask::new(
        move |s: &mut EvalNodeView<'_, '_, &'_ G, Fdds, ComputeStateVec>| {
            let num_undominated = s.prev().undominated_count;
            if num_undominated == 0 {
                return Step::Done;
            }
            let is_candidate = s.prev().is_candidate;
            let should_dominate = if is_candidate {
                let mut candidate_sum = s
                    .neighbours()
                    .iter()
                    .filter(|n| !n.prev().is_covered)
                    .map(|n| n.prev().candidate_count)
                    .sum::<usize>();
                if !s.prev().is_covered {
                    candidate_sum += s.prev().candidate_count;
                }
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
            if s.prev().undominated_count == 0 {
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
            let num_undominated = s.prev().undominated_count;
            if num_undominated == 0 {
                return Step::Done;
            }
            let mut new_undominated_count = s
                .neighbours()
                .iter()
                .filter(|n| !n.prev().is_covered)
                .count();
            if !s.prev().is_covered {
                new_undominated_count += 1;
            }

            let s_state = s.get_mut();
            s_state.undominated_count = new_undominated_count;
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

