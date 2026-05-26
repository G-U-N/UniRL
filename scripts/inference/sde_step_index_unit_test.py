"""Unit test for the SDE-path _step_index bookkeeping fix.

Reproduces the bug at the scheduler level — no model load required.

Bug:
    pipeline calls `self.scheduler.set_begin_index(0)`, then for the first
    `num_sde` steps it bypasses `scheduler.step()` (using `sde_step_with_logprob`
    instead) so `_step_index` stays None. When the first `scheduler.step()` is
    finally called for the deterministic tail, `_init_step_index` reads
    `_begin_index=0` and the tail walks `sigmas[0,1,2,3]` instead of
    `sigmas[num_sde, num_sde+1, ...]`.

Fix:
    in the SDE branch, also bump `_step_index` so the tail picks up where the
    SDE prefix left off.

This test runs the schedule explicitly with and without the fix and prints the
sigmas each deterministic step reads, then asserts the fixed version reads the
correct (non-overlapping) tail.
"""

import argparse

import torch
from diffusers import FlowMatchEulerDiscreteScheduler


def reset_scheduler(num_inference_steps=8, shift=3.0):
    sched = FlowMatchEulerDiscreteScheduler(shift=shift)
    sched.set_timesteps(num_inference_steps=num_inference_steps, device="cpu")
    sched.set_begin_index(0)
    return sched


def simulate(num_inference_steps, num_sde, apply_fix):
    sched = reset_scheduler(num_inference_steps=num_inference_steps)
    timesteps = sched.timesteps
    sigmas_before = sched.sigmas.tolist()

    sample = torch.zeros((1, 4, 64, 64), dtype=torch.float32)
    deterministic_sigmas_seen = []
    for i, t in enumerate(timesteps):
        if i < num_sde:
            # SDE branch — does NOT touch scheduler.step internally,
            # only reads sigmas via index_for_timestep.
            _ = sched.sigmas[sched.index_for_timestep(t)]
            if apply_fix:
                if sched.step_index is None:
                    sched._init_step_index(t)
                sched._step_index += 1
        else:
            # Deterministic branch — calls scheduler.step which internally
            # initializes _step_index from _begin_index if it's None.
            sigma_before_step = sched.sigmas[sched.step_index if sched.step_index is not None else sched._begin_index]
            sample = sched.step(torch.zeros_like(sample), t, sample, return_dict=False)[0]
            deterministic_sigmas_seen.append(float(sigma_before_step))
    return sigmas_before, deterministic_sigmas_seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_inference_steps", type=int, default=8)
    parser.add_argument("--num_sde", type=int, default=4)
    args = parser.parse_args()

    print(f"num_inference_steps={args.num_inference_steps}  num_sde={args.num_sde}\n")

    sigmas, broken = simulate(args.num_inference_steps, args.num_sde, apply_fix=False)
    _,      fixed  = simulate(args.num_inference_steps, args.num_sde, apply_fix=True)

    sigmas = [round(s, 4) for s in sigmas]
    expected_tail = [round(s, 4) for s in sigmas[args.num_sde : args.num_inference_steps]]
    broken_rounded = [round(s, 4) for s in broken]
    fixed_rounded  = [round(s, 4) for s in fixed]

    print(f"full sigmas:        {sigmas}")
    print(f"expected tail [num_sde..N): {expected_tail}")
    print(f"broken  (no fix) tail:     {broken_rounded}")
    print(f"fixed   (with fix) tail:   {fixed_rounded}")

    assert broken_rounded != expected_tail, "expected the broken path to read wrong sigmas, but it matched"
    assert fixed_rounded == expected_tail, (
        f"fix did NOT restore correct schedule: got {fixed_rounded}, expected {expected_tail}"
    )
    # And the broken path should be reading the prefix sigmas (sigmas[0:num_inference_steps-num_sde]):
    expected_broken = [round(s, 4) for s in sigmas[: args.num_inference_steps - args.num_sde]]
    assert broken_rounded == expected_broken, (
        f"broken path should re-read sigmas[0:{args.num_inference_steps - args.num_sde}]={expected_broken}, "
        f"got {broken_rounded}"
    )
    print("\nPASS: fix restores correct sigma schedule for the deterministic tail.")
    print("      broken path was reading sigmas[0:N-num_sde] instead — exactly the bug observed.")


if __name__ == "__main__":
    main()
