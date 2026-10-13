# JDSA revision results (frozen after Phase 1B)

Numerical corrections and reviewer-requested analyses for the major revision of
"A Reproducible Event-Fixed Protocol for Evaluating Financial Stress Detectors".
Every file here is written by a script in `scripts/jdsa_revision/`; no submitted result
file and no submitted event file outside this folder was changed.

Start with `FROZEN_RESULTS.md` (the numerical freeze) and `final_audit.csv` (every audited
manuscript number with its submitted and revised value). Manuscript and response-letter edits
take their values from these files.

Reproduce everything with

    python scripts/jdsa_revision/run_all.py
    python scripts/jdsa_revision/run_all.py --panel submitted --out results/jdsa_revision/pre_E03_fix --skip t3 t12
    python scripts/jdsa_revision/compare_runs.py results/jdsa_revision/pre_E03_fix results/jdsa_revision
    python scripts/jdsa_revision/t12_audit.py
    python scripts/jdsa_revision/t15_freeze.py

The first command uses the corrected event panel `jdsa_revision_v2`
(`data/jdsa_revision/events_spy_seed_v2.csv`, E03 dated 2002-07-22, the first trading day after
WorldCom's Sunday 2002-07-21 Chapter 11 filing). The second reruns the same code on the
submitted panel; `pre_E03_fix_vs_post_E03_fix.csv` lists every output value that the E03
correction changes. The revision pipeline always scores observable events only
(SPY 13, QQQ 13, IWM 12, HYG 10, TLT 10).

| Folder | Script | Reviewer item | Contents |
|---|---|---|---|
| `t0_event_panel/` | `t0_event_panel.py` | R2 M4(b) | Active panel, per-asset mapping, differences from the submitted panel |
| `t1_denominators/` | `t1_observable_denominators.py` | R1 m4, R2 M4(c) | Tables 5, 8, 9, 11, 12 with observable denominators; Fig. 2; changes against the submitted files |
| `t2_evaluation_periods/` | `t2_evaluation_periods.py` | R1 F1-F2, R2 M2-M3 | Derived operational start; Table A (post-calibration retrospective, from 2016-11-09); Table B (post-calibration causal, from the derived start; legacy file and row names, see the notes below); full-period reference |
| `t3_causality/` | `t3_causality.py` | R1 F1, R2 M2 | Forward reach and perturbation tests of every rule, including dates from the operational start |
| `t4_operating_curves/` | `t4_operating_curves.py` | R1 main, R2 M1, m3 | Operating curves and matched-FPR comparisons on all five assets; figures |
| `t5_common_event_delay/` | `t5_common_event_delay.py` | R2 m4 | Delays on events detected by both methods |
| `t6_uncertainty/` | `t6_uncertainty.py` | R1 m7 | Paired event, common-event and circular block bootstraps (20, 60, 121 days) for both comparisons on all assets |
| `t7_cusum/` | `t7_cusum_diagnostic.py` | R1 m1, R2 M5 | Behaviour of the submitted CUSUM baseline |
| `t8_z_dominance/` | `t8_z_dominance.py` | R1 F3, R2 m2 | x_t against z_t on every asset |
| `t9_mechanism/` | `t9_mechanism.py` | R1 F4 | Section 4.4.1 and Appendix B quantities against the manuscript |
| `t10_parameters/` | `t10_parameters.py` | R2 m3 | Parameter inventory, percentile sensitivity, development-history note |
| `t11_event_provenance/` | `t11_event_provenance.py` | R2 M4(a)-(b) | Documented provenance of all 13 dates with sources, window gaps, candidate-list search |
| `t13_paper_tables/` | `t13_paper_tables.py` | all | Every event-dependent submitted SPY table (3, 4, 6, 7, 10, 13-16 and related results) under the active panel |
| `t14_figures/` | `t14_figures.py` | all | Figs. 1, 4, 5, 6, 7 and a candidate Fig. 8 from the revision outputs |
| `t16_followups/` | `t16_followups.py` | R1 main, F1; R2 M2 | Fully causal rule at a matched false positive rate (full period and post-calibration); event detections by event-window width |
| `t12_audit/` | `t12_audit.py` | all | Manuscript numbers to change, unexpected differences, integrity checks |
| `t15_freeze/` | `t15_freeze.py` | all | Nesting check; writes `FROZEN_RESULTS.md` |
| `pre_E03_fix/` | same scripts, `--panel submitted` | | Comparison run on the submitted panel (not for citation) |

Notes for reading the outputs:

* Differences are the second method minus the first, as named in each file.
* Under nested alarms (every confirmation alarm is also a volatility alarm) the signs of the
  recall, FPR, detection-rate and common-event delay differences are fixed by construction.
  Rows where this applies are flagged `sign_fixed_by_nesting`.
* Table B uses only calibration quantities fitted on prices before its first scored day. Its five
  events were part of the development panel, so it is not a design-blind holdout. The
  full-period causal variant (Table 16) uses constants fitted on later prices for its
  calibration-period alarms.
* Legacy names: the file names `table_B_prospective_operational*`, the period label
  `B_prospective_operational`, and the `t5_common_event_delay` rows labelled `prospective operational`
  keep names from before the revision so that the frozen outputs stay comparable. They denote
  what the manuscript now calls the post-calibration causal evaluation; they do not denote a
  design-blind prospective holdout.
* `t10_parameters/development_history_note.md` is generated from the local `archive/` folder,
  which is not in the public repository.
