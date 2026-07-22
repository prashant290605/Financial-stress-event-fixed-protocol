# Cover Letter

**To:** The Editor-in-Chief, *International Journal of Data Science and Analytics*
**Date:** 22 July 2026
**Manuscript:** A Reproducible Event-Fixed Protocol for Evaluating Financial Stress Detectors
**Contribution type:** Regular paper
**Authors:** Prashant Singh (corresponding), Pranav Singh
**Affiliation:** Department of Mathematics, Indian Institute of Technology Ropar, Rupnagar, Punjab 140001, India

---

Dear Editor-in-Chief,

We submit "A Reproducible Event-Fixed Protocol for Evaluating Financial Stress Detectors" for consideration as a regular paper in the *International Journal of Data Science and Analytics*.

**The problem and the contribution.** Detectors for financial stress are compared using scores computed over alarm days. That scoring choice is made before any data is examined, and we show it can decide which detector appears better. Our contribution is an evaluation protocol, not a detector. The protocol fixes a panel of 13 documented U.S. equity stress episodes before any detector output is inspected, maps them to each asset's trading calendar, and reports pointwise, event-level, delay, and cost-sensitive metrics side by side.

Applied to a volatility percentile trigger and a volatility-plus-instability confirmation rule on SPY, the two detectors trade places. Confirmation cuts the false positive rate from 0.044 to 0.005 and raises precision from 0.640 to 0.886, yet detects 8 of 13 episodes against 10 and confirms the Lehman episode 40 trading days later. Both statements describe the same two output series over the same 24 years; only the scoring rule differs.

Three things make this more than an observation. First, the reversal reproduces when the broad trigger is the Cboe Volatility Index rather than a rule we designed, so it belongs to the evaluation rather than to a hand-built detector. Second, we identify the mechanism: the confirmation gate runs on a self-normalizing input, so it responds to repricing that outruns the trailing volatility scale and is blind to elevation that scale has absorbed. A per-event component diagnosis and a controlled synthetic experiment confirm both directions of this prediction. Third, we convert the tradeoff into a crossover cost ratio, 0.021 on SPY, above which each detector is the cheaper choice, reported for five assets and three delay penalties.

**Fit with the journal.** The paper is an applied data-science methods contribution: it concerns benchmark and scoring design, a problem the journal's readership meets whenever detection methods are compared. It engages the anomaly-detection benchmarking literature, including the Numenta Anomaly Benchmark and Wu and Keogh's critique of flawed benchmarks, and connects it to the exception-based evaluation traditions of financial risk management and the signals approach to early warning. We place substantial weight on reproducibility: the pipeline, the archived data snapshots underlying every reported number, the fixed event panel with documented sources, the random seed, and scripts that regenerate every table and figure are publicly available at https://github.com/prashant290605/Financial-stress-event-fixed-protocol.

We also report findings that are unfavourable to our own illustrative detector, including that its composite weights are effectively inert and that a fully causal version loses most of its selectivity. We regard this as the protocol doing its job.

**Declarations.** All authors have read and approved the submitted manuscript. The work described has not been published before and is not under consideration for publication elsewhere, in whole or in part, at any journal or conference. The authors declare no competing interests, financial or non-financial. The research received no specific funding. No human participants, human data, or animals were involved, so ethics approval was not required. A data availability statement appears in the manuscript; all market data are publicly available and the exact snapshots are archived in the repository. As stated in the manuscript, generative artificial intelligence was used for assistance with formatting and language presentation; the authors reviewed and edited the resulting text and remain responsible for the study design, analyses, results, interpretations, and final manuscript.

**Submitted files.** manuscript.tex, manuscript.pdf, sn-jnl.cls, sn-mathphys-num.bst, sn-bibliography.bib, and ten figure files Fig1.pdf through Fig10.pdf, all in a single flat folder with no subdirectories.

We hope the manuscript is suitable for the journal and look forward to the reviewers' comments.

Yours sincerely,

Prashant Singh
Corresponding author
Department of Mathematics, Indian Institute of Technology Ropar
Rupnagar, Punjab 140001, India
2023mcb1309@iitrpr.ac.in

On behalf of both authors.

---

## Before sending

1. The date is set to 22 July 2026 — change it if you submit on a different day.
2. This letter is written for a **regular submission**. The finance special
   issue that prompted the earlier draft closed in 2021; if you later target a
   currently open JDSA call, add after the first paragraph: "We submit this
   manuscript for consideration in the special issue on [EXACT TITLE]."
3. Springer collects **Author Contribution** and **Competing Interest**
   statements through the submission interface as well as in the manuscript.
   Only what you enter in the interface reaches the published version, so enter
   both there even though they also appear in the paper.
