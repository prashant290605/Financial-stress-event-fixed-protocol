# Cover Letter

**To:** The Editor-in-Chief, *International Journal of Data Science and Analysis*
**Date:** [INSERT SUBMISSION DATE]
**Manuscript:** A Reproducible Event-Fixed Protocol for Evaluating Financial Stress Detectors
**Authors:** Prashant Singh (corresponding), Pranav Singh
**Affiliation:** Department of Mathematics, Indian Institute of Technology Ropar, Rupnagar, Punjab 140001, India

---

Dear Editor-in-Chief,

We submit the manuscript "A Reproducible Event-Fixed Protocol for Evaluating Financial Stress Detectors" for consideration as a research article in the *International Journal of Data Science and Analysis*.

**What the paper reports.** Detectors for financial stress are usually compared by pointwise scores computed over alarm days. We show that this choice, made before any data is scored, can decide the outcome. Under a protocol that fixes a panel of 13 documented U.S. equity stress episodes before any detector output is inspected, a volatility-plus-instability confirmation rule and a plain volatility percentile trigger swap places: on SPY the confirmation rule cuts the false positive rate from 0.044 to 0.005 and raises precision from 0.640 to 0.886, yet detects 8 of 13 episodes against 10 and confirms the Lehman episode 40 trading days later. Both readings describe the same two output series over the same 24 years.

We do not stop at the observation. The paper traces the reversal to a specific mechanism: the confirmation gate runs on a self-normalizing input, so it responds to repricing that outruns the trailing volatility scale and is blind to elevation that scale has already absorbed. A per-event component diagnosis and a controlled synthetic experiment, in which two engineered episodes of identical peak volatility differ only in whether the trailing scale can absorb them, confirm both directions of that prediction. We then convert the tradeoff into a decision rule practitioners can apply: a crossover cost ratio, reported for five assets and three delay penalties, above which each detector is the cheaper choice.

**Fit with the journal.** The work sits squarely within the journal's scope in machine learning, statistics, and data analysis applied to real-world data at scale. Its contribution is an evaluation methodology rather than a new detector, which speaks to a problem the journal's readership encounters directly: benchmark and scoring design determining apparent progress. The manuscript engages the anomaly-detection benchmarking literature (including the Numenta Anomaly Benchmark and Wu and Keogh's critique of flawed benchmarks) and connects it to the exception-based evaluation traditions of financial risk management and the early-warning signals literature. We also place substantial weight on reproducibility: the complete pipeline, the archived data snapshots used for every reported number, the fixed event panel with documented sources, the random seed, and scripts that regenerate every table and figure are publicly available at https://github.com/prashant290605/Financial-stress-event-fixed-protocol.

**Declarations.** All authors have read and approved the submitted manuscript and agree to its submission to this journal. The manuscript is original, has not been published previously, and is not under consideration for publication elsewhere, in whole or in part. The authors declare no conflicts of interest, financial or non-financial. The research received no specific funding. No human participants, human data, or animals were involved, so ethics approval was not required. As stated in the manuscript, generative artificial intelligence was used for assistance with formatting and language presentation only; the authors reviewed and edited the resulting text and remain responsible for the study design, analyses, results, interpretations, and final manuscript.

**Submitted files.** manuscript_sciencepg.tex and manuscript_sciencepg.pdf, supporting_information.tex and supporting_information.pdf, ten figure files, and the journal logo file.

We hope the manuscript is suitable for publication in the *International Journal of Data Science and Analysis* and look forward to the reviewers' comments.

Yours sincerely,

Prashant Singh
Corresponding author
Department of Mathematics, Indian Institute of Technology Ropar
Rupnagar, Punjab 140001, India
2023mcb1309@iitrpr.ac.in

On behalf of both authors.

---

## Before sending, check these two things

1. **Submission date** at the top.
2. **Special issue.** This letter is written for a regular submission. The
   journal's special-issue listing was not publicly reachable at the time of
   writing. If you are submitting to a named special issue, add one sentence
   after the first paragraph: "We submit this manuscript for consideration in
   the special issue on [EXACT SPECIAL ISSUE TITLE]." Use the exact title as
   printed on the journal's call for papers.
