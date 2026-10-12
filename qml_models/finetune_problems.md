# BiomedCLIP ViT Fine-Tuning Plot Flaw Report

## Scope and evidence basis

This report documents the problems detected while reviewing every plot in the
`vit_finetuning` output folder. The folder contains 32 PNG plots and no
accompanying CSV files, prediction arrays, evaluation logs, dataset manifest,
source code, checkpoint metadata, sample counts, or confidence intervals.

The findings are therefore divided into:

1. **Direct inconsistencies:** claims or numbers that conflict across plots or
   are mathematically suspicious.
2. **Evaluation deficiencies:** information needed to establish whether the
   reported values are clinically meaningful is missing.
3. **Interpretation risks:** plots may be visually plausible but do not support
   the clinical conclusions suggested by their titles or annotations.

The plots appear to describe a binary Normal-versus-Pathology task. The report
uses “Pathology” as the positive clinical class when discussing sensitivity and
false negatives.

---

## Executive assessment

The plots suggest that fine-tuning improves in-distribution representation
separation and average performance. They do **not**, by themselves, establish
reliable clinical classification accuracy.

The highest-risk findings are:

1. The normalized confusion matrix implies a balanced accuracy of 0.915, while
   the metric trajectory reports approximately 0.93.
2. The binary one-vs-rest ROC plot reports unequal class AUCs of 0.940 and
   0.910, which is suspicious when complementary binary probabilities are used
   on the same cases.
3. The two precision-recall curves visually overlap and report the same AP of
   0.859, despite the unequal ROC AUCs.
4. No deployed classification threshold is reported.
5. The confusion matrix implies that approximately 9% of pathology cases are
   missed at the displayed operating point.
6. AUROC falls from approximately 0.94 internally to approximately 0.85 at a
   low-SNR site, indicating potentially important clinical degradation under
   acquisition shift.
7. The calibration curve indicates overconfident predictions, and its reported
   ECE does not agree with the ECE trajectory plot.
8. Subgroup and external-site plots omit sample sizes, confidence intervals,
   threshold-specific error rates, and calibration.
9. Several attention and saliency plots are not overlaid on actual radiographs,
   so they do not demonstrate clinically valid anatomical reasoning.
10. No plot provides repeated-seed variability or confidence intervals, making
    reproducibility and statistical significance impossible to assess.

Until these issues are resolved, the results should be treated as exploratory
model-development evidence rather than validated clinical performance.

---

# 1. Critical cross-plot inconsistencies

## 1.1 Confusion matrix and balanced accuracy do not agree

### Evidence

`vit_normalized_confusion_matrix.png` reports the following row-normalized
values:

| True class | Predicted Normal | Predicted Pathology | Class recall |
|---|---:|---:|---:|
| Normal | 92% | 8% | 92% |
| Pathology | 9% | 91% | 91% |

For a binary classifier, the balanced accuracy implied by these recalls is:

```text
(0.92 + 0.91) / 2 = 0.915
```

`vit_f1_and_balanced_acc.png` shows final balanced accuracy of approximately
0.93.

### Why this is a flaw

If both figures describe the same checkpoint, split, labels, and threshold,
they cannot both be exact. The discrepancy could be caused by:

- The confusion matrix being produced from a different split.
- The metric curve and confusion matrix using different checkpoints.
- Different classification thresholds.
- An inconsistent Normal/Pathology label mapping.
- Incorrect balanced-accuracy computation.
- One plot using validation predictions and the other using test predictions.
- Rounding that is too aggressive, although the difference is larger than a
  simple one-decimal-percent rounding effect.

### Clinical effect

The confusion matrix implies approximately:

- Pathology sensitivity: **91%**.
- Pathology false-negative rate: **9%**.
- Normal specificity: **92%**.
- Normal false-positive rate: **8%**.

At 1,000 pathology-positive patients, a 9% false-negative rate would mean
approximately 90 missed cases at this operating point. The approximately 0.93
balanced-accuracy claim could make performance look safer than the confusion
matrix actually indicates.

### Required correction

Regenerate the confusion matrix and aggregate metrics from exactly the same
prediction table, checkpoint, split, label mapping, and threshold. Report raw
counts in addition to row-normalized percentages.

---

## 1.2 Binary ROC AUC values are mathematically suspicious

### Evidence

`vit_multiclass_roc_curves.png` reports:

- Class 0 AUC: **0.940**.
- Class 1 AUC: **0.910**.

The task shown elsewhere is binary, with Normal and Pathology classes.

### Why this is a flaw

For a standard binary softmax classifier, the two class probabilities are
complements:

```text
p(class 0) = 1 - p(class 1)
```

When the same examples and complementary scores are evaluated, the two
one-vs-rest ROC AUCs should be equal apart from numerical effects. Unequal AUCs
of 0.94 and 0.91 suggest that one of the following may have happened:

- Different score vectors were used for the two curves.
- Different subsets were used.
- Labels were mapped inconsistently.
- The binary outputs were passed through an unsuitable multiclass evaluation
  routine.
- The plot was generated from illustrative or manually specified values.

### Clinical effect

The reported discrimination cannot be trusted until the calculation is
audited. Even a valid AUROC does not establish adequate pathology sensitivity
at the deployed threshold, but an internally inconsistent AUROC makes the
underlying prediction evaluation more difficult to interpret.

### Required correction

For a binary task, calculate one ROC curve using the pathology probability and
report its AUC. If both one-vs-rest curves are retained, verify that they use
the same examples, labels, and complementary probability vectors.

---

## 1.3 Precision-recall curves appear duplicated

### Evidence

`vit_precision_recall_curves.png` shows two curves that visually overlap almost
exactly and reports:

- Class 0 AP: **0.859**.
- Class 1 AP: **0.859**.

### Why this is a flaw

Average precision can differ between binary classes because class prevalence
and the ranking of positive examples differ. Exact equality and near-identical
curves are suspicious, particularly alongside unequal ROC AUC values.

Possible causes include:

- The same precision-recall arrays being plotted twice.
- The same score vector being used for both classes.
- Class labels not being independently evaluated.
- The figure being generated from illustrative values rather than predictions.

### Clinical effect

Average precision is important when pathology prevalence is low or when the
positive class is clinically important. A duplicated curve could conceal poor
positive predictive performance or incorrect handling of the pathology class.

### Required correction

Plot the pathology precision-recall curve from pathology probabilities and
report prevalence, AP confidence intervals, precision at clinically relevant
recall values, and recall at clinically relevant precision values.

---

## 1.4 Calibration values conflict across plots

### Evidence

`vit_prediction_calibration_curve.png` reports **ECE = 0.038** in its legend.
`vit_ece_brier_trajectory.png` appears to end at approximately **ECE = 0.031**.

### Why this is a flaw

The values may be computed with different splits, bin counts, checkpoints,
class definitions, or calibration procedures, but the plots do not document
those differences. If they represent the same final model and evaluation set,
the values are inconsistent.

### Clinical effect

Calibration is used to interpret confidence. An apparently low ECE may give
users unwarranted trust in the model if it was calculated on a different split
or with a favorable binning choice. Calibration may also degrade under the
external-site shifts shown elsewhere.

### Required correction

Document the calibration dataset, number of bins, binning strategy, ECE
definition, Brier-score definition, calibration method, and checkpoint. Use
the same locked test predictions for the final calibration report.

---

# 2. Missing clinical operating-point information

## 2.1 No classification threshold is reported

The ROC and precision-recall plots show threshold sweeps, but no selected
operating threshold is identified. The confusion matrix necessarily represents
some threshold, yet that value is absent.

The folder does not specify:

- The threshold used for the confusion matrix.
- Whether it was selected on training, validation, or test data.
- Whether it maximized accuracy, F1, Youden’s index, or another criterion.
- Whether the threshold was fixed before external-site testing.
- Whether the threshold was recalibrated per site.

### Clinical effect

AUC measures ranking across thresholds, not clinical accuracy at a specific
decision point. A model can have strong AUC but unacceptable sensitivity or
specificity at the threshold used in practice.

The report must include, at minimum:

- Pathology sensitivity/recall.
- Specificity.
- False-negative rate.
- False-positive rate.
- PPV.
- NPV.
- Accuracy and balanced accuracy.
- The threshold producing those results.
- Confidence intervals for each metric.

---

## 2.2 Macro F1 and balanced accuracy conceal class-specific performance

`vit_f1_and_balanced_acc.png` shows final macro F1 and balanced accuracy near
0.93, but it does not show:

- F1 for Normal.
- F1 for Pathology.
- Pathology precision.
- Pathology recall.
- Raw class counts.
- Confidence intervals.

Macro averaging can make a model appear balanced while hiding clinically
important differences in the positive class. The confusion matrix suggests that
Pathology recall is approximately 91%, but the metric plot does not expose this
directly.

### Clinical effect

An aggregate score should not be used as a substitute for disease sensitivity.
For a screening use case, a missed pathology case may be more consequential
than a false positive Normal-to-Pathology referral.

---

## 2.3 Accuracy, AUROC, and AP are not interchangeable

The folder uses AUROC, average precision, macro F1, balanced accuracy, loss,
and calibration metrics. These answer different questions:

- **Loss:** probability-fitting objective, not a clinical error rate.
- **AUROC:** ranking discrimination across thresholds.
- **Average precision:** positive-ranking performance affected by prevalence.
- **F1:** threshold-dependent precision-recall balance.
- **Balanced accuracy:** average of class recalls.
- **ECE/Brier:** probability calibration.

The plots do not clearly distinguish these concepts. A high AUROC or a lower
validation loss does not prove that the chosen clinical threshold has an
acceptable pathology false-negative rate.

---

# 3. Calibration and confidence flaws

## 3.1 The calibration curve indicates overconfidence

`vit_prediction_calibration_curve.png` places the empirical-accuracy curve
below the perfect-calibration diagonal over the displayed confidence range.
For example, at confidence near 0.90, empirical accuracy appears closer to
0.88–0.89.

This means the model is generally overconfident: its stated probability is
higher than the observed fraction of correct predictions.

Missing information includes:

- Number of samples per calibration bin.
- Confidence intervals for each bin.
- Calibration results separately for Normal and Pathology.
- Calibration results at each external site.
- Whether temperature scaling or another calibration method was applied.
- Whether the calibration set was independent of model and threshold
  selection.

### Clinical effect

Overconfident errors are dangerous. A user may interpret a high-confidence
Normal result as reassurance even when the case is actually Pathology. This is
especially concerning if high-confidence errors are concentrated among false
negatives or shifted-site images.

---

## 3.2 Confidence histogram contains nontrivial high-confidence errors

`vit_confidence_distribution_hist.png` shows most correct predictions at high
confidence, but errors remain at moderate and higher confidence values.

Problems with the plot:

- No total number of correct predictions or errors is reported.
- No percentages are shown, only histogram counts.
- No class-specific confidence distributions are shown.
- No site-specific or subgroup-specific distributions are shown.
- The plot does not identify high-confidence pathology false negatives.

### Clinical effect

If high-confidence errors are not routed for review, an uncertainty-based safety
strategy may fail. A clinically useful confidence analysis must identify the
rate of high-confidence errors and specifically quantify high-confidence false
negatives.

---

## 3.3 The uncertainty-error relationship is not quantified

`vit_uncertainty_vs_error_scatter.png` shows a positive visual relationship
between predictive entropy and plotted residual error.

Problems:

- No Pearson or Spearman correlation is reported.
- No R-squared value or confidence interval is reported.
- “Cross-entropy residual error” is not defined.
- The fitted line extends below zero even though the plotted error quantity
  appears nonnegative.
- No abstention or referral threshold is proposed.
- The plot does not identify pathology false negatives.

### Clinical effect

The model may be more uncertain on difficult cases, but this has not been shown
to provide a safe mechanism for flagging errors. In particular, low-entropy
false negatives are the clinically dangerous cases and should be explicitly
reported.

---

# 4. Generalization and distribution-shift flaws

## 4.1 External-site performance drops substantially

`vit_cross_site_domain_shift.png` reports approximately:

| Site | AUROC |
|---|---:|
| Site A, internal | 0.94 |
| Site B, external | 0.89 |
| Site C, scanner shift | 0.88 |
| Site D, low SNR | 0.85 |

`vit_ood_performance_drop.png` reports fine-tuned AUROC approximately:

| Condition | Fine-tuned AUROC |
|---|---:|
| In-distribution | 0.94 |
| Gaussian noise | 0.88 |
| Contrast shift | 0.85 |
| External hospital | 0.87 |

### Why this is a flaw

The approximate 0.09 drop from internal data to the low-SNR site is clinically
meaningful. However, the plots do not provide:

- Patient and image counts by site.
- Pathology prevalence by site.
- Confidence intervals.
- Threshold-specific sensitivity and specificity.
- Site-specific calibration.
- Scanner, protocol, demographic, and acquisition details.
- Confirmation that the external hospital data are truly independent.

### Clinical effect

A threshold selected at Site A may not maintain the same sensitivity or
specificity at Sites B–D. The low-SNR site is particularly concerning because
the model may miss pathology in degraded images even when ranking performance
remains above chance.

### Required correction

For each site, report AUROC, AUPRC, sensitivity, specificity, FNR, FPR, PPV,
NPV, calibration, sample counts, prevalence, and confidence intervals. Keep the
internal threshold fixed when evaluating external sites, then separately report
the effect of any recalibration.

---

## 4.2 OOD comparisons do not establish robustness

The OOD plot compares pre-fine-tuned and fine-tuned AUROC under synthetic noise,
contrast shift, and an external hospital condition.

Problems:

- Synthetic corruption severity is not defined.
- It is unclear whether the same images are used before and after corruption.
- AUROC does not show clinical operating-point behavior.
- No uncertainty is shown.
- No baseline beyond the pre-fine-tuned model is shown.
- “External hospital” may not be comparable to the synthetic conditions.

### Clinical effect

The plot supports a relative improvement claim, not a claim of clinical safety
under shift. Fine-tuning can improve AUROC while still increasing false
negatives at the deployment threshold.

---


None of the plots documents whether the split was:

- Patient-level.
- Image-level.
- Site-level.
- Study-level.
- Time-based.

There is also no duplicate or near-duplicate check. Chest-image datasets can
contain multiple images from the same patient or related studies.

### Clinical effect

Patient or study leakage can inflate validation and internal test metrics. A
model that performs well on related images may perform substantially worse on
new patients or institutions.

---


## 5.1 The subgroup chart reports only AUROC

`vit_subgroup_disparity_chart.png` reports:

| Subgroup | AUROC |
|---|---:|
| Age >= 50 | 0.930 |
| Age < 50 | 0.920 |
| Male cohort | 0.950 |
| Female cohort | 0.930 |
| Overall | 0.940 |

### Why this is insufficient

The plot omits:

- Subgroup sample sizes.
- Pathology prevalence.
- Confidence intervals.
- Sensitivity.
- Specificity.
- False-negative rate.
- False-positive rate.
- PPV and NPV.
- Calibration.
- Intersectional groups, such as older females.
- Subgroup-by-site performance.

An AUROC difference of 0.01–0.02 may be sampling noise or may conceal a large
threshold-specific difference.

### Clinical effect

Equal AUROC does not imply equal clinical safety. Two subgroups can have the
same AUROC while having different false-negative rates at the fixed deployment
threshold.

### Required correction

Report class-specific operating metrics and uncertainty by subgroup, with
intersectional and site-stratified analyses.

---



`vit_saliency_iou_roi.png` reports:

- Pre-fine-tuning mean IoU: **0.28**.
- Post-fine-tuning mean IoU: **0.62**.

The increase is encouraging, but a mean IoU of 0.62 still indicates meaningful
non-overlap between saliency and the pathology ROI.

Missing information includes:

- Number of images.
- Annotation source and quality.
- Confidence intervals.
- Class-specific localization results.
- Correct versus incorrect prediction breakdown.
- Saliency method and thresholding rule.
- Whether ROIs were available during training.
- Whether the masks identify disease, anatomy, or a coarse region.

### Clinical effect

Improved overlap does not prove that the model uses causal pathology features.
It also does not prove improved classification sensitivity. A model can achieve
better average saliency IoU while still missing small, diffuse, or peripheral
lesions.

---

## 6.2 Background-occlusion results are under-specified

`vit_background_occlusion_sensitivity.png` shows approximately:

- Fine-tuned ViT: 0.93 at 0% masking, declining to 0.88 at 80% masking.
- Frozen pretrained model: 0.92 at 0% masking, declining to 0.40 at 80%
  masking.

Positive interpretation: fine-tuning appears to reduce background dependence.

Problems:

- The masking algorithm and mask location are not described.
- No foreground-occlusion control is shown.
- No confidence intervals are shown.
- Results are not separated by class.
- No pathology false-negative analysis is shown.
- The masked region may not be clinically irrelevant background.
- The 0%-masking performance is not reconciled with the confusion matrix.

### Clinical effect

The model appears more robust to background removal, but the residual decline
indicates that contextual or image-quality features may still affect the
decision. It is not established that the model relies on the lesion rather
than on a fixed spatial pattern, image framing, or acquisition artifact.

---

## 6.3 Attention shift map is not patient-specific evidence

`vit_attention_shift_map.png` shows a diffuse pre-fine-tuning pattern, a
rectangular post-fine-tuning pattern, and their difference.

Problems:

- No actual radiograph is shown.
- No patient or sample identifier is provided.
- The layer and attention head are not specified.
- Normalization and aggregation methods are not specified.
- The rectangular post-fine-tuning focus could represent a fixed spatial prior.
- The map is not linked to a correct or incorrect prediction.

### Clinical effect

The figure cannot establish that the model attends to pathology. A fixed
central or rectangular prior may perform well on a curated dataset while
failing when lesion location, positioning, or acquisition geometry changes.

---

## 6.4 CLS attention heatmaps are not anatomically interpretable

`vit_cls_attention_heatmaps.png` displays four highly similar central heatmaps
on black backgrounds.

Problems:

- Original radiographs are absent.
- No color scale is provided.
- No image dimensions or patch grid are provided.
- No layer or head is identified.
- Prediction labels and correctness are absent.
- The nearly identical central square patterns are suspicious.
- There is no comparison with pathology masks.

### Clinical effect

These images do not demonstrate clinically valid localization. They could
reflect a fixed spatial attention pattern, averaging artifact, or a plotting
template rather than patient-specific evidence.

---

## 6.5 Lower attention entropy is not automatically beneficial

`vit_attention_entropy_per_layer.png` shows post-fine-tuning entropy declining
much more sharply than pre-fine-tuning entropy, reaching approximately 1.10
versus approximately 1.84 at the deepest displayed layer.

Possible interpretations include more focused attention, but also excessive
concentration.

Missing information:

- Head-level variability.
- Entropy confidence intervals.
- Correct versus incorrect prediction comparisons.
- Relationship to lesion size and location.
- Attention behavior under external-site shift.

### Clinical effect

Over-concentrated attention may improve average performance while missing
multifocal, diffuse, peripheral, or subtle disease. Lower entropy should not be
treated as evidence of better clinical reasoning without outcome validation.

---

## 6.6 Fourier energy is not equivalent to lesion detection

`vit_attention_frequency_energy.png` shows high-frequency relative energy
increasing from approximately 0.32 to 0.45 while low-frequency energy decreases.

Problems:

- Fourier frequency is not directly equivalent to anatomical scale.
- The method used to calculate spatial attention energy is not specified.
- No frequency perturbation or lesion-specific validation is shown.
- No relation to noise robustness or false negatives is reported.

### Clinical effect

Increased high-frequency reliance might improve fine-detail detection, but it
may also increase sensitivity to noise, compression, scanner characteristics,
and preprocessing artifacts. This is particularly relevant because the model
shows lower AUROC on low-SNR and contrast-shifted data.

---



`vit_inter_intra_class_distance.png` reports approximately:

- Within-class diagonal values: 0.28 and 0.31.
- Between-class off-diagonal values: 0.86.

Problems:

- No sample counts or dispersion are reported.
- The embedding layer is not identified.
- The distance metric and normalization are not documented.
- No relationship to threshold-specific errors is shown.
- Averages can hide difficult clinical subpopulations.

### Clinical effect

Separated average embeddings can coexist with difficult cases and
high-confidence false negatives. Representation distance is not a substitute
for patient-level sensitivity and specificity.

---

## 7.2 UMAP can create visually persuasive but non-clinical clusters

`vit_pre_post_umap.png` shows substantial class overlap before fine-tuning and
strong separation after fine-tuning.

Problems:

- UMAP is nonlinear and can create or exaggerate apparent clusters.
- Pre- and post-fine-tuning coordinates have different scales and cannot be
  compared directly by visual geometry.
- The random seed, number of neighbors, distance metric, and preprocessing are
  not documented.
- No quantitative cluster statistic is reported.
- No independent test or external-site coloring is shown.

### Clinical effect

Clean-looking embeddings do not prove generalization to new patients or
hospitals. A projection can look separated while the actual clinical decision
boundary remains poorly calibrated or threshold-sensitive.

---

## 7.3 t-SNE comparison has the same limitations

`vit_pre_post_tsne.png` has the same risks as the UMAP plot and adds the fact
that t-SNE preserves local neighborhoods rather than global distances.

Problems:

- Separate t-SNE projections cannot be compared by absolute position or
  distance.
- Hyperparameters and random seed are absent.
- No quantitative validation accompanies the visual separation.
- No error overlays identify where false negatives or false positives occur.

### Clinical effect

The plot may overstate how cleanly the classes separate in unseen clinical
data.

---

## 7.4 Effective rank is not a clinical quality metric

`vit_effective_rank_curve.png` rises from approximately 22.7 to 32.3.

Problems:

- The matrix or representation being analyzed is not identified.
- The maximum possible rank is not shown.
- No uncertainty or replicate runs are shown.
- Higher effective rank is not necessarily better for classification.

### Clinical effect

The increased effective rank does not establish improved sensitivity,
calibration, or robustness. It may reflect broader representation use without
improving clinically relevant features.

---

## 7.5 Singular-value spectrum interpretations are unsupported

`vit_singular_value_spectrum.png` labels the pre-fine-tuning curve “Steep
Decay / Collapse” and the post-fine-tuning curve “Preserved Capacity.”

Problems:

- The layer, matrix, and data source are not identified.
- The curves are smooth and provide no replicate variability.
- “Collapse” and “preserved capacity” are interpretations, not directly
  measured clinical outcomes.
- No link to performance under shift is shown.

### Clinical effect

Preserved representation capacity does not guarantee valid features or robust
clinical predictions.

---

## 7.6 Cosine representation drift is only an average

`vit_cosine_similarity_drift_hist.png` reports mean drift of approximately
0.0715.

Problems:

- The layer and representation are not specified.
- Drift is not separated by class, site, subgroup, or prediction correctness.
- The average can hide large changes in a clinically important subset.
- Kernel-density smoothing may hide multimodality or tails.

### Clinical effect

Small average drift does not rule out harmful changes for pathology cases,
minority groups, or external-site images.

---

## 7.7 CKA heatmap is ambiguously defined

`vit_cka_layer_heatmap.png` displays a 12-by-12 layer similarity heatmap.

Problems:

- It is unclear whether CKA compares layers within one model or pre- and
  post-fine-tuning models.
- If it is a self-comparison, the diagonal being 1 is expected and not
  informative.
- The two representations being compared are not identified.
- No aggregate preservation statistic is reported.

### Clinical effect

The plot cannot establish whether useful pretrained features were preserved or
whether fine-tuning introduced harmful changes.

---

## 7.8 Weight displacement is global and hides localized changes

`vit_weight_displacement_trajectory.png` shows relative displacement rising
from approximately 0.02 to 0.075.

Problems:

- No layer-specific displacement is shown.
- No relationship to metric changes or error subgroups is shown.
- A small global displacement can conceal large changes in a small but decisive
  layer.

### Clinical effect

The global value does not rule out changes to the decision boundary that
increase pathology false negatives.

---


`vit_layerwise_grad_norms.png` shows only:

- Probe head.
- ViT Block 11.
- ViT Block 10.
- Post-layer normalization.

Problems:

- Most transformer blocks are omitted.
- No batch-level variation or error-conditioned gradients are shown.
- The smooth monotonic decay may be an aggregate or illustrative trajectory.
- It is not possible to assess whether early and middle layers were updated
  appropriately.

### Clinical effect

The figure cannot demonstrate that all clinically relevant feature layers were
optimized or that gradient behavior improved clinical sensitivity.

---


`vit_input_grad_norm_sensitivity.png` shows lower post-fine-tuning loss
sensitivity than pre-fine-tuning sensitivity over perturbations from 0 to 0.08.

Problems:

- The perturbation norm and image-space units are unspecified.
- The perturbations are not shown to be clinically realistic.
- No attack method or perturbation direction is provided.
- No confidence intervals are shown.
- No error-conditioned analysis is provided.

### Clinical effect

Lower gradient sensitivity may indicate a smoother model, but it does not
guarantee better performance on scanner shifts, noise, positioning changes, or
subtle pathology.

---


`vit_lr_schedule.png` shows a steep learning-rate decay over ten epochs. The
final portion approaches or falls below the visible lower plotting range.

Problems:

- The final learning-rate value is not clearly readable.
- Warm-up, optimizer, scheduler type, and layer-specific learning rates are
  not documented.
- The schedule is not linked to validation sensitivity, calibration, or the
  chosen checkpoint.

### Clinical effect

If the learning rate becomes too small before clinically relevant errors are
resolved, optimization may stop improving the decision boundary even while
training loss continues to decrease.

---



`vit_train_val_loss_curve.png` shows:

- Training loss falling from approximately 0.77 to 0.19.
- Validation loss falling from approximately 0.85 to 0.264.

This indicates improving optimization and no obvious late validation-loss
explosion. However:

- No test loss is shown.
- No uncertainty is shown.
- No patient-level split information is shown.
- Loss does not establish sensitivity or specificity.
- The selected checkpoint is not justified by a clinically relevant metric.

### Clinical effect

The model can continue reducing cross-entropy while making little or no
improvement in pathology recall. Checkpoint selection should be based on a
pre-specified clinical metric, not loss alone.

---


`vit_train_val_loss_ratio.png` decreases from approximately 0.91 to about 0.68
and rises to about 0.71 at epoch 10.

Problems:

- The ratio is scale-dependent.
- It is less interpretable than absolute differences or clinically relevant
  metrics.
- It does not show false negatives or calibration.
- A lower ratio does not imply safer clinical predictions.

### Clinical effect

The plot can create an impression of stable generalization without showing
whether the model makes clinically acceptable decisions.

---


`vit_generalization_gap.png` shows a positive loss gap of approximately
0.08–0.11, peaking around epochs 7–8.

Problems:

- It measures validation loss minus training loss, not clinical accuracy.
- It does not show test performance.
- It does not show sensitivity or false-negative rate.
- A positive gap is normal and does not alone establish harmful overfitting.

### Clinical effect

The plot cannot establish whether epoch 10 is the safest checkpoint for
pathology detection.

---


`vit_sample_efficiency_learning_curves.png` shows approximate values:

| Training data available | AUROC | Macro F1 |
|---:|---:|---:|
| 10% | 0.76 | 0.72 |
| 25% | 0.84 | 0.80 |
| 50% | 0.89 | 0.86 |
| 75% | 0.92 | 0.89 |
| 100% | 0.94 | 0.91 |

Problems:

- No sample counts are given.
- No repeated subsampling or seed variation is shown.
- It is unclear whether the same locked test set was used for all points.
- Threshold selection at each training size is not described.
- Macro F1 remains below AUROC, showing that ranking quality and classification
  quality differ.

### Clinical effect

The model is considerably weaker at lower data budgets, and the full-data
result still does not prove acceptable pathology sensitivity. The curve should
not be used to claim sample-efficient clinical deployment without uncertainty
and independent test evaluation.

---


## 9.1 “Multiclass” terminology is inconsistent with the displayed task

The ROC plot is titled “One-Vs-Rest ROC Analysis” and the confusion matrix has
two classes: Normal and Pathology. The task therefore appears binary, yet the
ROC file is named and presented as multiclass.

This may be harmless terminology, but it raises the possibility that binary
outputs were processed through a multiclass evaluation path incorrectly.

### Clinical effect

Incorrect task handling can affect probability extraction, ROC/AUPRC
calculation, label mapping, and the reported positive-class metrics.

### Required correction

State clearly:

- Number of classes.
- Class-to-index mapping.
- Positive clinical class.
- Probability column used for pathology.
- Exact metric implementation.

---


The plots use the broad label “Pathology,” but do not identify:

- Which diseases are included.
- Whether multiple findings are grouped together.
- Whether the labels are radiologist labels, reports, or metadata.
- Whether “Normal” excludes other non-target abnormalities.
- Whether the task is screening, diagnosis, or abnormality detection.

### Clinical effect

Combining heterogeneous abnormalities can make the task easier while obscuring
which clinically important conditions are missed. The model may detect broad
image differences without reliably detecting the target disease.

---



None of the plots provides error bars, bootstrap intervals, cross-validation
variation, or repeated-seed variation.

This affects the interpretation of all reported values, including:

- AUROC.
- Average precision.
- Balanced accuracy.
- F1.
- ECE and Brier score.
- Site comparisons.
- Subgroup comparisons.
- Saliency IoU.
- Attention and representation trajectories.

### Clinical effect

An AUROC of 0.94 may be substantially less certain if the test set is small or
contains few pathology cases. Without intervals, the reliability of the
reported performance and the significance of site/subgroup differences cannot
be judged.

---


Raw counts are absent from the confusion matrix and all aggregate plots.
Prevalence is not shown for the internal, external, or subgroup evaluations.

### Clinical effect

PPV and NPV depend strongly on prevalence. A model can have acceptable
internal PPV but poor real-world PPV at a lower disease prevalence. Conversely,
NPV may be overstated in a highly imbalanced evaluation set.

---


The folder does not document how images were divided among training,
validation, and test sets. There is no evidence of:

- Patient-level splitting.
- Study-level splitting.
- Duplicate removal.
- Near-duplicate detection.
- Temporal holdout.
- Site holdout.

### Clinical effect

Leakage between related studies or images from the same patient can inflate
internal performance and make external degradation appear unexpectedly large.

---


The loss plot identifies “Best Val Loss (0.2642 @ Ep 10),” but the metric plots
do not establish whether epoch 10 was selected using:

- Validation loss.
- Balanced accuracy.
- Pathology sensitivity.
- F1.
- AUROC.
- Calibration.

### Clinical effect

The checkpoint with the best loss may not be the checkpoint with the best
clinical sensitivity or calibration. Selection on a metric that is not aligned
with the intended use can increase missed pathology cases.

---



The confusion matrix suggests approximately 91% pathology recall, or 9%
pathology false negatives, at an unspecified threshold. This is the most direct
clinical concern in the folder.

## 11.2 Performance may be site-dependent

The external-site plots suggest AUROC declines from approximately 0.94 to
approximately 0.85 under low-SNR conditions. The deployment threshold may
therefore produce different sensitivity and specificity at different sites.

## 11.3 High confidence may not mean safe prediction

The calibration curve indicates overconfidence and the confidence histogram
contains errors at moderate and high confidence. The model may fail to flag

## 11.4 Aggregate performance may mask class and subgroup errors

Macro F1, balanced accuracy, and AUROC do not expose all clinically relevant
error patterns. Subgroup AUROC alone does not establish equitable sensitivity,
false-negative rates, or calibration.

## 11.5 Explainability claims are not clinically validated

Saliency IoU and attention plots suggest more concentrated or ROI-aligned
representations, but the evidence does not demonstrate that the model uses
actual disease anatomy on patient images. Fixed spatial or acquisition
shortcuts remain possible.

---


## Priority 0: Reconcile all metrics

Regenerate all results from a single documented prediction table containing:

- Patient or study identifier.
- True label.
- Normal probability.
- Pathology probability.
- Predicted label.
- Decision threshold.
- Split name.
- Site.
- Demographic subgroup.
- Checkpoint identifier.

Use the same prediction table for the confusion matrix, F1, balanced accuracy,
ROC, PR, calibration, and confidence plots.

## Priority 1: Report threshold-specific clinical metrics

For the selected threshold, report:

- Pathology sensitivity.
- Specificity.
- False-negative rate.
- False-positive rate.
- PPV.
- NPV.
- Accuracy.
- Balanced accuracy.
- Class-specific F1.
- Raw confusion-matrix counts.
- 95% confidence intervals.

Also report sensitivity at clinically relevant specificity levels and identify
the threshold used for every confusion matrix.

## Priority 2: Validate external generalization

For every external site and shift condition:

- Keep the internal threshold fixed for the primary analysis.
- Report prevalence and sample counts.
- Report threshold-specific error rates.
- Report calibration and confidence behavior.
- Provide bootstrap or exact confidence intervals.
- Document scanner, protocol, demographic, and image-quality differences.

## Priority 3: Complete the subgroup audit

For each subgroup and intersectional subgroup, report sensitivity, specificity,
FNR, FPR, PPV, NPV, AUROC, AUPRC, calibration, sample size, and confidence
intervals. Include subgroup-by-site analyses.

## Priority 4: Validate explainability on actual images

For attention and saliency analyses:

- Show the original radiograph.
- Show the pathology annotation or ROI.
- Show the explanation overlay.
- Identify layer, head, and normalization.
- Include correct and incorrect predictions.
- Report localization uncertainty.
- Test explanation stability.
- Test whether explanations remain valid under site and scanner shift.

## Priority 5: Establish reproducibility and leakage controls

Document:

- Patient-level splitting.
- Duplicate and near-duplicate checks.
- External-site independence.
- Random seeds.
- Number of repeated runs.
- Checkpoint-selection rule.
- Threshold-selection rule.
- Exact software and metric versions.

Report means, standard deviations, and confidence intervals across repeated
runs.

---


The plots show promising exploratory changes after BiomedCLIP ViT fine-tuning,
including lower validation loss, improved apparent class separation, greater
saliency overlap, and improved robustness relative to the frozen baseline.

However, the evidence is not internally consistent enough to support a strong
clinical accuracy claim. The most important unresolved issues are the
conflicting balanced-accuracy and confusion-matrix values, suspicious ROC and
precision-recall calculations, missing threshold information, apparent
overconfidence, substantial external-site degradation, incomplete subgroup
auditing, absent uncertainty estimates, and insufficiently grounded attention
visualizations.

The model should not be characterized solely as a “93% accurate” clinical
classifier. The available evidence indicates a model with approximately 91%
pathology recall at one undocumented operating point, potentially higher error
rates under distribution shift, and unresolved calibration and reproducibility
problems.
