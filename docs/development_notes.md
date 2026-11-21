# Development Notes

## 1. Purpose of This Review

This document provides a retrospective technical review of the original
**Latent Space Explorer** project, developed in 2023.

The purpose is to reflect on the project's design decisions,
implementation approach, experimental methodology and limitations from
my current understanding. This is not a record of changes made to the
original implementation. The original source files are preserved in
`original_work/` to maintain the integrity of the academic work.

The observations below identify areas that could be investigated or
approached differently if the project were developed again.

## 2. Implementation and Software Design

The original project brought together feature loading, dimensionality
reduction, clustering, visualization and evaluation in a small
Python-based experimental workflow. Looking back, several design choices
could be made more flexible and maintainable. These observations do not mean that the original project failed to meet
its academic purpose. They reflect the difference between a small
experimental implementation and a more reusable software tool.

| Area | Retrospective observation | Possible future improvement |
|---|---|---|
| Data loading | Data loading is closely coupled to parts of the clustering implementation through shared global variable | Seperate data preparation from algorithm implementatios and pass the required data explicitly. |
| Code organization | some responsibilities and setup steps are repeated between the library and runner scripts. | Seperate reusable functions for clarity and experiments. |
| Execution | Some operations are triggered when a module is imported. | Keep imports side-effect free and place script executionbehind an explicit entry point. |
| Configuration | File paths and some settings are hardcoded. | Make paths and experimental parameters configurable. |
| Outputs | Outputs do not always follow a consistent structure. | Define a common result format. |

## 3. Algorithm Selection and Parameter Considerations

The project explored four dimensionality-reduction techniques---PCA,
LDA, UMAP and t-SNE---and two clustering algorithms---K-Means and
Spectral Clustering.

The original experiment focused on applying selected algorithms and
comparing their resulting visualizations. A more systematic
investigation would also examine how parameter choices affect the
observed structures. It would be useful to document the parameter
settings for every run and, where appropriate, compare multiple settings
rather than relying on a single configuration.

## 4. Evaluation Methodology

The project used visual inspection, silhouette scores and
processing-time measurements to compare experimental results. These
provide useful perspectives, but each has limitations.

| Evaluation aspect    | Retrospective observation | 
|---|---|
| Silhouette score | the score summarizes cluster compactness and separation under a chosen distance measure. It does not independantly establish whether clusters are semantically meaningful. |
| Visual inspection | Two-dimensional plots help communicate patterns, but dimensionality reduction can distort relationships in the original feature space. |
| Comparison between methods | Different algorithms have different assumptions. Their outputs should not automatically be treated as directly comparable without explaining those differences. |
| Evaluation consistency | Scores should be calculated using the intended representation and corresponding cluster labels for each experiment.|
| Parameter sensitivity | Investigating how parameter changes affects scores and visualizations would provide a more complete evaluation  |

## 5. Performance and Computational Considerations

The project recorded processing times for dimensionality reduction and
clustering combinations. These measurements provide an initial
indication of the computational cost of the selected methods.

The results also illustrate that computational cost can vary
considerably between algorithm combinations. In particular, the recorded
t-SNE and Spectral Clustering combination took substantially longer than
several of the other combinations in the original experiment.

Looking back, benchmarking could be made more informative by:

-   Measuring data loading, dimensionality reduction, clustering and
    evaluation separately.
-   Repeating experiments and reporting an appropriate summary, rather
    than relying on a single run.
-   Recording the hardware, software versions, dataset size and relevant
    algorithm parameters.
-   Ensuring that timing boundaries are consistent across all
    combinations.

The original timing results should be interpreted in the context of the
specific dataset, environment and implementation used. They should not
be treated as general performance rankings for the algorithms.

## 6. Reproducibility and Experimental Reliability

Reproducing an experiment requires more than retaining its source code.
The data structure, dependencies, configuration and execution conditions
also matter.

From a retrospective perspective, the following areas could be
documented more explicitly:

-   **Input data:** The original workflow expects feature files in a
    particular directory structure, with class names represented by
    subfolder names.
-   **File paths:** Local absolute paths make the experiment dependent
    on the original development environment.
-   **Randomness:** Some algorithms can produce different results
    depending on random initialization. Random-state settings should be
    recorded where supported.
-   **Dependencies:** Library versions should be documented because APIs
    and algorithm behavior can change between releases.
-   **Parameters:** The values used for each algorithm should be
    recorded alongside the generated results.

These considerations would make it easier for another person to
understand the experimental setup and assess whether results can be
reproduced.

## 7. What I Would Approach Differently Today

If I were to approach this project again in 2026, I would begin by defining a
clearer experimental pipeline with separate responsibilities for data
preparation, dimensionality reduction, clustering, visualization and
evaluation.

I would also make the experiment configuration explicit so that each run
could be traced to its input data, algorithm parameters and software
environment. For evaluation, I would distinguish between scores
calculated on original features and those calculated on reduced
representations, and explain what each measurement is intended to
assess.

From a software-design perspective, I would aim for reusable components
with consistent inputs and outputs. This would make it easier to extend
the experiment to additional algorithms or datasets without duplicating
workflow logic.

These are reflections on how I would design a future version; they
should not be read as features implemented in the original project.

## 8. Key Takeaways

Looking back at this first-semester project, the main learning outcomes
were:

-   Gained practical experience applying dimensionality-reduction
    methods to high-dimensional feature representations.
-   Explored how clustering algorithms behave when applied to original
    and reduced feature spaces.
-   Learned that supervised and unsupervised dimensionality-reduction
    methods require different interpretations.
-   Developed experience with Python-based machine-learning libraries
    and visualization.
-   Recognized the importance of evaluation consistency and
    understanding what a metric actually measures.
-   Gained a better appreciation of reproducibility, parameter
    sensitivity, computational cost and modular software design.

The project served as an initial practical exploration of latent feature
representations. Revisiting it provides an opportunity to assess the
original work more critically and identify how my understanding of
machine-learning experimentation and software development has developed
since then.
