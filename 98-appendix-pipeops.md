## Integrated Pipe Operators {#list-pipeops}



\begin{tabular}{l|l|l|l}
\hline
Id & Packages & Train & Predict\\
\hline
[`boxcox`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_boxcox.html) & [bestNormalize](https://cran.r-project.org/package=bestNormalize) & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`branch`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_branch.html) &  & * \$ightarrow * & *\$ightarrow*\\
\hline
[`chunk`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_chunk.html) &  & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`classbalancing`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_classbalancing.html) &  & TaskClassif \$ightarrow TaskClassif & TaskClassif\$ightarrowTaskClassif\\
\hline
[`classifavg`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_classifavg.html) & [stats](https://cran.r-project.org/package=stats) & NULL \$ightarrow NULL & PredictionClassif\$ightarrowPredictionClassif\\
\hline
[`classweights`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_classweights.html) &  & TaskClassif \$ightarrow TaskClassif & TaskClassif\$ightarrowTaskClassif\\
\hline
[`colapply`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_colapply.html) &  & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`collapsefactors`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_collapsefactors.html) &  & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`copy`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_copy.html) &  & * \$ightarrow * & *\$ightarrow*\\
\hline
[`crankcompose`](https://mlr3proba.mlr-org.com/reference/PipeOpCrankCompositor.html) & [distr6](https://cran.r-project.org/package=distr6) & NULL \$ightarrow NULL & PredictionSurv\$ightarrowPredictionSurv\\
\hline
[`distrcompose`](https://mlr3proba.mlr-org.com/reference/PipeOpDistrCompositor.html) & [distr6](https://cran.r-project.org/package=distr6) & NULL, NULL \$ightarrow NULL & PredictionSurv, PredictionSurv\$ightarrowPredictionSurv\\
\hline
[`encode`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_encode.html) & [stats](https://cran.r-project.org/package=stats) & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`encodeimpact`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_encodeimpact.html) &  & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`encodelmer`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_encodelmer.html) & [lme4](https://cran.r-project.org/package=lme4), [nloptr](https://cran.r-project.org/package=nloptr) & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`featureunion`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_featureunion.html) &  & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`filter`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_filter.html) &  & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`fixfactors`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_fixfactors.html) &  & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`histbin`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_histbin.html) & [graphics](https://cran.r-project.org/package=graphics) & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`ica`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_ica.html) & [fastICA](https://cran.r-project.org/package=fastICA) & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`imputehist`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_imputehist.html) & [graphics](https://cran.r-project.org/package=graphics) & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`imputemean`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_imputemean.html) &  & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`imputemedian`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_imputemedian.html) & [stats](https://cran.r-project.org/package=stats) & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`imputenewlvl`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_imputenewlvl.html) &  & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`imputesample`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_imputesample.html) &  & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`kernelpca`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_kernelpca.html) & [kernlab](https://cran.r-project.org/package=kernlab) & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`learner`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_learner.html) &  & TaskClassif \$ightarrow NULL & TaskClassif\$ightarrowPredictionClassif\\
\hline
[`learner\_cv`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_learner\_cv.html) &  & TaskClassif \$ightarrow TaskClassif & TaskClassif\$ightarrowTaskClassif\\
\hline
[`missind`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_missind.html) &  & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`modelmatrix`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_modelmatrix.html) & [stats](https://cran.r-project.org/package=stats) & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`mutate`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_mutate.html) &  & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`nop`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_nop.html) &  & * \$ightarrow * & *\$ightarrow*\\
\hline
[`pca`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_pca.html) &  & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`quantilebin`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_quantilebin.html) & [stats](https://cran.r-project.org/package=stats) & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`regravg`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_regravg.html) &  & NULL \$ightarrow NULL & PredictionRegr\$ightarrowPredictionRegr\\
\hline
[`removeconstants`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_removeconstants.html) &  & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`scale`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_scale.html) &  & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`scalemaxabs`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_scalemaxabs.html) &  & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`scalerange`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_scalerange.html) &  & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`select`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_select.html) &  & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`smote`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_smote.html) & [smotefamily](https://cran.r-project.org/package=smotefamily) & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`spatialsign`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_spatialsign.html) &  & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`subsample`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_subsample.html) &  & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
[`unbranch`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_unbranch.html) &  & * \$ightarrow * & *\$ightarrow*\\
\hline
[`yeojohnson`](https://mlr3pipelines.mlr-org.com/reference/mlr\_pipeops\_yeojohnson.html) & [bestNormalize](https://cran.r-project.org/package=bestNormalize) & Task \$ightarrow Task & Task\$ightarrowTask\\
\hline
\end{tabular}
