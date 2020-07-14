## The Building Blocks: PipeOps {#pipe-pipeops}

The building blocks of [mlr3pipelines](https://mlr3pipelines.mlr-org.com) are **PipeOp**-objects (PO).
They can be constructed directly using `PipeOp<NAME>$new()`, but the recommended way is to retrieve them from the `mlr_pipeops` dictionary:


```r
library("mlr3pipelines")
as.data.table(mlr_pipeops)
```

```
##                 key      packages input.num output.num input.type.train
##  1:          boxcox bestNormalize         1          1             Task
##  2:          branch                       1         NA                *
##  3:           chunk                       1         NA             Task
##  4:  classbalancing                       1          1      TaskClassif
##  5:      classifavg         stats        NA          1             NULL
##  6:    classweights                       1          1      TaskClassif
##  7:        colapply                       1          1             Task
##  8: collapsefactors                       1          1             Task
##  9:            copy                       1         NA                *
## 10:          encode         stats         1          1             Task
## 11:    encodeimpact                       1          1             Task
## 12:      encodelmer   lme4,nloptr         1          1             Task
## 13:    featureunion                      NA          1             Task
## 14:          filter                       1          1             Task
## 15:      fixfactors                       1          1             Task
## 16:         histbin      graphics         1          1             Task
## 17:             ica       fastICA         1          1             Task
## 18:      imputehist      graphics         1          1             Task
## 19:      imputemean                       1          1             Task
## 20:    imputemedian         stats         1          1             Task
## 21:    imputenewlvl                       1          1             Task
## 22:    imputesample                       1          1             Task
## 23:       kernelpca       kernlab         1          1             Task
## 24:         learner                       1          1      TaskClassif
## 25:      learner_cv                       1          1      TaskClassif
## 26:         missind                       1          1             Task
## 27:     modelmatrix         stats         1          1             Task
## 28:          mutate                       1          1             Task
## 29:             nop                       1          1                *
## 30:             pca                       1          1             Task
## 31:     quantilebin         stats         1          1             Task
## 32:         regravg                      NA          1             NULL
## 33: removeconstants                       1          1             Task
## 34:           scale                       1          1             Task
## 35:     scalemaxabs                       1          1             Task
## 36:      scalerange                       1          1             Task
## 37:          select                       1          1             Task
## 38:           smote   smotefamily         1          1             Task
## 39:     spatialsign                       1          1             Task
## 40:       subsample                       1          1             Task
## 41:        unbranch                      NA          1                *
## 42:      yeojohnson bestNormalize         1          1             Task
##                 key      packages input.num output.num input.type.train
##     input.type.predict output.type.train output.type.predict
##  1:               Task              Task                Task
##  2:                  *                 *                   *
##  3:               Task              Task                Task
##  4:        TaskClassif       TaskClassif         TaskClassif
##  5:  PredictionClassif              NULL   PredictionClassif
##  6:        TaskClassif       TaskClassif         TaskClassif
##  7:               Task              Task                Task
##  8:               Task              Task                Task
##  9:                  *                 *                   *
## 10:               Task              Task                Task
## 11:               Task              Task                Task
## 12:               Task              Task                Task
## 13:               Task              Task                Task
## 14:               Task              Task                Task
## 15:               Task              Task                Task
## 16:               Task              Task                Task
## 17:               Task              Task                Task
## 18:               Task              Task                Task
## 19:               Task              Task                Task
## 20:               Task              Task                Task
## 21:               Task              Task                Task
## 22:               Task              Task                Task
## 23:               Task              Task                Task
## 24:        TaskClassif              NULL   PredictionClassif
## 25:        TaskClassif       TaskClassif         TaskClassif
## 26:               Task              Task                Task
## 27:               Task              Task                Task
## 28:               Task              Task                Task
## 29:                  *                 *                   *
## 30:               Task              Task                Task
## 31:               Task              Task                Task
## 32:     PredictionRegr              NULL      PredictionRegr
## 33:               Task              Task                Task
## 34:               Task              Task                Task
## 35:               Task              Task                Task
## 36:               Task              Task                Task
## 37:               Task              Task                Task
## 38:               Task              Task                Task
## 39:               Task              Task                Task
## 40:               Task              Task                Task
## 41:                  *                 *                   *
## 42:               Task              Task                Task
##     input.type.predict output.type.train output.type.predict
```

Single POs can be created using `mlr_pipeops$get(<name>)`:


```r
pca = mlr_pipeops$get("pca")
```

or using **syntactic sugar**


```r
pca = po("pca")
```

Some POs require additional arguments for construction:


```r
learner = mlr_pipeops$get("learner")

# Error in as_learner(learner) : argument "learner" is missing, with no default argument "learner" is missing, with no default
```


```r
learner = mlr_pipeops$get("learner", mlr_learners$get("classif.rpart"))
```

or in short `po("learner", lrn("classif.rpart"))`.

Hyperparameters of POs can be set through the `param_vals` argument.
Here we set the fraction of features for a filter:


```r
filter = mlr_pipeops$get("filter",
  filter = mlr3filters::FilterVariance$new(),
  param_vals = list(filter.frac = 0.5))
```

or in short notation:


```r
po("filter", mlr3filters::FilterVariance$new(), filter.frac = 0.5)
```

The figure below shows an exemplary `PipeOp`.
It takes an input, transforms it during `.$train` and `.$predict` and returns data:


\begin{center}\includegraphics{images/po_viz} \end{center}
