## Integrated Filter Methods {#list-filters}

### Standalone filter methods {#fs-filter-list}


\begin{tabular}{l|l|l|l}
\hline
Id & Packages & Task Types & Feature Types\\
\hline
[`anova`](https://mlr3filters.mlr-org.com/reference/mlr\_filters\_anova.html) & [stats](https://cran.r-project.org/package=stats) & classif & int, dbl\\
\hline
[`auc`](https://mlr3filters.mlr-org.com/reference/mlr\_filters\_auc.html) & [mlr3measures](https://cran.r-project.org/package=mlr3measures) & classif & int, dbl\\
\hline
[`carscore`](https://mlr3filters.mlr-org.com/reference/mlr\_filters\_carscore.html) & [care](https://cran.r-project.org/package=care) & regr & dbl\\
\hline
[`cmim`](https://mlr3filters.mlr-org.com/reference/mlr\_filters\_cmim.html) & [praznik](https://cran.r-project.org/package=praznik) & classif, regr & int, dbl, fct, ord\\
\hline
[`correlation`](https://mlr3filters.mlr-org.com/reference/mlr\_filters\_correlation.html) & [stats](https://cran.r-project.org/package=stats) & regr & int, dbl\\
\hline
[`disr`](https://mlr3filters.mlr-org.com/reference/mlr\_filters\_disr.html) & [praznik](https://cran.r-project.org/package=praznik) & classif & int, dbl, fct, ord\\
\hline
[`find\_correlation`](https://mlr3filters.mlr-org.com/reference/mlr\_filters\_find\_correlation.html) & [stats](https://cran.r-project.org/package=stats) & classif, regr & int, dbl\\
\hline
importance & [rpart](https://cran.r-project.org/package=rpart) & classif & lgl, int, dbl, fct, ord\\
\hline
[`information\_gain`](https://mlr3filters.mlr-org.com/reference/mlr\_filters\_information\_gain.html) & [FSelectorRcpp](https://cran.r-project.org/package=FSelectorRcpp) & classif, regr & int, dbl, fct, ord\\
\hline
[`jmi`](https://mlr3filters.mlr-org.com/reference/mlr\_filters\_jmi.html) & [praznik](https://cran.r-project.org/package=praznik) & classif & int, dbl, fct, ord\\
\hline
[`jmim`](https://mlr3filters.mlr-org.com/reference/mlr\_filters\_jmim.html) & [praznik](https://cran.r-project.org/package=praznik) & classif & int, dbl, fct, ord\\
\hline
[`kruskal\_test`](https://mlr3filters.mlr-org.com/reference/mlr\_filters\_kruskal\_test.html) & [stats](https://cran.r-project.org/package=stats) & classif & int, dbl\\
\hline
[`mim`](https://mlr3filters.mlr-org.com/reference/mlr\_filters\_mim.html) & [praznik](https://cran.r-project.org/package=praznik) & classif & int, dbl, fct, ord\\
\hline
[`mrmr`](https://mlr3filters.mlr-org.com/reference/mlr\_filters\_mrmr.html) & [praznik](https://cran.r-project.org/package=praznik) & classif & int, dbl, fct, ord\\
\hline
[`njmim`](https://mlr3filters.mlr-org.com/reference/mlr\_filters\_njmim.html) & [praznik](https://cran.r-project.org/package=praznik) & classif & int, dbl, fct, ord\\
\hline
[`performance`](https://mlr3filters.mlr-org.com/reference/mlr\_filters\_performance.html) &  & classif & lgl, int, dbl, fct, ord\\
\hline
[`variance`](https://mlr3filters.mlr-org.com/reference/mlr\_filters\_variance.html) & [stats](https://cran.r-project.org/package=stats) & classif, regr & int, dbl\\
\hline
\end{tabular}

### Learners With Embedded Filter Methods {#fs-filter-embedded-list}


```
##  [1] "classif.featureless" "classif.ranger"      "classif.rpart"      
##  [4] "classif.xgboost"     "regr.featureless"    "regr.ranger"        
##  [7] "regr.rpart"          "regr.xgboost"        "surv.gamboost"      
## [10] "surv.gbm"            "surv.mboost"         "surv.ranger"        
## [13] "surv.rpart"          "surv.xgboost"
```
