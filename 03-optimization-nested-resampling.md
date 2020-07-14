## Nested Resampling {#nested-resampling}

In order to obtain unbiased performance estimates for learners, all parts of the model building (preprocessing and model selection steps) should be included in the resampling, i.e., repeated for every pair of training/test data.
For steps that themselves require resampling like hyperparameter tuning or feature-selection (via the wrapper approach) this results in two nested resampling loops.


\begin{center}\includegraphics[width=0.98\linewidth]{images/nested_resampling} \end{center}

The graphic above illustrates nested resampling for parameter tuning with 3-fold cross-validation in the outer and 4-fold cross-validation in the inner loop.

In the outer resampling loop, we have three pairs of training/test sets.
On each of these outer training sets parameter tuning is done, thereby executing the inner resampling loop.
This way, we get one set of selected hyperparameters for each outer training set.
Then the learner is fitted on each outer training set using the corresponding selected hyperparameters.
Subsequently, we can evaluate the performance of the learner on the outer test sets.

In [mlr3](https://mlr3.mlr-org.com), you can run nested resampling for free without programming any loops by using the [`mlr3tuning::AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) class.
This works as follows:

1. Generate a wrapped Learner via class [`mlr3tuning::AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) or `mlr3filters::AutoSelect` (not yet implemented).
2. Specify all required settings - see section ["Automating the Tuning"](#autotuner) for help.
3. Call function [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) or [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html) with the created [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html).

You can freely combine different inner and outer resampling strategies.

A common setup is prediction and performance evaluation on a fixed outer test set.
This can be achieved by passing the [`Resampling`](https://mlr3.mlr-org.com/reference/Resampling.html) strategy (`rsmp("holdout")`) as the outer resampling instance to either [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) or [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html).

The inner resampling strategy could be a cross-validation one (`rsmp("cv")`) as the sizes of the outer training sets might differ.
Per default, the inner resample description is instantiated once for every outer training set.

Note that nested resampling is computationally expensive.
For this reason we use relatively small search spaces and a low number of resampling iterations in the examples shown below.
In practice, you normally have to increase both.
As this is computationally intensive you might want to have a look at the section on [Parallelization](#parallelization).

### Execution {#nested-resamp-exec}

To optimize hyperparameters or conduct feature selection in a nested resampling you need to create learners using either:

* the [`AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) class, or
* the `mlr3filters::AutoSelect` class (not yet implemented)

We use the example from section ["Automating the Tuning"](#autotuner) and pipe the resulting learner into a [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) call.


```r
library("mlr3tuning")
task = tsk("iris")
learner = lrn("classif.rpart")
resampling = rsmp("holdout")
measure = msr("classif.ce")
param_set = paradox::ParamSet$new(
  params = list(paradox::ParamDbl$new("cp", lower = 0.001, upper = 0.1)))
terminator = term("evals", n_evals = 5)
tuner = tnr("grid_search", resolution = 10)

at = AutoTuner$new(learner, resampling, measure = measure,
  param_set, terminator, tuner = tuner)
```

Now construct the [`resample()`](https://mlr3.mlr-org.com/reference/resample.html) call:


```r
resampling_outer = rsmp("cv", folds = 3)
rr = resample(task = task, learner = at, resampling = resampling_outer)
```

```
## INFO  [11:26:40.852] Starting to optimize 1 parameter(s) with '<OptimizerGridSearch>' and '<TerminatorEvals>' 
## INFO  [11:26:40.881] Evaluating 1 configuration(s) 
## INFO  [11:26:40.959] Result of batch 1: 
## INFO  [11:26:40.962]   cp classif.ce  resample_result 
## INFO  [11:26:40.962]  0.1    0.06061 <ResampleResult> 
## INFO  [11:26:40.964] Evaluating 1 configuration(s) 
## INFO  [11:26:41.047] Result of batch 2: 
## INFO  [11:26:41.049]     cp classif.ce  resample_result 
## INFO  [11:26:41.049]  0.067    0.06061 <ResampleResult> 
## INFO  [11:26:41.051] Evaluating 1 configuration(s) 
## INFO  [11:26:41.099] Result of batch 3: 
## INFO  [11:26:41.101]     cp classif.ce  resample_result 
## INFO  [11:26:41.101]  0.023    0.06061 <ResampleResult> 
## INFO  [11:26:41.103] Evaluating 1 configuration(s) 
## INFO  [11:26:41.157] Result of batch 4: 
## INFO  [11:26:41.159]     cp classif.ce  resample_result 
## INFO  [11:26:41.159]  0.001    0.06061 <ResampleResult> 
## INFO  [11:26:41.161] Evaluating 1 configuration(s) 
## INFO  [11:26:41.210] Result of batch 5: 
## INFO  [11:26:41.212]     cp classif.ce  resample_result 
## INFO  [11:26:41.212]  0.034    0.06061 <ResampleResult> 
## INFO  [11:26:41.218] Finished optimizing after 5 evaluation(s) 
## INFO  [11:26:41.219] Result: 
## INFO  [11:26:41.220]   cp learner_param_vals x_domain classif.ce 
## INFO  [11:26:41.220]  0.1             <list>   <list>    0.06061 
## INFO  [11:26:41.266] Starting to optimize 1 parameter(s) with '<OptimizerGridSearch>' and '<TerminatorEvals>' 
## INFO  [11:26:41.274] Evaluating 1 configuration(s) 
## INFO  [11:26:41.329] Result of batch 1: 
## INFO  [11:26:41.331]     cp classif.ce  resample_result 
## INFO  [11:26:41.331]  0.001    0.06061 <ResampleResult> 
## INFO  [11:26:41.333] Evaluating 1 configuration(s) 
## INFO  [11:26:41.382] Result of batch 2: 
## INFO  [11:26:41.383]     cp classif.ce  resample_result 
## INFO  [11:26:41.383]  0.012    0.06061 <ResampleResult> 
## INFO  [11:26:41.385] Evaluating 1 configuration(s) 
## INFO  [11:26:41.434] Result of batch 3: 
## INFO  [11:26:41.436]     cp classif.ce  resample_result 
## INFO  [11:26:41.436]  0.045    0.06061 <ResampleResult> 
## INFO  [11:26:41.438] Evaluating 1 configuration(s) 
## INFO  [11:26:41.488] Result of batch 4: 
## INFO  [11:26:41.490]     cp classif.ce  resample_result 
## INFO  [11:26:41.490]  0.034    0.06061 <ResampleResult> 
## INFO  [11:26:41.496] Evaluating 1 configuration(s) 
## INFO  [11:26:41.545] Result of batch 5: 
## INFO  [11:26:41.547]     cp classif.ce  resample_result 
## INFO  [11:26:41.547]  0.089    0.06061 <ResampleResult> 
## INFO  [11:26:41.552] Finished optimizing after 5 evaluation(s) 
## INFO  [11:26:41.553] Result: 
## INFO  [11:26:41.554]     cp learner_param_vals x_domain classif.ce 
## INFO  [11:26:41.554]  0.001             <list>   <list>    0.06061 
## INFO  [11:26:41.600] Starting to optimize 1 parameter(s) with '<OptimizerGridSearch>' and '<TerminatorEvals>' 
## INFO  [11:26:41.603] Evaluating 1 configuration(s) 
## INFO  [11:26:41.654] Result of batch 1: 
## INFO  [11:26:41.656]     cp classif.ce  resample_result 
## INFO  [11:26:41.656]  0.089     0.0303 <ResampleResult> 
## INFO  [11:26:41.658] Evaluating 1 configuration(s) 
## INFO  [11:26:41.711] Result of batch 2: 
## INFO  [11:26:41.713]     cp classif.ce  resample_result 
## INFO  [11:26:41.713]  0.067     0.0303 <ResampleResult> 
## INFO  [11:26:41.715] Evaluating 1 configuration(s) 
## INFO  [11:26:41.764] Result of batch 3: 
## INFO  [11:26:41.766]     cp classif.ce  resample_result 
## INFO  [11:26:41.766]  0.034     0.0303 <ResampleResult> 
## INFO  [11:26:41.768] Evaluating 1 configuration(s) 
## INFO  [11:26:41.818] Result of batch 4: 
## INFO  [11:26:41.820]     cp classif.ce  resample_result 
## INFO  [11:26:41.820]  0.045     0.0303 <ResampleResult> 
## INFO  [11:26:41.822] Evaluating 1 configuration(s) 
## INFO  [11:26:41.876] Result of batch 5: 
## INFO  [11:26:41.877]     cp classif.ce  resample_result 
## INFO  [11:26:41.877]  0.012     0.0303 <ResampleResult> 
## INFO  [11:26:41.883] Finished optimizing after 5 evaluation(s) 
## INFO  [11:26:41.883] Result: 
## INFO  [11:26:41.884]     cp learner_param_vals x_domain classif.ce 
## INFO  [11:26:41.884]  0.089             <list>   <list>     0.0303
```

### Evaluation {#nested-resamp-eval}

With the created [`ResampleResult`](https://mlr3.mlr-org.com/reference/ResampleResult.html) we can now inspect the executed resampling iterations more closely.
See the section on [Resampling](#resampling) for more detailed information about [`ResampleResult`](https://mlr3.mlr-org.com/reference/ResampleResult.html) objects.

For example, we can query the aggregated performance result:


```r
rr$aggregate()
```

```
## classif.ce 
##    0.08667
```

Check for any errors in the folds during execution (if there is not output, warnings or errors recorded, this is an empty `data.table()`:


```r
rr$errors
```

```
## Empty data.table (0 rows and 2 cols): iteration,msg
```

Or take a look at the confusion matrix of the joined predictions:


```r
rr$prediction()$confusion
```

```
##             truth
## response     setosa versicolor virginica
##   setosa         50          0         0
##   versicolor      0         45         8
##   virginica       0          5        42
```
