## Hyperparameter Tuning {#tuning}

Hyperparameters are second-order parameters of machine learning models that, while often not explicitly optimized during the model estimation process, can have important impacts on the outcome and predictive performance of a model.
Typically, hyperparameters are fixed before training a model.
However, because the output of a model can be sensitive to the specification of hyperparameters, it is often recommended to make an informed decision about which hyperparameter settings may yield better model performance.
In many cases, hyperparameter settings may be chosen _a priori_, but it can be advantageous to try different settings before fitting your model on the training data.
This process is often called 'tuning' your model.

Hyperparameter tuning is supported via the extension package [mlr3tuning](https://mlr3tuning.mlr-org.com).
Below you can find an illustration of the process:


\begin{center}\includegraphics{images/tuning_process} \end{center}

At the heart of [mlr3tuning](https://mlr3tuning.mlr-org.com) are the R6 classes:

* [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html), [`TuningInstanceMultiCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceMultiCrit.html): This two classes describe the tuning problem and store the results.
* [`Tuner`](https://mlr3tuning.mlr-org.com/reference/Tuner.html): This class is the base class for implementations of tuning algorithms.

### The `TuningInstance` Classes {#tuning-optimization}

The following sub-section examines the optimization of a simple classification tree on the [`Pima Indian Diabetes`](https://mlr3.mlr-org.com/reference/mlr_tasks_pima.html) data set.


```r
task = tsk("pima")
print(task)
```

```
## <TaskClassif:pima> (768 x 9)
## * Target: diabetes
## * Properties: twoclass
## * Features (8):
##   - dbl (8): age, glucose, insulin, mass, pedigree, pregnant, pressure,
##     triceps
```

We use the classification tree from [rpart](https://cran.r-project.org/package=rpart) and choose a subset of the hyperparameters we want to tune.
This is often referred to as the "tuning space".


```r
learner = lrn("classif.rpart")
learner$param_set
```

```
## <ParamSet>
##                id    class lower upper levels     default value
## 1:       minsplit ParamInt     1   Inf                 20      
## 2:      minbucket ParamInt     1   Inf        <NoDefault>      
## 3:             cp ParamDbl     0     1               0.01      
## 4:     maxcompete ParamInt     0   Inf                  4      
## 5:   maxsurrogate ParamInt     0   Inf                  5      
## 6:       maxdepth ParamInt     1    30                 30      
## 7:   usesurrogate ParamInt     0     2                  2      
## 8: surrogatestyle ParamInt     0     1                  0      
## 9:           xval ParamInt     0   Inf                 10     0
```

Here, we opt to tune two parameters:

* The complexity `cp`
* The termination criterion `minsplit`

The tuning space has to be bound, therefore one has to set lower and upper bounds:


```r
library("paradox")
tune_ps = ParamSet$new(list(
  ParamDbl$new("cp", lower = 0.001, upper = 0.1),
  ParamInt$new("minsplit", lower = 1, upper = 10)
))
tune_ps
```

```
## <ParamSet>
##          id    class lower upper levels     default value
## 1:       cp ParamDbl 0.001   0.1        <NoDefault>      
## 2: minsplit ParamInt 1.000  10.0        <NoDefault>
```

Next, we need to specify how to evaluate the performance.
For this, we need to choose a [`resampling strategy`](https://mlr3.mlr-org.com/reference/Resampling.html) and a [`performance measure`](https://mlr3.mlr-org.com/reference/Measure.html).


```r
hout = rsmp("holdout")
measure = msr("classif.ce")
```

Finally, one has to select the budget available, to solve this tuning instance.
This is done by selecting one of the available [`Terminators`](https://bbotk.mlr-org.com/reference/Terminator.html):

* Terminate after a given time ([`TerminatorClockTime`](https://bbotk.mlr-org.com/reference/mlr_terminators_clock_time.html))
* Terminate after a given amount of iterations ([`TerminatorEvals`](https://bbotk.mlr-org.com/reference/mlr_terminators_evals.html))
* Terminate after a specific performance is reached ([`TerminatorPerfReached`](https://bbotk.mlr-org.com/reference/mlr_terminators_perf_reached.html))
* Terminate when tuning does not improve ([`TerminatorStagnation`](https://bbotk.mlr-org.com/reference/mlr_terminators_stagnation.html))
* A combination of the above in an *ALL* or *ANY* fashion ([`TerminatorCombo`](https://bbotk.mlr-org.com/reference/mlr_terminators_combo.html))

For this short introduction, we specify a budget of 20 evaluations and then put everything together into a [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html):


```r
library("mlr3tuning")

evals20 = term("evals", n_evals = 20)

instance = TuningInstanceSingleCrit$new(
  task = task,
  learner = learner,
  resampling = hout,
  measure = measure,
  search_space = tune_ps,
  terminator = evals20
)
instance
```

```
## <TuningInstanceSingleCrit>
## * State:  Not optimized
## * Objective: <ObjectiveTuning:classif.rpart_on_pima>
## * Search Space:
## <ParamSet>
##          id    class lower upper levels     default value
## 1:       cp ParamDbl 0.001   0.1        <NoDefault>      
## 2: minsplit ParamInt 1.000  10.0        <NoDefault>      
## * Terminator: <TerminatorEvals>
## * Terminated: FALSE
## * Archive:
## <Archive>
## Null data.table (0 rows and 0 cols)
```

To start the tuning, we still need to select how the optimization should take place.
In other words, we need to choose the **optimization algorithm** via the [`Tuner`](https://mlr3tuning.mlr-org.com/reference/Tuner.html) class.

### The `Tuner` Class

The following algorithms are currently implemented in [mlr3tuning](https://mlr3tuning.mlr-org.com):

* Grid Search ([`TunerGridSearch`](https://mlr3tuning.mlr-org.com/reference/mlr_tuners_grid_search.html))
* Random Search ([`TunerRandomSearch`](https://mlr3tuning.mlr-org.com/reference/mlr_tuners_random_search.html)) [@bergstra2012]
* Generalized Simulated Annealing ([`TunerGenSA`](https://mlr3tuning.mlr-org.com/reference/mlr_tuners_gensa.html))

In this example, we will use a simple grid search with a grid resolution of 5.


```r
tuner = tnr("grid_search", resolution = 5)
```

Since we have only numeric parameters, [`TunerGridSearch`](https://mlr3tuning.mlr-org.com/reference/mlr_tuners_grid_search.html) will create an equidistant grid between the respective upper and lower bounds.
As we have two hyperparameters with a resolution of 5, the two-dimensional grid consists of $5^2 = 25$ configurations.
Each configuration serves as hyperparameter setting for the previously defined [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) and triggers a 3-fold cross validation on the task.
All configurations will be examined by the tuner (in a random order), until either all configurations are evaluated or the [`Terminator`](https://bbotk.mlr-org.com/reference/Terminator.html) signals that the budget is exhausted.

### Triggering the Tuning {#tuning-triggering}

To start the tuning, we simply pass the [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html) to the `$optimize()` method of the initialized [`Tuner`](https://mlr3tuning.mlr-org.com/reference/Tuner.html).
The tuner proceeds as follows:

1. The [`Tuner`](https://mlr3tuning.mlr-org.com/reference/Tuner.html) proposes at least one hyperparameter configuration (the [`Tuner`](https://mlr3tuning.mlr-org.com/reference/Tuner.html) may propose multiple points to improve parallelization, which can be controlled via the setting `batch_size`).
2. For each configuration, the given [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) is fitted on the [`Task`](https://mlr3.mlr-org.com/reference/Task.html) using the provided [`Resampling`](https://mlr3.mlr-org.com/reference/Resampling.html).
   All evaluations are stored in the archive of the [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html).
3. The [`Terminator`](https://bbotk.mlr-org.com/reference/Terminator.html) is queried if the budget is exhausted.
   If the budget is not exhausted, restart with 1) until it is.
4. Determine the configuration with the best observed performance.
5. Store the best configurations as result in the instance object. 
   The best hyperparameter settings (`$result_learner_param_vals`) and the corresponding measured performance (`$result_y`) can be acessed from the instance.


```r
tuner$optimize(instance) #should return invisible(NULL)
```

```
## INFO  [11:26:32.553] Starting to optimize 2 parameter(s) with '<OptimizerGridSearch>' and '<TerminatorEvals>' 
## INFO  [11:26:32.589] Evaluating 1 configuration(s) 
## INFO  [11:26:32.734] Result of batch 1: 
## INFO  [11:26:32.736]     cp minsplit classif.ce  resample_result 
## INFO  [11:26:32.736]  0.001        8     0.2656 <ResampleResult> 
## INFO  [11:26:32.738] Evaluating 1 configuration(s) 
## INFO  [11:26:32.822] Result of batch 2: 
## INFO  [11:26:32.823]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:32.823]  0.02575       10     0.2812 <ResampleResult> 
## INFO  [11:26:32.825] Evaluating 1 configuration(s) 
## INFO  [11:26:32.875] Result of batch 3: 
## INFO  [11:26:32.877]   cp minsplit classif.ce  resample_result 
## INFO  [11:26:32.877]  0.1        8     0.2812 <ResampleResult> 
## INFO  [11:26:32.878] Evaluating 1 configuration(s) 
## INFO  [11:26:32.933] Result of batch 4: 
## INFO  [11:26:32.935]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:32.935]  0.02575        8     0.2812 <ResampleResult> 
## INFO  [11:26:32.937] Evaluating 1 configuration(s) 
## INFO  [11:26:32.986] Result of batch 5: 
## INFO  [11:26:32.988]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:32.988]  0.07525        5     0.2812 <ResampleResult> 
## INFO  [11:26:32.990] Evaluating 1 configuration(s) 
## INFO  [11:26:33.043] Result of batch 6: 
## INFO  [11:26:33.045]     cp minsplit classif.ce  resample_result 
## INFO  [11:26:33.045]  0.001        1     0.3359 <ResampleResult> 
## INFO  [11:26:33.047] Evaluating 1 configuration(s) 
## INFO  [11:26:33.103] Result of batch 7: 
## INFO  [11:26:33.105]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:33.105]  0.02575        5     0.2812 <ResampleResult> 
## INFO  [11:26:33.106] Evaluating 1 configuration(s) 
## INFO  [11:26:33.158] Result of batch 8: 
## INFO  [11:26:33.160]     cp minsplit classif.ce  resample_result 
## INFO  [11:26:33.160]  0.001       10     0.2578 <ResampleResult> 
## INFO  [11:26:33.162] Evaluating 1 configuration(s) 
## INFO  [11:26:33.214] Result of batch 9: 
## INFO  [11:26:33.215]   cp minsplit classif.ce  resample_result 
## INFO  [11:26:33.215]  0.1        5     0.2812 <ResampleResult> 
## INFO  [11:26:33.217] Evaluating 1 configuration(s) 
## INFO  [11:26:33.271] Result of batch 10: 
## INFO  [11:26:33.273]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:33.273]  0.07525       10     0.2812 <ResampleResult> 
## INFO  [11:26:33.275] Evaluating 1 configuration(s) 
## INFO  [11:26:33.326] Result of batch 11: 
## INFO  [11:26:33.328]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:33.328]  0.02575        1     0.2812 <ResampleResult> 
## INFO  [11:26:33.330] Evaluating 1 configuration(s) 
## INFO  [11:26:33.381] Result of batch 12: 
## INFO  [11:26:33.383]      cp minsplit classif.ce  resample_result 
## INFO  [11:26:33.383]  0.0505        1     0.2812 <ResampleResult> 
## INFO  [11:26:33.385] Evaluating 1 configuration(s) 
## INFO  [11:26:33.440] Result of batch 13: 
## INFO  [11:26:33.441]   cp minsplit classif.ce  resample_result 
## INFO  [11:26:33.441]  0.1       10     0.2812 <ResampleResult> 
## INFO  [11:26:33.443] Evaluating 1 configuration(s) 
## INFO  [11:26:33.494] Result of batch 14: 
## INFO  [11:26:33.496]      cp minsplit classif.ce  resample_result 
## INFO  [11:26:33.496]  0.0505        8     0.2812 <ResampleResult> 
## INFO  [11:26:33.498] Evaluating 1 configuration(s) 
## INFO  [11:26:33.551] Result of batch 15: 
## INFO  [11:26:33.553]      cp minsplit classif.ce  resample_result 
## INFO  [11:26:33.553]  0.0505        3     0.2812 <ResampleResult> 
## INFO  [11:26:33.555] Evaluating 1 configuration(s) 
## INFO  [11:26:33.611] Result of batch 16: 
## INFO  [11:26:33.612]      cp minsplit classif.ce  resample_result 
## INFO  [11:26:33.612]  0.0505       10     0.2812 <ResampleResult> 
## INFO  [11:26:33.614] Evaluating 1 configuration(s) 
## INFO  [11:26:33.668] Result of batch 17: 
## INFO  [11:26:33.670]     cp minsplit classif.ce  resample_result 
## INFO  [11:26:33.670]  0.001        5     0.2773 <ResampleResult> 
## INFO  [11:26:33.672] Evaluating 1 configuration(s) 
## INFO  [11:26:33.725] Result of batch 18: 
## INFO  [11:26:33.727]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:33.727]  0.07525        1     0.2812 <ResampleResult> 
## INFO  [11:26:33.734] Evaluating 1 configuration(s) 
## INFO  [11:26:33.785] Result of batch 19: 
## INFO  [11:26:33.787]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:33.787]  0.02575        3     0.2812 <ResampleResult> 
## INFO  [11:26:33.789] Evaluating 1 configuration(s) 
## INFO  [11:26:33.842] Result of batch 20: 
## INFO  [11:26:33.843]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:33.843]  0.07525        3     0.2812 <ResampleResult> 
## INFO  [11:26:33.849] Finished optimizing after 20 evaluation(s) 
## INFO  [11:26:33.850] Result: 
## INFO  [11:26:33.851]     cp minsplit learner_param_vals x_domain classif.ce 
## INFO  [11:26:33.851]  0.001       10             <list>   <list>     0.2578
```

```r
instance$result_learner_param_vals
```

```
## $xval
## [1] 0
## 
## $cp
## [1] 0.001
## 
## $minsplit
## [1] 10
```

```r
instance$result_y
```

```
## classif.ce 
##     0.2578
```

One can investigate all resamplings which were undertaken, as they are stored in the archive of the [`TuningInstanceSingleCrit`](https://mlr3tuning.mlr-org.com/reference/TuningInstanceSingleCrit.html) and can be accessed through `$data()` method:


```r
instance$archive$data()
```

```
##          cp minsplit classif.ce  resample_result x_domain           timestamp
##  1: 0.00100        8     0.2656 <ResampleResult>   <list> 2020-07-14 11:26:32
##  2: 0.02575       10     0.2812 <ResampleResult>   <list> 2020-07-14 11:26:32
##  3: 0.10000        8     0.2812 <ResampleResult>   <list> 2020-07-14 11:26:32
##  4: 0.02575        8     0.2812 <ResampleResult>   <list> 2020-07-14 11:26:32
##  5: 0.07525        5     0.2812 <ResampleResult>   <list> 2020-07-14 11:26:32
##  6: 0.00100        1     0.3359 <ResampleResult>   <list> 2020-07-14 11:26:33
##  7: 0.02575        5     0.2812 <ResampleResult>   <list> 2020-07-14 11:26:33
##  8: 0.00100       10     0.2578 <ResampleResult>   <list> 2020-07-14 11:26:33
##  9: 0.10000        5     0.2812 <ResampleResult>   <list> 2020-07-14 11:26:33
## 10: 0.07525       10     0.2812 <ResampleResult>   <list> 2020-07-14 11:26:33
## 11: 0.02575        1     0.2812 <ResampleResult>   <list> 2020-07-14 11:26:33
## 12: 0.05050        1     0.2812 <ResampleResult>   <list> 2020-07-14 11:26:33
## 13: 0.10000       10     0.2812 <ResampleResult>   <list> 2020-07-14 11:26:33
## 14: 0.05050        8     0.2812 <ResampleResult>   <list> 2020-07-14 11:26:33
## 15: 0.05050        3     0.2812 <ResampleResult>   <list> 2020-07-14 11:26:33
## 16: 0.05050       10     0.2812 <ResampleResult>   <list> 2020-07-14 11:26:33
## 17: 0.00100        5     0.2773 <ResampleResult>   <list> 2020-07-14 11:26:33
## 18: 0.07525        1     0.2812 <ResampleResult>   <list> 2020-07-14 11:26:33
## 19: 0.02575        3     0.2812 <ResampleResult>   <list> 2020-07-14 11:26:33
## 20: 0.07525        3     0.2812 <ResampleResult>   <list> 2020-07-14 11:26:33
##     batch_nr
##  1:        1
##  2:        2
##  3:        3
##  4:        4
##  5:        5
##  6:        6
##  7:        7
##  8:        8
##  9:        9
## 10:       10
## 11:       11
## 12:       12
## 13:       13
## 14:       14
## 15:       15
## 16:       16
## 17:       17
## 18:       18
## 19:       19
## 20:       20
```

In sum, the grid search evaluated 20/25 different configurations of the grid in a random order before the [`Terminator`](https://bbotk.mlr-org.com/reference/Terminator.html) stopped the tuning.

Now the optimized hyperparameters can take the previously created [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html), set the returned hyperparameters and train it on the full dataset.


```r
learner$param_set$values = instance$result_learner_param_vals
learner$train(task)
```

The trained model can now be used to make a prediction on external data.
Note that predicting on observations present in the `task`,  should be avoided.
The model has seen these observations already during tuning and therefore results would be statistically biased.
Hence, the resulting performance measure would be over-optimistic.
Instead, to get statistically unbiased performance estimates for the current task, [nested resampling](#nested-resamling) is required.

### Automating the Tuning {#autotuner}

The [`AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) wraps a learner and augments it with an automatic tuning for a given set of hyperparameters.
Because the [`AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) itself inherits from the [`Learner`](https://mlr3.mlr-org.com/reference/Learner.html) base class, it can be used like any other learner.
Analogously to the previous subsection, a new classification tree learner is created.
This classification tree learner automatically tunes the parameters `cp` and `minsplit` using an inner resampling (holdout).
We create a terminator which allows 10 evaluations, and use a simple random search as tuning algorithm:


```r
library("paradox")
library("mlr3tuning")

learner = lrn("classif.rpart")
tune_ps = ParamSet$new(list(
  ParamDbl$new("cp", lower = 0.001, upper = 0.1),
  ParamInt$new("minsplit", lower = 1, upper = 10)
))
terminator = term("evals", n_evals = 10)
tuner = tnr("random_search")

at = AutoTuner$new(
  learner = learner,
  resampling = rsmp("holdout"),
  measure = msr("classif.ce"),
  search_space = tune_ps,
  terminator = terminator,
  tuner = tuner
)
at
```

```
## <AutoTuner:classif.rpart.tuned>
## * Model: -
## * Parameters: xval=0
## * Packages: rpart
## * Predict Type: response
## * Feature types: logical, integer, numeric, factor, ordered
## * Properties: importance, missings, multiclass, selected_features,
##   twoclass, weights
```

We can now use the learner like any other learner, calling the `$train()` and `$predict()` method.
This time however, we pass it to [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html) to compare the tuner to a classification tree without tuning.
This way, the [`AutoTuner`](https://mlr3tuning.mlr-org.com/reference/AutoTuner.html) will do its resampling for tuning on the training set of the respective split of the outer resampling.
The learner then undertakes predictions using the test set of the outer resampling.
This yields unbiased performance measures, as the observations in the test set have not been used during tuning or fitting of the respective learner.
This is called [nested resampling](#nested-resampling).

To compare the tuned learner with the learner using its default, we can use [`benchmark()`](https://mlr3.mlr-org.com/reference/benchmark.html):


```r
grid = benchmark_grid(
  task = tsk("pima"),
  learner = list(at, lrn("classif.rpart")),
  resampling = rsmp("cv", folds = 3)
)

# avoid console output from mlr3tuning
logger = lgr::get_logger("mlr3tuning")
logger$set_threshold("warn")

bmr = benchmark(grid)
```

```
## INFO  [11:26:34.261] Starting to optimize 2 parameter(s) with '<OptimizerRandomSearch>' and '<TerminatorEvals>' 
## INFO  [11:26:34.287] Evaluating 1 configuration(s) 
## INFO  [11:26:34.338] Result of batch 1: 
## INFO  [11:26:34.340]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:34.340]  0.05672        5     0.2339 <ResampleResult> 
## INFO  [11:26:34.344] Evaluating 1 configuration(s) 
## INFO  [11:26:34.396] Result of batch 2: 
## INFO  [11:26:34.397]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:34.397]  0.08773        4     0.2339 <ResampleResult> 
## INFO  [11:26:34.401] Evaluating 1 configuration(s) 
## INFO  [11:26:34.457] Result of batch 3: 
## INFO  [11:26:34.459]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:34.459]  0.02869        7     0.2339 <ResampleResult> 
## INFO  [11:26:34.463] Evaluating 1 configuration(s) 
## INFO  [11:26:34.514] Result of batch 4: 
## INFO  [11:26:34.516]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:34.516]  0.07377        3     0.2339 <ResampleResult> 
## INFO  [11:26:34.520] Evaluating 1 configuration(s) 
## INFO  [11:26:34.575] Result of batch 5: 
## INFO  [11:26:34.577]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:34.577]  0.03554        2     0.2339 <ResampleResult> 
## INFO  [11:26:34.581] Evaluating 1 configuration(s) 
## INFO  [11:26:34.633] Result of batch 6: 
## INFO  [11:26:34.635]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:34.635]  0.01479        5     0.2222 <ResampleResult> 
## INFO  [11:26:34.639] Evaluating 1 configuration(s) 
## INFO  [11:26:34.696] Result of batch 7: 
## INFO  [11:26:34.698]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:34.698]  0.01341        6     0.2456 <ResampleResult> 
## INFO  [11:26:34.701] Evaluating 1 configuration(s) 
## INFO  [11:26:34.752] Result of batch 8: 
## INFO  [11:26:34.753]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:34.753]  0.07361        4     0.2339 <ResampleResult> 
## INFO  [11:26:34.757] Evaluating 1 configuration(s) 
## INFO  [11:26:34.809] Result of batch 9: 
## INFO  [11:26:34.811]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:34.811]  0.06566       10     0.2339 <ResampleResult> 
## INFO  [11:26:34.815] Evaluating 1 configuration(s) 
## INFO  [11:26:34.900] Result of batch 10: 
## INFO  [11:26:34.902]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:34.902]  0.08648        2     0.2339 <ResampleResult> 
## INFO  [11:26:34.910] Finished optimizing after 10 evaluation(s) 
## INFO  [11:26:34.911] Result: 
## INFO  [11:26:34.912]       cp minsplit learner_param_vals x_domain classif.ce 
## INFO  [11:26:34.912]  0.01479        5             <list>   <list>     0.2222 
## INFO  [11:26:34.962] Starting to optimize 2 parameter(s) with '<OptimizerRandomSearch>' and '<TerminatorEvals>' 
## INFO  [11:26:34.978] Evaluating 1 configuration(s) 
## INFO  [11:26:35.037] Result of batch 1: 
## INFO  [11:26:35.039]      cp minsplit classif.ce  resample_result 
## INFO  [11:26:35.039]  0.0783        4     0.2632 <ResampleResult> 
## INFO  [11:26:35.043] Evaluating 1 configuration(s) 
## INFO  [11:26:35.095] Result of batch 2: 
## INFO  [11:26:35.097]        cp minsplit classif.ce  resample_result 
## INFO  [11:26:35.097]  0.007256        1     0.3041 <ResampleResult> 
## INFO  [11:26:35.101] Evaluating 1 configuration(s) 
## INFO  [11:26:35.159] Result of batch 3: 
## INFO  [11:26:35.161]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:35.161]  0.03947        6     0.2632 <ResampleResult> 
## INFO  [11:26:35.165] Evaluating 1 configuration(s) 
## INFO  [11:26:35.216] Result of batch 4: 
## INFO  [11:26:35.218]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:35.218]  0.07897        7     0.2632 <ResampleResult> 
## INFO  [11:26:35.222] Evaluating 1 configuration(s) 
## INFO  [11:26:35.273] Result of batch 5: 
## INFO  [11:26:35.274]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:35.274]  0.02098        7     0.2515 <ResampleResult> 
## INFO  [11:26:35.278] Evaluating 1 configuration(s) 
## INFO  [11:26:35.336] Result of batch 6: 
## INFO  [11:26:35.338]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:35.338]  0.03763        9     0.2632 <ResampleResult> 
## INFO  [11:26:35.341] Evaluating 1 configuration(s) 
## INFO  [11:26:35.393] Result of batch 7: 
## INFO  [11:26:35.395]      cp minsplit classif.ce  resample_result 
## INFO  [11:26:35.395]  0.0107        2     0.2573 <ResampleResult> 
## INFO  [11:26:35.399] Evaluating 1 configuration(s) 
## INFO  [11:26:35.456] Result of batch 8: 
## INFO  [11:26:35.458]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:35.458]  0.02818        1     0.2515 <ResampleResult> 
## INFO  [11:26:35.462] Evaluating 1 configuration(s) 
## INFO  [11:26:35.513] Result of batch 9: 
## INFO  [11:26:35.515]        cp minsplit classif.ce  resample_result 
## INFO  [11:26:35.515]  0.004199       10     0.3099 <ResampleResult> 
## INFO  [11:26:35.519] Evaluating 1 configuration(s) 
## INFO  [11:26:35.571] Result of batch 10: 
## INFO  [11:26:35.573]        cp minsplit classif.ce  resample_result 
## INFO  [11:26:35.573]  0.002972        2     0.3041 <ResampleResult> 
## INFO  [11:26:35.581] Finished optimizing after 10 evaluation(s) 
## INFO  [11:26:35.582] Result: 
## INFO  [11:26:35.583]       cp minsplit learner_param_vals x_domain classif.ce 
## INFO  [11:26:35.583]  0.02098        7             <list>   <list>     0.2515 
## INFO  [11:26:35.638] Starting to optimize 2 parameter(s) with '<OptimizerRandomSearch>' and '<TerminatorEvals>' 
## INFO  [11:26:35.650] Evaluating 1 configuration(s) 
## INFO  [11:26:35.702] Result of batch 1: 
## INFO  [11:26:35.704]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:35.704]  0.09007       10     0.2749 <ResampleResult> 
## INFO  [11:26:35.708] Evaluating 1 configuration(s) 
## INFO  [11:26:35.766] Result of batch 2: 
## INFO  [11:26:35.768]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:35.768]  0.06572        1     0.2749 <ResampleResult> 
## INFO  [11:26:35.772] Evaluating 1 configuration(s) 
## INFO  [11:26:35.825] Result of batch 3: 
## INFO  [11:26:35.826]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:35.826]  0.03365        7     0.2749 <ResampleResult> 
## INFO  [11:26:35.830] Evaluating 1 configuration(s) 
## INFO  [11:26:35.887] Result of batch 4: 
## INFO  [11:26:35.889]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:35.889]  0.09594        7     0.2749 <ResampleResult> 
## INFO  [11:26:35.893] Evaluating 1 configuration(s) 
## INFO  [11:26:35.945] Result of batch 5: 
## INFO  [11:26:35.946]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:35.946]  0.02513        1     0.2982 <ResampleResult> 
## INFO  [11:26:35.950] Evaluating 1 configuration(s) 
## INFO  [11:26:36.007] Result of batch 6: 
## INFO  [11:26:36.010]      cp minsplit classif.ce  resample_result 
## INFO  [11:26:36.010]  0.0902        5     0.2749 <ResampleResult> 
## INFO  [11:26:36.014] Evaluating 1 configuration(s) 
## INFO  [11:26:36.066] Result of batch 7: 
## INFO  [11:26:36.068]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:36.068]  0.03387        3     0.2749 <ResampleResult> 
## INFO  [11:26:36.072] Evaluating 1 configuration(s) 
## INFO  [11:26:36.126] Result of batch 8: 
## INFO  [11:26:36.128]        cp minsplit classif.ce  resample_result 
## INFO  [11:26:36.128]  0.001811        1     0.3392 <ResampleResult> 
## INFO  [11:26:36.132] Evaluating 1 configuration(s) 
## INFO  [11:26:36.192] Result of batch 9: 
## INFO  [11:26:36.193]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:36.193]  0.03039        6     0.2749 <ResampleResult> 
## INFO  [11:26:36.197] Evaluating 1 configuration(s) 
## INFO  [11:26:36.248] Result of batch 10: 
## INFO  [11:26:36.250]       cp minsplit classif.ce  resample_result 
## INFO  [11:26:36.250]  0.08325        4     0.2749 <ResampleResult> 
## INFO  [11:26:36.258] Finished optimizing after 10 evaluation(s) 
## INFO  [11:26:36.259] Result: 
## INFO  [11:26:36.260]       cp minsplit learner_param_vals x_domain classif.ce 
## INFO  [11:26:36.260]  0.09007       10             <list>   <list>     0.2749
```

```r
bmr$aggregate(msrs(c("classif.ce", "time_train")))
```

```
##    nr  resample_result task_id          learner_id resampling_id iters
## 1:  1 <ResampleResult>    pima classif.rpart.tuned            cv     3
## 2:  2 <ResampleResult>    pima       classif.rpart            cv     3
##    classif.ce time_train
## 1:     0.2500   0.674000
## 2:     0.2526   0.006667
```

Note that we do not expect any differences compared to the non-tuned approach for multiple reasons:

* the task is too easy
* the task is rather small, and thus prone to overfitting
* the tuning budget (10 evaluations) is small
* [rpart](https://cran.r-project.org/package=rpart) does not benefit that much from tuning

