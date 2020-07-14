## Nodes, Edges and Graphs {#pipe-nodes-edges-graphs}



POs are combined into [`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html)s.
The manual way (= hard way) to construct a [`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html)  is to create an empty graph first.
Then one fills the empty graph with POs, and connects edges between the POs.
Conceptually, this may look like this:


\begin{center}\includegraphics{images/po_nodes} \end{center}

POs are identified by their `$id`.
Note that the operations all modify the object in-place and return the object itself.
Therefore, multiple modifications can be chained.

For this example we use the `pca` PO defined above and a new PO named "mutate".
The latter creates a new feature from existing variables.
Additionally, we use the filter PO again.


```r
mutate = mlr_pipeops$get("mutate")

filter = mlr_pipeops$get("filter",
  filter = mlr3filters::FilterVariance$new(),
  param_vals = list(filter.frac = 0.5))
```


```r
graph = Graph$new()$
  add_pipeop(mutate)$
  add_pipeop(filter)$
  add_edge("mutate", "variance")  # add connection mutate -> filter
```

The much quicker way is to use the `%>>%` operator to chain POs or [`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html) s.
The same result as above can be achieved by doing the following:


```r
graph = mutate %>>% filter
```

Now the [`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html)  can be inspected using its `$plot()` function:


```r
graph$plot()
```



\begin{center}\includegraphics{04-pipelines-nodes-edges-graphs_files/figure-latex/04-pipelines-nodes-edges-graphs-006-1} \end{center}

**Chaining multiple POs of the same kind**

If multiple POs of the same kind should be chained, it is necessary to change the `id` to avoid name clashes.
This can be done by either accessing the `$id` slot or during construction:


```r
graph$add_pipeop(mlr_pipeops$get("pca"))
```


```r
graph$add_pipeop(mlr_pipeops$get("pca", id = "pca2"))
```
