# Evaluation
Evaluation is a process that takes a number of inputs/outputs pairs and aggregates them. You can always use the model directly and just parse its inputs/outputs manually to perform evaluation. Alternatively, evaluation is implemented in LiBai using the `DatasetEvaluator` interface.

LiBai includes a few `DatasetEvaluator` that computes metrics like Acc@N, PPL. You can also implement your own `DatasetEvaluator` that performs some other jobs using the inputs/outputs pairs. For example, to count how many instances are detected on the validation set:
``` Python
class Counter(DatasetEvaluator):
  def reset(self):
    self.count = 0
  def process(self, inputs, outputs):
    for output in outputs:
      self.count += len(output["instances"])
  def evaluate(self):
    # save self.count somewhere, or print it, or return it.
    return {"count": self.count}
```
## Use evaluators
To evaluate using the methods of evaluators manually:
```
def get_all_inputs_outputs():
  for data in data_loader:
    yield data, model(data)

evaluator.reset()
for inputs, outputs in get_all_inputs_outputs():
  evaluator.process(inputs, outputs)
eval_results = evaluator.evaluate()
```

Evaluators can also be used with `inference_on_dataset`. For example,
```
eval_results = inference_on_dataset(
    model,
    data_loader,
    evaluator,
    ...
)
```
## Customize Evaluator using DatasetEvaluator
`DatasetEvaluator` is the Base class for a dataset evaluator. This class will accumulate information of the inputs/outputs (by `process`), and produce evaluation results in the end (by `evaluate`).

Firstly, create a new file in `libai/evaluation/`, and declare a new evaluator class that inherits the `DatasetEvaluator` and overwrites its `process` and `evaluation` functions to satisfy the needs.

For example, declare a `MyEvaluator` class in `libai/evaluator/myevaluator`.
``` Python
class MyEvaluator(DatasetEvaluator):
    def __init__(self):
        self._predictions = []

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        pred_logits = outputs["prediction_scores"]
        labels = inputs["labels"]

        # measure accuracy
        preds = pred_logits.cpu().topk(1)[1].squeeze(1).numpy()
        labels = labels.cpu().numpy()

        self._predictions.append({"preds": preds, "labels": labels})

    def evaluate(self):
        correct = 0.0
        all_sample = 0.0
        for pred in self._predictions:
            preds = pred["preds"]
            labels = pred["labels"]
            correct += (preds==labels).sum()
            all_sample += len(preds)
        self._results = OrderedDict()
        self._results["acc"] = correct/all_sample
        return copy.deepcopy(self._results)
```

Secondly, import the customized class in `libai/evaluation/__init__.py`
``` Python
from from .myevaluator import MyEvaluator
```
Lastly, set the evaluation in config.
``` Python
from libai.evaluation import MyEvaluator
evaluation=dict(
    enabled=True,
    evaluator=LazyCall(MyEvaluator)(),  # calculate top-k acc
    eval_metric="acc",
)
```
