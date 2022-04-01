# Evaluation
Evaluation is a process that takes a number of inputs/outputs pairs and calculates them to get metrics. You can always use the model directly and parse its inputs/outputs manually to perform evaluation. Alternatively, evaluation can be implemented in LiBai using the `DatasetEvaluator` interface.

LiBai includes a few `DatasetEvaluator` that computes metrics like top-N accuracy, PPL(Perplexity), etc. You can also implement your own `DatasetEvaluator` that performs some other jobs using the inputs/outputs pairs. For example, to count how many instances are detected on the validation set:
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

## Customize Evaluator using DatasetEvaluator
`DatasetEvaluator` is the Base class for a dataset evaluator. This class accumulates information of the inputs/outputs (by `process`) after every batch inference, and produces evaluation results in the end (by `evaluate`). The input is from the `trainer.get_batch()`, which converts the outputs of `dataset.__getitem__()` to dict. The output is from the dict return of `model.forward()`.

Firstly, declare a new evaluator class that inherits the `DatasetEvaluator` and overwrites its `process` and `evaluation` functions to satisfy the needs.

For example, declare a `MyEvaluator` class in `libai/evaluator/myevaluator.py`:
``` Python
class MyEvaluator(DatasetEvaluator):
    def __init__(self):
        self._predictions = []

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        # the key of inputs/outputs can be customized
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

Secondly, import the customized class and set the evaluation in config:
``` Python
from libai.evaluation.myevaluator import MyEvaluator
evaluation=dict(
      enabled=True,
      # evaluator for calculating top-k acc
      evaluator=LazyCall(MyEvaluator)(),
      eval_period=5000,
      eval_iter=1e9,  # running steps for validation/test
      # Metrics to be used for best model checkpoint.
      eval_metric="acc", # your returned metric key in MyEvaluator.evaluate()
      eval_mode="max", # set `max` or `min` for saving best model according to your metric
)
```

## Run Evaluator Manually
To check your evaluator code outside `LiBai`, use the methods of evaluators manually:
``` Python
def get_all_inputs_outputs():
  for data in data_loader:
    yield data, model(data)

evaluator.reset()
for inputs, outputs in get_all_inputs_outputs():
  evaluator.process(inputs, outputs)
eval_results = evaluator.evaluate()
```

Evaluators can also be used with `inference_on_dataset`. For example:
``` Python
eval_results = inference_on_dataset(
    model,
    data_loader,
    evaluator,
    ...
)
```