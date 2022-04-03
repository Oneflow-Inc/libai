import oneflow.nn as nn

from flowvision.models import ModelCreator

class VisionModel(nn.Module):
    """
    Wrap the model from flowvision to be compatible in LiBai

    Args:
        model_name (str): model to be used for training.
        pretrained (bool): load the pretrained weight or not.
        num_classes (int): number of classes to be predicted.
    """
    def __init__(self, model_name, pretrained=False, num_classes=1000, loss_func=None, **kwargs):
        super().__init__()
        self.model = ModelCreator.create_model(
            model_name = model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            **kwargs
        )
        # Loss func
        self.loss_func = nn.CrossEntropyLoss() if loss_func is None else loss_func
    
    def forward(self, images, labels=None):
        """

        Args:
            images (flow.Tensor): training samples.
            labels (flow.LongTensor, optional): training targets

        Returns:
            dict:
                A dict containing :code:`loss_value` or :code:`logits`
                depending on training or evaluation mode.
                :code:`{"losses": loss_value}` when training,
                :code:`{"prediction_scores": logits}` when evaluating.
        """
        x = self.model(images)

        if labels is not None and self.training:
            losses = self.loss_func(x, labels)
            return {"losses": losses}
        else:
            return {"prediction_scores": x}