from torch import nn
from .modeling.feature_decoder.encoder import PyramidPoolingModuleFPN
from .modeling.transformer_decoder.decoder import QORT_Decoder
from .modeling.backbone.resnet import build_resnet_backbone
from detectron2.layers import ShapeSpec

class QORT_Former(nn.Module):
    """
    Main class for QORT-Former architectures.
    """

    def __init__(
            self,
            *,
            num_classes
    ):
        super().__init__()
        self.backbone = build_resnet_backbone(input_shape={'res3': ShapeSpec(channels=512, height=None, width=None, stride=8),
                                                                  'res4': ShapeSpec(channels=1024, height=None, width=None, stride=16),
                                                                  'res5': ShapeSpec(channels=2048, height=None, width=None, stride=32)})

        self.feature_decoder = PyramidPoolingModuleFPN(input_shape={'res3': ShapeSpec(channels=512, height=None, width=None, stride=8),
                                                                  'res4': ShapeSpec(channels=1024, height=None, width=None, stride=16),
                                                                  'res5': ShapeSpec(channels=2048, height=None, width=None, stride=32)},
                                                     convs_dim=256,
                                                     mask_dim=256,
                                                     norm="GN")
        self.num_classes = num_classes
        self.predictor = QORT_Decoder(num_classes = self.num_classes, hidden_dim=256,
                                      num_queries=100, num_aux_queries=8, nheads=8,
                                      dec_layers=1, pre_norm=False, dim_feedforward=1024)

    def forward(self, images):
        features = self.backbone(images)
        mask_features, multi_scale_features = self.feature_decoder.forward_features(features)
        outputs = self.predictor(multi_scale_features, mask_features)
        return outputs