# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class MultiPointsPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import MultiPointsPredictor

        args = dict(model="yolo11n.pt", source=ASSETS)
        predictor = MultiPointsPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        if not hasattr(self, 'n_p'):
            m = self.model.model.model[-1]  # last layer
            self.n_p = m.n_p

        preds = ops.multipoints_nms(
            preds,
            self.args.conf,
            self.args.iou,
            n_p=self.n_p,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            pred[:, 6:6+2*self.n_p] = ops.scale_multipoints(img.shape[2:], pred[:, 6:6+2*self.n_p], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, multipoints=pred, n_p=self.n_p))
        return results
