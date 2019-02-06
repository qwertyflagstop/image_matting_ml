import imgaug.augmenters as Augments
import imgaug as ia
import imgaug.parameters as Parameters
from imgaug import KeypointsOnImage, Keypoint, SegmentationMapOnImage
import numpy as np


class ImageSegAugmentor(object):

    def __init__(self):
        super(ImageSegAugmentor, self).__init__()
        sometimes = lambda aug: Augments.Sometimes(0.6, aug)
        self.pipeline = Augments.Sequential([
            Augments.OneOf([
            sometimes(Augments.Sharpen(alpha=(0, 0.5), lightness=(0.5, 2.0))),
            sometimes(Augments.OneOf([
                Augments.GaussianBlur((0, 4.0)),
                Augments.AverageBlur(k=(2, 8)),
                Augments.MedianBlur(k=(3, 15)),
            ]))]),
            sometimes(Augments.AdditiveGaussianNoise(loc=0, scale=5.0)),
            sometimes(Augments.AddToHueAndSaturation((-20,20))),
            Augments.Affine(scale=(0.95, 1.4)),
            Augments.Affine(rotate=(-20, 20)),
        ],random_order=False)

    def augment_image(self, image):
        augmented_images = self.pipeline.augment_images([image.astype(np.uint8)])[0]
        return augmented_images

    def augment_crop_images_masks(self, image, mask, num_classes):
        try:
            p = self.pipeline.to_deterministic()
            augmented_images = p.augment_images([image])[0]
            mask_on_image = SegmentationMapOnImage(arr=mask, shape=image.shape, nb_classes=num_classes)
            augmented_masks = p.augment_segmentation_maps([mask_on_image])[0]
            return augmented_images, augmented_masks.get_arr_int()
        except Exception as e:
            print(e)