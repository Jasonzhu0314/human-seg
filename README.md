# human-seg
Using LIP data and mobilenet for human segmentation.

常见的模型大小和计算量分析

|model_name|Base Model|Segmentation Model|Params|FLOPs|Model Size|Available|
|---|---|---|---|---|---|---|
|enet|ENet|Enet|371,558|759,829|1.4Mb|True|
|fcn8|Vanilla CNN|FCN8|3,609,196|7220708|29.0Mb|True|
|unet|Vanilla CNN|UNet|7,167,618|14,344,197|57.6Mb|True|
|attunet|Vanilla CNN|AttUNet|8,913,058|17,841,087|71.7Mb|True|
|r2unet|Vanilla CNN|R2UNet|17,652,930|51,065,008|141.7Mb|True|
|r2attunet|Vanilla CNN|R2AttUNet|16,958,530|46,532,640|136.2Mb|True|
|unet++|Vanilla CNN|NestedUNet|9,171,170|18,353,631|73.7Mb|True|
|segnet|Vanilla CNN|SegNet|2,941,218|5,888,377|11.9Mb|True|
|icnet|Vanilla CNN|ICNet|6,740,610|13,524,726|27.6Mb|True|
|pspnet*|Vanilla CNN|PSPNet|964,226|8,894,120|3.9Mb|True|
|mobilenet_unet|MobileNet|MobileNetUnet|407,778|825,856|1.9Mb|True|
|mobilenet_fcn8|MobileNet|MobileNetFCN8|3,432,764|6,880,358|14Mb|False|
|seunet|SENet|SEUNet|1,964,530|3,932,843|8.2Mb|True|
|scseunet|SCSENet|scSEUNet|1,959,266|3,923,359|8.1Mb|True|
|vggunet|VGGNet|VGGUnet|25,884,170|51,789,952|103.8Mb|True|
|unet_xception_resnetblock|XceptionNet|Unet_Xception_ResNetBlock|38,431,730|88,041,130|154.5Mb|True|
|deeplab_v2|DeepLab|DeepLabV2|37,799,752|75,574,697|151.3Mb|True|
|hrnet|HRNet|HRNet|9524168|57,356,440|117.1Mb|True|