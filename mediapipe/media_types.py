# data type supported
class dtype:
    """
    Class defining media data types supported.

    """
    NDT = ""
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"

class lastBatchStrategy:
    """
    Class defining last batch intended behaviour.

    """
    NONE = 0
    DROP = 1
    PARTIAL = 2
    PAD = 3
    CYCLIC = 4
    FILL = 5

# filter types
class ftype:
    """
    Class defining media decoder filters supported.

    """
    LINEAR = 0
    LANCZOS = 1
    NEAREST = 2
    BI_LINEAR = 3
    BICUBIC = 4
    SPLINE = 5
    BOX = 6


# layout types
class layout:
    """
    Class defining media layout supported.

    """
    NA = ""   # interleaved
    NHWC = "CWHN"   # interleaved
    NCHW = "WHCN"   # planar
    FHWC = "CWHC"   # video


# image type
class imgtype:
    """
    Class defining media decoder image types supported.

    """
    RGB_I = "rgb-i"
    RGB_P = "rgb-p"


class readerOutType:
    """
    Class defining media reader output type.

    """
    FILE_LIST = 0
    BUFFER_LIST = 1
    ADDRESS_LIST = 2


class randomCropType:
    """
    Class defining media random crop types.

    """
    NO_RANDOM_CROP = 0
    RANDOMIZED_AREA_AND_ASPECT_RATIO_CROP = 1
    RANDOMIZED_ASPECT_RATIO_CROP = 2
    CENTER_CROP = 3


class decoderStage:
    """
    Class defining media decoder stages.

    """
    ENABLE_ALL_STAGES = 0
    ENABLE_SINGLE_STAGE = 1
    ENABLE_TWO_STAGES = 2


class decoderType:
    """
    Class defining media decoder types.

    """
    IMAGE_DECODER = "image_decoder"
    VIDEO_DECODER = "video_decoder"


class clipSampler:
    """
    Class defining sampler for video clips

    """
    RANDOM_SAMPLER = 0
    UNIFORM_SAMPLER = 1

class cropWindowType:
    """
    Class defining media random crop types.

    """
    NO_CROP = 0
    RANDOMIZED_AREA_AND_ASPECT_RATIO_CROP = 1
    RANDOMIZED_ASPECT_RATIO_CROP = 2
    CENTER_CROP = 3

class mediaDeviceType:
    CPU = "cpu"
    MIXED = "mixed"
    LEGACY = "legacy"


