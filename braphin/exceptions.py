class BRAPHINError(Exception):
    """
    Base exception for the BRAPHIN library.
    All project-specific exceptions inherit from this class.
    """
    pass


class BRAPHINInputError(BRAPHINError):
    """
    Error related to data input.
    Examples:
    - unsupported file extension,
    - non-existent file,
    - invalid file combination,
    - etc.
    """
    pass


class BRAPHINFormatError(BRAPHINError):
    """
    Error related to the internal format of the data.
    Examples:
    - a matrix that is not 2-D,
    - an unexpected shape,
    - a NIfTI image without the expected dimensions,
    - etc.
    """
    pass


class AtlasError(BRAPHINError):
    """
    Error related to the atlas.
    Examples:
    - unsupported atlas,
    - missing atlas file,
    - incompatible configuration,
    - etc.
    """
    pass


class PreprocessingError(BRAPHINError):
    """
    Error related to the preprocessing stage.
    """
    pass


class DenoisingError(BRAPHINError):
    """
    Error related to the denoising stage.
    """
    pass


class TransformationError(BRAPHINError):
    """
    Error related to data transformation;
    for example, failures when mapping the atlas or extracting ROIs.
    """
    pass


class ConnectivityError(BRAPHINError):
    """
    Error related to connectivity computation.
    Examples:
    - no ROI time series available,
    - incorrect dimensions,
    - unsupported connectivity method,
    - etc.
    """
    pass
