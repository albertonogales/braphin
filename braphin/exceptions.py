class BRAPHINError(Exception):
    """
    Base exception for the BRAPHIN library.
    All project-specific exceptions inherit from this class.
    """

    pass


class BRAPHINInputError(BRAPHINError):
    """Error related to data input."""

    pass


class BRAPHINFormatError(BRAPHINError):
    """Error related to the internal format of the data."""

    pass


class AtlasError(BRAPHINError):
    """Error related to the atlas."""

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
    """Error related to connectivity computation."""

    pass
