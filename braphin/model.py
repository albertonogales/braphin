"""
braphin.model
=============

fMRI-specific model class for the BRAPHIN pipeline.

``ModelMRIData`` orchestrates the full five-stage fMRI connectivity pipeline:

    Input → Preprocess → Denoise → Transform → Connectivity

It is the fMRI counterpart of ``eegraph.modelateData.ModelData`` (EEG) and is
invoked from ``braphin.graph.BRAPHINGraph.modelate()`` when ``modality='fmri'``.
"""

import numpy as np


class ModelMRIData:
    """
    fMRI model class that integrates all pipeline stages without modifying
    EEGraph's original ``ModelData`` class.

    Unlike EEG, the full fMRI modelling requires running several sequential stages:

    1. Preprocess
    2. Denoise
    3. Transform to ROI space
    4. Connectivity

    This keeps the user-facing API simple via ``BRAPHINGraph``:

    .. code-block:: python

        g = Graph()
        g.load_data("subject.nii.gz", modality="fmri")
        G, matrix = g.modelate(connectivity="pearson_correlation")
    """

    def __init__(
        self,
        data,
        ch_names=None,
        connectivity="pearson_correlation",
        atlas_data=None,
        atlas_config=None,
        preprocess_config=None,
        denoise_config=None,
        connectivity_config=None,
    ):
        """
        Parameters
        ----------
        data : BRAPHINInputBundle
            Input bundle produced by ``InputfMRIData.load()``.
        ch_names : list, optional
            Channel names.  Accepted for API compatibility with EEG; in fMRI
            the real ROI labels are populated after the Transform stage.
        connectivity : str
            Connectivity method name (e.g. ``"pearson_correlation"``).
        atlas_data : nibabel image, optional
            Pre-loaded 3-D atlas image.
        atlas_config : AtlasConfig, optional
            Atlas parcellation configuration.
        preprocess_config : PreprocessConfig, optional
            Preprocessing configuration.
        denoise_config : DenoiseConfig, optional
            Denoising configuration.
        connectivity_config : ConnectivityConfig, optional
            Connectivity computation configuration.
        """
        self.data = data
        self.ch_names = ch_names if ch_names is not None else []
        self.connectivity = connectivity

        self.atlas_data = atlas_data
        self.atlas_config = atlas_config
        self.preprocess_config = preprocess_config
        self.denoise_config = denoise_config
        self.connectivity_config = connectivity_config

        # Intermediate bundles kept for debugging / inspection
        self.input_bundle = None
        self.preprocess_bundle = None
        self.denoise_bundle = None
        self.transform_bundle = None
        self.connectivity_bundle = None

    def _validate_input_bundle(self):
        """
        Verify that the received object looks like a valid BRAPHINInputBundle.
        """
        if self.data is None:
            raise ValueError("ModelMRIData received no input data.")

        if not hasattr(self.data, "fmri_image"):
            raise ValueError(
                "ModelMRIData expects a BRAPHINInputBundle with an 'fmri_image' attribute."
            )

        if not hasattr(self.data, "fmri_metadata"):
            raise ValueError(
                "ModelMRIData expects a BRAPHINInputBundle with an 'fmri_metadata' attribute."
            )

    def _normalize_connectivity_name(self, connectivity):
        """
        Normalise a connectivity method name.

        Delegates to ``CONNECTIVITY_ALIASES`` in ``braphin.tools`` for full
        coverage of all supported method names and aliases.
        """
        from braphin.tools import CONNECTIVITY_ALIASES

        if connectivity is None:
            return "pearson_correlation"
        normalized = str(connectivity).strip().lower()
        return CONNECTIVITY_ALIASES.get(normalized, normalized)

    def _build_connectivity_config(self, window_size, threshold):
        """
        Build a ``ConnectivityConfig`` from the API parameters.
        """
        from braphin.config import ConnectivityConfig

        if self.connectivity_config is not None:
            config = self.connectivity_config
        else:
            config = ConnectivityConfig()

        config.method = self._normalize_connectivity_name(self.connectivity)
        config.window_size = window_size
        config.threshold = threshold

        return config

    def connectivity_workflow(self, bands=None, window_size=None, threshold=None):
        """
        Execute the full MRI workflow.

        Note:
        - 'bands' is accepted for compatibility with EEG. If a list of fMRI bands
          is specified (e.g. ["slow4", "broadband"]), band-filtered connectivity
          will also be computed and stored in self.band_connectivity.
        - Returns (G, connectivity_matrix), as expected by graph.py.
        """
        if bands is None:
            bands = [None]
        self._validate_input_bundle()

        from braphin.connectivity import ModelBRAPHINConnectivityData
        from braphin.denoise import DenoiseBRAPHINData
        from braphin.preprocess import PreprocessBRAPHINData
        from braphin.transform import TransformBRAPHINData

        self.input_bundle = self.data

        # ------------------------------------------------------
        # 1) PREPROCESS
        # ------------------------------------------------------
        preprocessor = PreprocessBRAPHINData(
            input_bundle=self.input_bundle,
            config=self.preprocess_config,
        )
        self.preprocess_bundle = preprocessor.run()

        # ------------------------------------------------------
        # 2) DENOISE
        # ------------------------------------------------------
        denoiser = DenoiseBRAPHINData(
            preprocess_bundle=self.preprocess_bundle,
            config=self.denoise_config,
        )
        self.denoise_bundle = denoiser.run()

        # ------------------------------------------------------
        # 3) TRANSFORM
        # ------------------------------------------------------
        transformer = TransformBRAPHINData(
            denoise_bundle=self.denoise_bundle,
            atlas_data=self.atlas_data,
            config=self.atlas_config,
        )
        self.transform_bundle = transformer.run()

        # ------------------------------------------------------
        # 4) CONNECTIVITY
        # ------------------------------------------------------
        connectivity_config = self._build_connectivity_config(window_size, threshold)

        connectivity_modeler = ModelBRAPHINConnectivityData(
            transform_bundle=self.transform_bundle,
            config=connectivity_config,
        )
        self.connectivity_bundle = connectivity_modeler.run()

        # Copy before zeroing the diagonal so the stored bundle is never mutated.
        # Without the copy, fill_diagonal would corrupt connectivity_bundle.connectivity_matrix
        # in place, making it inconsistent with connectivity_metadata["diagonal_all_ones"].
        connectivity_matrix = np.array(self.connectivity_bundle.connectivity_matrix, copy=True)
        np.fill_diagonal(connectivity_matrix, 0.0)  # Remove self-loops for graph/visualisation only

        roi_labels = self.connectivity_bundle.roi_labels

        # Populate ch_names with real ROI labels once parcellation is complete
        if roi_labels:
            self.ch_names = list(roi_labels)

        roi_centroids_3d = getattr(self.transform_bundle, "roi_centroids_3d", None)

        centroid_coordinate_space = getattr(
            self.transform_bundle, "centroid_coordinate_space", None
        )

        if centroid_coordinate_space is None:
            centroid_coordinate_space = self.transform_bundle.transform_metadata.get(
                "centroid_coordinate_space", "world"
            )

        # Band-filtered connectivity (optional)
        from braphin.bands import FMRI_BANDS, compute_band_connectivity

        if bands is not None and bands != [None]:
            # bands can be a list like ["slow4"] or ["slow3", "broadband"]
            band_results = {}
            tr = getattr(self.transform_bundle, "tr", None)
            if tr is None:
                tr = self.transform_bundle.transform_metadata.get("tr", 2.0)
            for band_name in bands:
                if band_name in FMRI_BANDS:
                    band_results[band_name] = compute_band_connectivity(
                        roi_time_series=self.transform_bundle.roi_time_series,
                        tr=tr,
                        band=band_name,
                        method=self._normalize_connectivity_name(self.connectivity),
                    )
            if band_results:
                # Store in connectivity_bundle metadata; primary matrix unchanged
                self.band_connectivity = band_results

        from braphin.visualize import build_fmri_graph

        G = build_fmri_graph(
            connectivity_matrix=connectivity_matrix,
            roi_labels=roi_labels,
            roi_centroids_3d=roi_centroids_3d,
            centroid_coordinate_space=centroid_coordinate_space,
            projection="axial",
        )

        return G, connectivity_matrix

    def display_info(self, bundle=None):
        """
        Print a summary of the final fMRI connectivity result.
        """
        if bundle is None:
            bundle = self.connectivity_bundle

        if bundle is None:
            raise ValueError(
                "No BRAPHINConnectivityBundle is available. Call connectivity_workflow() first."
            )

        print("\n[BRAPHIN] fMRI connectivity computed")
        print(f"fMRI source: {bundle.fmri_path}")
        print(f"Method: {bundle.connectivity_metadata.get('method')}")
        print(f"ROI × time shape: {bundle.connectivity_metadata.get('roi_time_series_shape')}")
        print(
            f"Connectivity matrix shape: "
            f"{bundle.connectivity_metadata.get('connectivity_matrix_shape')}"
        )
        print(f"Symmetric matrix: {bundle.connectivity_metadata.get('matrix_is_symmetric')}")
        print(f"Diagonal all ones: {bundle.connectivity_metadata.get('diagonal_all_ones')}")
        print(
            f"Mean connectivity (off-diagonal): "
            f"{bundle.connectivity_metadata.get('mean_connectivity')}"
        )

        if bundle.applied_steps:
            print("Applied steps:")
            for step in bundle.applied_steps:
                print(f" - {step}")

        if bundle.pending_steps:
            print("Pending steps (not yet implemented):")
            for step in bundle.pending_steps:
                print(f" - {step}")
