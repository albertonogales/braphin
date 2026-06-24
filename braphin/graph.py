"""
braphin.graph — Multimodal graph class extending EEGraph.

``BRAPHINGraph`` subclasses ``eegraph.graph.Graph`` and adds full fMRI
support while keeping the parent's EEG pipeline completely unchanged.

Dependency arrow is strictly one-way:

    braphin  →  eegraph        (extension)
    eegraph     (knows nothing about braphin)

Usage
-----
    from braphin import Graph   # BRAPHINGraph exported as Graph

    # EEG — delegates entirely to the parent class
    g = Graph()
    g.load_data("subject.edf", modality="eeg")
    G, matrix = g.modelate(window_size=None, connectivity="plv", bands=["alpha"])

    # fMRI — handled by the child class
    g = Graph()
    g.load_data("subject.nii.gz", modality="fmri")
    G, matrix = g.modelate(window_size=None, connectivity="pearson_correlation")
"""

from eegraph.graph import Graph


class BRAPHINGraph(Graph):
    """
    Multimodal connectivity graph — EEG + fMRI.

    Inherits the full EEG pipeline from ``eegraph.graph.Graph`` and adds
    the five-stage fMRI pipeline (Input → Preprocess → Denoise → Transform
    → Connectivity) from the braphin package.

    Parameters
    ----------
    None — use ``load_data(path, modality=...)`` to load data.
    """

    def __init__(self):
        super().__init__()
        self.modality = None   # "eeg" | "fmri"

    # ------------------------------------------------------------------ #
    #  load_data                                                           #
    # ------------------------------------------------------------------ #

    def load_data(
        self,
        path,
        exclude=[None],
        electrode_montage_path=None,
        modality=None,
        **kwargs,
    ):
        """
        Load input data for either modality.

        Parameters
        ----------
        path : str
            Path to the input file.
        exclude : list, optional
            Channel names to exclude (EEG only).
        electrode_montage_path : str, optional
            Path to a montage file for channel renaming (EEG only).
        modality : {"eeg", "fmri"}, optional
            Select the processing pipeline.  "mri" is accepted as an alias
            for "fmri".
        **kwargs
            Extra arguments forwarded to the fMRI loader, e.g.:
            ``auxiliary_paths``, ``config``.
        """
        if modality == "mri":
            modality = "fmri"
        self.modality = modality

        # ── EEG: delegate entirely to the parent class ──────────────────
        if modality in ("eeg", None):
            # Use BRAPHIN's richer input abstraction so the EEGInputBundle
            # is available, then hand the MNE Raw object up to the parent.
            from braphin.importEEGData import InputEEGData

            input_data = InputEEGData(
                path,
                exclude=exclude,
                electrode_montage_path=electrode_montage_path,
            )
            bundle = input_data.load()

            # Store exactly what the parent's modelate() expects.
            self.data = bundle.raw_data        # MNE Raw
            self.ch_names = bundle.ch_names
            self.metadata = bundle.eeg_metadata or {}

            input_data.display_info(bundle)

        # ── fMRI: BRAPHIN pipeline ───────────────────────────────────────
        elif modality == "fmri":
            from braphin.importBRAPHINData import InputfMRIData

            input_data = InputfMRIData(path, **kwargs)
            self.data = input_data.load()      # BRAPHINInputBundle

            # ROI labels are not available until after Transform; ch_names
            # will be populated by modelate() once parcellation runs.
            self.ch_names = []
            self.metadata = {
                "fmri_path":       self.data.fmri_path,
                "fmri_metadata":   self.data.fmri_metadata,
                "auxiliary_files": self.data.auxiliary_files,
                "input_stage":     "import",
            }

            input_data.display_info(self.data)

        else:
            raise ValueError(
                f"Unsupported modality '{modality}'. Use 'eeg' or 'fmri'."
            )

    # ------------------------------------------------------------------ #
    #  modelate                                                            #
    # ------------------------------------------------------------------ #

    def modelate(
        self,
        window_size,
        connectivity,
        bands=[None],
        threshold=None,
        **kwargs,
    ):
        """
        Compute a connectivity graph from the loaded data.

        Parameters
        ----------
        window_size : int or float or None
            Window size.  ``None`` uses the full recording / ROI series.
        connectivity : str
            Connectivity measure name.
        bands : list, optional
            Frequency bands (EEG) or fMRI sub-bands from ``braphin.bands``.
        threshold : float, optional
            Edge weight threshold.
        **kwargs
            Extra arguments forwarded to ``ModelMRIData`` for fMRI, e.g.:
            ``atlas_data``, ``atlas_config``, ``preprocess_config``,
            ``denoise_config``, ``connectivity_config``.

        Returns
        -------
        G : NetworkX Graph
        connectivity_matrix : np.ndarray  (N × N)
        """
        # ── EEG: reuse the parent's implementation verbatim ─────────────
        if self.modality in ("eeg", None):
            return super().modelate(window_size, connectivity, bands, threshold)

        # ── fMRI: BRAPHIN five-stage pipeline ───────────────────────────
        elif self.modality == "fmri":
            from braphin.model import ModelMRIData

            model_data = ModelMRIData(
                data=self.data,
                ch_names=self.ch_names,
                connectivity=connectivity,
                **kwargs,
            )
            G, connectivity_matrix = model_data.connectivity_workflow(
                bands=bands,
                window_size=window_size,
                threshold=threshold,
            )

            # Propagate ROI labels back to Graph after parcellation.
            if model_data.ch_names:
                self.ch_names = model_data.ch_names

            self.metadata["modelate_stage"] = "completed"
            if model_data.transform_bundle is not None:
                self.metadata["transform_bundle"] = model_data.transform_bundle
            if model_data.connectivity_bundle is not None:
                self.metadata["connectivity_bundle"] = model_data.connectivity_bundle

            return G, connectivity_matrix

        else:
            raise ValueError(
                f"Unsupported modality '{self.modality}'. Use 'eeg' or 'fmri'."
            )
