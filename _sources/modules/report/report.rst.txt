pyCGM2.Report 
===========================================================

.. toctree::
   :maxdepth: 1

   plot <plot>
   normativeDataset <normativeData>


Filters
-------

.. currentmodule:: pyCGM2.Report.plotFilters

.. autosummary::
    :toctree: generated
    :template: class.rst

    PlottingFilter

Viewers
-------

base viewers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Report.Viewers.plotViewers

.. autosummary::
    :toctree: generated
    :template: class.rst

    PlotViewer


spatiotemporal parameter viewers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Report.Viewers.plotViewers

.. autosummary::
    :toctree: generated
    :template: class.rst

    SpatioTemporalPlotViewer


kinematics and kinetics viewers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Report.Viewers.plotViewers

.. autosummary::
    :toctree: generated
    :template: class.rst

    GpsMapPlotViewer
    NormalizedKinematicsPlotViewer
    TemporalKinematicsPlotViewer
    NormalizedKineticsPlotViewer
    TemporalKineticsPlotViewer

emg viewers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Report.Viewers.emgPlotViewers

.. autosummary::
    :toctree: generated
    :template: class.rst

    TemporalEmgPlotViewer
    CoactivationEmgPlotViewer
    EnvEmgGaitPlotPanelViewer
    MultipleAnalysis_EnvEmgPlotPanelViewer
    
ground reaction viewers
^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Report.Viewers.groundReactionPlotViewers

.. autosummary::
    :toctree: generated
    :template: class.rst

    NormalizedGroundReactionForcePlotViewer
    NormalizedGaitGrfIntegrationPlotViewer
    NormalizedGaitMeanGrfIntegrationPlotViewer

muscle viewers
^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Report.Viewers.musclePlotViewers

.. autosummary::
    :toctree: generated
    :template: class.rst

    MuscleNormalizedPlotPanelViewer
    
custom viewers
^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Report.Viewers.customPlotViewers 

.. autosummary::
    :toctree: generated
    :template: class.rst

    SaggitalGagePlotViewer

comparison viewers
^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Report.Viewers.comparisonPlotViewers 

.. autosummary::
    :toctree: generated
    :template: class.rst

    KinematicsPlotComparisonViewer
    KineticsPlotComparisonViewer

.. _report-utils:

Utils
-------

.. currentmodule:: pyCGM2.Report.plotUtils

.. autosummary::
    :toctree: generated
    :template: functions.rst

    colorContext
