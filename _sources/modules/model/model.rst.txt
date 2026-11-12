pyCGM2.Model 
===========================================================

.. toctree::
   :maxdepth: 1

   Components <modelComponents>
   BodySegmentParameters <bodySegmentParameters>
   CGM <cgm2>
   


Filters
-------

.. currentmodule:: pyCGM2.Model.modelFilters

.. autosummary::
   :toctree: generated
   :template: class.rst
   
   ModelCalibrationFilter
   ModelMotionFilter
   ModelJCSFilter
   ModelAbsoluteAnglesFilter
   ForcePlateAssemblyFilter
   GroundReactionIntegrationFilter
   GroundReactionForceAdapterFilter
   InverseDynamicFilter
   JointPowerFilter
   CoordinateSystemDisplayFilter
   CentreOfMassFilter
   ModelMotionCorrectionFilter
   ModelQualityFilter



Procedures
----------

coordinate system
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Model.modelFilters

.. autosummary::
   :toctree: generated
   :template: class.rst

   GeneralCoordinateSystemProcedure
   ModelCoordinateSystemProcedure

calibration
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Model.modelFilters

.. autosummary::
   :toctree: generated
   :template: class.rst

   GeneralCalibrationProcedure
   StaticCalibrationProcedure

inverse dynamics
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Model.modelFilters

.. autosummary::
   :toctree: generated
   :template: class.rst

   InverseDynamicProcedure
   CGMLowerlimbInverseDynamicProcedure

model quality
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Model.Procedures.modelQuality

.. autosummary::
   :toctree: generated
   :template: class.rst

   QualityProcedure
   WandAngleQualityProcedure
   GeneralScoreResidualProcedure
   ModelScoreResidualProcedure

model correction
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Model.Procedures.modelMotionCorrection

.. autosummary::
   :toctree: generated
   :template: class.rst

   ModelCorrectionProcedure
   Naim2019ThighMisaligmentCorrectionProcedure

force plate integration
^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Model.Procedures.forcePlateIntegrationProcedures

.. autosummary::
   :toctree: generated
   :template: class.rst

   ForcePlateIntegrationProcedure
   GaitForcePlateIntegrationProcedure


Decorators
----------


Classes
^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Model.modelDecorator

.. autosummary::
   :toctree: generated
   :template: class.rst

   DecoratorModel
   Kad
   Cgm1ManualOffsets
   HipJointCenterDecorator
   KneeCalibrationDecorator
   AnkleCalibrationDecorator


functions
^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Model.modelDecorator
    
.. autosummary::
   :toctree: generated
   :template: function.rst

   footJointCentreFromMet
   VCMJointCentre
   chord
   midPoint
   calibration2Dof
   saraCalibration
   haraRegression
   harringtonRegression
   davisRegression
   bellRegression


Other models and tools
----------------------

single body model
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Model.Models.singleBody

.. autosummary::
   :toctree: generated
   :template: class.rst

   SingleBody

additional filters
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Model.modelFilters

.. autosummary::
   :toctree: generated
   :template: class.rst

   TrackingMarkerDecompositionFilter

motion
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Model.motion

.. autosummary::
   :toctree: generated
   :template: function.rst

   segmentalLeastSquare

anthropometric measurements
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Model.anthropometricMeasurement

.. autosummary::
   :toctree: generated
   :template: function.rst

   measureNorm

