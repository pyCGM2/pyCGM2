pyCGM2.Model.Opensim.interface 
===========================================================

.. toctree::
   :maxdepth: 1



Interface
---------

.. currentmodule:: pyCGM2.Model.Opensim.interface.opensimInterface

.. autosummary::
   :toctree: generated
   :template: class.rst
   
   osimInterface
   osimCgmInterface
   opensimXmlInterface


Filters
-------

.. currentmodule:: pyCGM2.Model.Opensim.interface.opensimInterfaceFilters

.. autosummary::
   :toctree: generated
   :template: class.rst
   
   opensimInterfaceScalingFilter
   opensimInterfaceInverseKinematicsFilter
   opensimInterfaceInverseDynamicsFilter
   opensimInterfaceStaticOptimizationFilter
   opensimInterfaceAnalysesFilter

Procedures
----------

scaling
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Model.Opensim.interface.procedures.scaling.opensimScalingInterfaceProcedure

.. autosummary::
   :toctree: generated
   :template: class.rst

   ScalingXmlProcedure
   ScalingXmlCgmProcedure

kinematic fitting
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Model.Opensim.interface.procedures.inverseKinematics.opensimInverseKinematicsInterfaceProcedure

.. autosummary::
   :toctree: generated
   :template: class.rst

   InverseKinematicXmlProcedure
   InverseKinematicXmlCgmProcedure

inverse dynamics
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Model.Opensim.interface.procedures.inverseDynamics.opensimInverseDynamicsInterfaceProcedure

.. autosummary::
   :toctree: generated
   :template: class.rst

   InverseDynamicsXmlProcedure
   InverseDynamicsXmlCgmProcedure


static optimization
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Model.Opensim.interface.procedures.staticOptimisation.opensimStaticOptimizationInterfaceProcedure

.. autosummary::
   :toctree: generated
   :template: class.rst

   StaticOptimisationXmlProcedure
   StaticOptimisationXmlCgmProcedure

analysis report
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Model.Opensim.interface.procedures.analysisReport.opensimAnalysesInterfaceProcedure

.. autosummary::
   :toctree: generated
   :template: class.rst

   AnalysesXmlProcedure
   AnalysesXmlCgmProcedure
   AnalysesXmlCgmDrivenModelProcedure


Additional procedures
---------------------

base procedure
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Model.Opensim.interface.procedures.opensimProcedures

.. autosummary::
   :toctree: generated
   :template: class.rst

   OpensimInterfaceXmlProcedure

Kalman kinematic fitting
^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.Model.Opensim.interface.procedures.inverseKinematics.opensimInverseKinematicsInterfaceProcedure

.. autosummary::
   :toctree: generated
   :template: class.rst

   KalmanInverseKinematicXmlCgmProcedure

Legacy OpenSim filters
----------------------

.. currentmodule:: pyCGM2.Model.Opensim.opensimFilters

.. autosummary::
   :toctree: generated
   :template: class.rst

   GeneralOpensimCalibrationProcedure
   CgmOpensimCalibrationProcedures
   GeneralOpensimFittingProcedure
   CgmOpensimFittingProcedure
   opensimCalibrationFilter
   opensimFittingFilter

OpenSim files
--------------------

.. currentmodule:: pyCGM2.Model.Opensim.opensimIO

.. autosummary::
   :toctree: generated
   :template: class.rst

   ImuStorageFile
   OpensimDataFrame
   TrcDataFrame

OpenSim processing
--------------------

.. currentmodule:: pyCGM2.Model.Opensim.osimProcessing

.. autosummary::
   :toctree: generated
   :template: class.rst

   opensimModel
   opensimKinematicFitting

.. currentmodule:: pyCGM2.Model.Opensim.osimProcessing

.. autosummary::
   :toctree: generated
   :template: function.rst

   globalTransformationLabToOsim
   setGlobalTransormation_lab_osim
   smartTrcExport
   sto2pointValues
   mot2pointValues

