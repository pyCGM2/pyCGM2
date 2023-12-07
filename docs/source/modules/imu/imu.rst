pyCGM2.IMU
===========================================================

.. toctree::
   :maxdepth: 2

Imu
------------


.. currentmodule:: pyCGM2.IMU.imu
.. autosummary::
   :toctree: generated
   :template: class.rst
   
   Imu



Filters
-------

.. currentmodule:: pyCGM2.IMU.imuFilters
.. autosummary::
   :toctree: generated
   :template: class.rst
   
   ImuReaderFilter

.. currentmodule:: pyCGM2.IMU.opensense.interface.opensenseFilters
    
.. autosummary::
   :toctree: generated
   :template: class.rst
   
   opensenseInterfaceImuPlacerFilter
   opensenseInterfaceImuInverseKinematicFilter


Procedures
----------

reader 
^^^^^^

.. currentmodule:: pyCGM2.IMU.Procedures.imuReaderProcedures
    
.. autosummary::
   :toctree: generated
   :template: class.rst
   
   ImuReaderProcedure
   CsvProcedure
   DataframeProcedure
   C3dBlueTridentProcedure


   


motion 
^^^^^^

.. currentmodule:: pyCGM2.IMU.Procedures.imuMotionProcedure 
.. autosummary::
   :toctree: generated
   :template: class.rst
   
   ImuMotionProcedure
   QuaternionMotionProcedure
   GlobalAngleMotionProcedure
   RealignedMotionProcedure
    
relative angles 
^^^^^^^^^^^^^^^

.. currentmodule:: pyCGM2.IMU.Procedures.relativeImuAngleProcedures 
.. autosummary::
   :toctree: generated
   :template: class.rst
   
   RelativeAnglesProcedure
   RelativeAnglesProcedure


opensense procedures 
^^^^^^^^^^^^^^^^^^^^^

placer
~~~~~~
.. currentmodule:: pyCGM2.IMU.opensense.interface.procedures.opensenseImuPlacerInterfaceProcedure 
                  
.. autosummary::
   :toctree: generated
   :template: class.rst
   
   ImuPlacerXMLProcedure

fitter
~~~~~~
.. currentmodule:: pyCGM2.IMU.opensense.interface.procedures.opensenseImuKinematicFitterProcedure 
.. autosummary::
   :toctree: generated
   :template: class.rst
   
   ImuInverseKinematicXMLProcedure






