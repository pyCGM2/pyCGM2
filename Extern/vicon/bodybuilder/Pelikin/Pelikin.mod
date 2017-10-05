{*   If printing out change page layout to landscape for maximum legiibility.   *}

{* =PeliKin=====================================================================*}
{* =======	      ****  **** *    *** *  *  *** *   *              ======== *}
{* =======            *   * *    *     *  *  *   *  **  *              ======== *}
{* =======	      ****  ***  *     *  ***    *  * * * 	       ======== *}
{* =======	      *     *    *     *  *  *   *  *  **   	       ======== *} 
{* =======	      *     **** **** *** *   * *** *   *	       ======== *}
{* ============================================================================ *}

{* 				  Version 1.0 					*}

{********************************************************************************}
{*										*}
{* Model to calculate pelvic kinematics in order rotation, obliquity, tilt	*}
{* Based on "Pelvic angles: a mathematically rigorous definition which is 	*}
{* consistent with a conventional clinical understanding of the terms". 	*}
{* Richard Baker. Gait and Posture 2001:13:16.					*}
{*										*}
{* Copywrite: Richard Baker 2001 						*}
{*										*}
{* Whilst every effort has been made to validate this model and ensure that it 	*}
{* is free from error, this cannot be guaranteed and users must conduct their 	*}
{* own tests to convice themselves of its appropriateness for their particular 	*}
{* application. Any corrections or general comments are welcome and should be 	*}
{* e-mailed to richard.baker@greenpark.n-i.nhs.uk.				*}
{*										*}
{* Model assumes lab axis system has z-up and that direction of progression	*}
{* is along x-axis in either direction.						*}
{*										*}
{* Requires markers labelled LASI, RASI and SACR.				*}
{*										*}
{* Ouput are LNewPelAngles and RNewPelAngles defined as the angles made with  	*}
{* the lab axis system re-defined such that the direction of progression	*}
{* is positive along the x-axis. 						*}
{*										*}
{* 	NewPelAngle(1) is tilt,   	anterior tilt is positive.		*}
{*	NewPelAngle(2) is obliquity	up is positive.				*}
{*      NewPelAngle(3) is rotation	forwards is positive			*}
{*										*}
{********************************************************************************}

{* Define pelvic origin and segment on basis of marker centres.			*}

SACR = (LPSI+RPSI)/2

PEL0	= (LASI+RASI)/2
Pelvis  = [PEL0,LASI-RASI,PEL0-SACR,yzx]

{* Work out which way subject is walking and define lab axis system such that   *}

If 1(PEL0) > 1(SACR) then 
  LAB = [{0,0,0}, {1,0,0},{0,0,1}, xyz]
else
  LAB = [{0,0,0}, {-1,0,0}, {0,0,1}, xyz]
endIf

{* Calculate pelvic angles and re-order to be consistent with above definitions.*}

NewPelAngles =-<LAB,Pelvis,zxy>
LNewPelAngles=<3(NewPelAngles),2(NewPelAngles),-1(NewPelAngles)>     
RNewPelAngles=<3(NewPelAngles),-2(NewPelAngles),1(NewPelAngles)>
output(LNewPelAngles,RNewPelAngles)


