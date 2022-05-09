# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Model
#APIDOC["Draft"]=False
#--end--
""" this module gathers classes/ functions for calibrating a model,
ie locating  joint centres and axis

"""

import numpy as np
from scipy.optimize import least_squares

import pyCGM2
from pyCGM2.Model import model
from pyCGM2 import enums
from pyCGM2.Tools import btkTools
from pyCGM2.Math import numeric
from pyCGM2.Math import geometry
from pyCGM2.Math import euler
from pyCGM2.Model import frame

LOGGER = pyCGM2.LOGGER


def footJointCentreFromMet(acq,side,frameInit,frameEnd,markerDiameter =14, offset =0 ):
    """calculate the foot centre from metatarsal markers.

    Args:
        acq (btk.acquisition): an acquisition
        side (str): body sideEnum
        frameInit (int): start frame
        frameEnd (int): end frame
        markerDiameter (double,Optional[14]): marker diameter.
        offset (double,Optional[0]): marker offset.

    """

    if side == "left":
        met2_base=acq.GetPoint("LTOE").GetValues()[frameInit:frameEnd,:].mean(axis=0)
        met5_head=acq.GetPoint("LVMH").GetValues()[frameInit:frameEnd,:].mean(axis=0)
        met1_head=acq.GetPoint("LFMH").GetValues()[frameInit:frameEnd,:].mean(axis=0)

        v1=(met5_head-met2_base)
        v1=np.nan_to_num(np.divide(v1,np.linalg.norm(v1)))

        v2=(met1_head-met2_base)
        v2=np.nan_to_num(np.divide(v2,np.linalg.norm(v2)))

        z = - np.cross(v1,v2)

        fjc = met2_base - (markerDiameter+offset) * z

    elif side == "right":
        met2_base=acq.GetPoint("RTOE").GetValues()[frameInit:frameEnd,:].mean(axis=0)
        met5_head=acq.GetPoint("RVMH").GetValues()[frameInit:frameEnd,:].mean(axis=0)
        met1_head=acq.GetPoint("RFMH").GetValues()[frameInit:frameEnd,:].mean(axis=0)

        v1=(met5_head-met2_base)
        v1=np.nan_to_num(np.divide(v1,np.linalg.norm(v1)))

        v2=(met1_head-met2_base)
        v2=np.nan_to_num(np.divide(v2,np.linalg.norm(v2)))

        z =  np.cross(v1,v2)

        fjc = met2_base - (markerDiameter+offset) * z

    return fjc


def VCMJointCentre(HalfJoint, JointMarker, TopJoint, StickMarker, beta = 0 ):
    """Calculate the joint centre according the Vicon Clinical Manager method
    ( ie chord function )

    Args:
        HalfJoint (double): radius of the joint
        JointMarker (array(1,3)): joint marker trajectory at a specific frame
        TopJoint (type): proximal joint centre trajectory at a specific frame
        StickMarker (type): lateral marker trajectory at a specific frame
        beta (double,Optional[0]): rotation angle offset.

    **Reference**

    Kabada, M., Ramakrishan, H., & Wooten, M. (1990). Measurement of lower extremity kinematics during level walking. Journal of Orthopaedic Research, 8, 383–392.
    """

    OffsetAngle = np.deg2rad(beta)

    if np.all(JointMarker==0) or np.all(TopJoint==0) or np.all(StickMarker==0):
        return np.zeros((3))

    X = TopJoint - JointMarker
    T = StickMarker - JointMarker
    P = np.cross( X, T )
    E = np.cross( X, P )

    E =np.divide(E,np.linalg.norm(E))

    x2 = np.dot( X, X )
    l = HalfJoint / x2
    m = 1 - HalfJoint * l


    if m > 0 :
        if np.abs( OffsetAngle ) > np.spacing(np.single(1)) :
            cosTheta = np.cos( OffsetAngle );
            r2 = HalfJoint * HalfJoint;
            r2cos2th = cosTheta * cosTheta * r2;
            r2cos2th_h2 = r2cos2th / (x2-r2);
            TdotX = np.dot( T, X );
            EdotT = np.dot( E, T );

            P =np.divide(P,np.linalg.norm(P))

            # solve quadratic
            a = 1 + r2cos2th_h2
            b= 2*(r2cos2th-TdotX*r2cos2th_h2-r2)
            c= r2*r2+r2cos2th_h2*TdotX*TdotX-r2cos2th*( np.dot( T, T )+r2 )

            disc=b*b-4*a*c;
            if disc < 0 :
                if disc < -np.abs(b)/1e6:
                    Solutions = [];
                  # return;
                else:
                    disc = 0;

            else:
              disc = np.sqrt( disc );

            Solutions = [ -b+disc, -b-disc ];
            if np.abs( a ) * 1e8 <= np.abs(c) and  np.min( np.abs(Solutions)) < 1e-8*np.max( np.abs( Solutions ) ) :
                Solutions = -c/b;
            else:
                a = a*2;
                Solutions = np.divide(Solutions,a)

            JointCentre = X * l * HalfJoint;
            lxt = l * HalfJoint * TdotX;
            r1l = r2 * m;

            n = len( Solutions );
            while( n > 0 ):
                if ( Solutions[n-1] < r2 ) == ( cosTheta > 0 ) :
                    mu = ( Solutions[n-1] - lxt ) / EdotT;
                    nu = r1l - mu*mu;
                    if nu > np.spacing(np.single(1)) :
                      nu = np.sqrt( nu );
                      if np.sin( OffsetAngle ) > 0 :
                        nu = -nu;

                      R = JointCentre + E*mu + P*nu;
                      return  R + JointMarker;
                n = n - 1;

            # if no solutions to the quadratic equation...
            E = X*l + E*(np.sqrt(m)*cosTheta) - P*np.sin(OffsetAngle);
            return JointMarker + E*HalfJoint;
        else:
            return JointMarker + X*(l*HalfJoint) + E*(np.sqrt(m)*HalfJoint);
    else:
        return JointMarker + E * HalfJoint;


def chord (offset,A1,A2,A3,beta=0.0, epsilon =0.001):
    # obsolete - use VCMJointCentre instead

    if (len(A1) != len(A2) or len(A1) != len(A3) or len(A2) != len(A3)):
        raise Exception ("length of input argument of chord function different")

    if np.all(A1==0) or np.all(A3==0) or np.all(A3==0):
        return np.zeros((3))

    arrayDim = len(A1.shape) # si 1 = array1d si 2 = array2d

    if arrayDim == 2:
        nrow = len(A1)
        returnValues = np.zeros((nrow,3))
    else:
        nrow =1
        returnValues = np.zeros((1,3))


    for i in range(0,nrow):
        I = A1[i,:] if arrayDim==2 else A1
        J = A2[i,:] if arrayDim==2 else A2
        K = A3[i,:] if arrayDim==2 else A3

        if beta == 0.0:
            y=np.nan_to_num(np.divide((J-I),np.linalg.norm(J-I)))
            x=np.cross(y,K-I)
            x=np.nan_to_num(np.divide((x),np.linalg.norm(x)))
            z=np.cross(x,y)

            matR=np.array([x,y,z]).T
            ori=(J+I)/2.0

            d=np.linalg.norm(I-J)
            theta=np.arcsin(offset/d)*2.0
            v_r=np.array([0, -d/2.0, 0])

            rot=np.array([[1,0,0],[0,np.cos(theta),-1.0*np.sin(theta)],[0,np.sin(theta),np.cos(theta)] ])

            P = np.dot(np.dot(matR,rot),v_r)+ori

        else:

            A=J
            B=I
            C=K
            L=offset

            eps =  epsilon

            AB = np.linalg.norm(A-B)
            alpha = np.arcsin(L/AB)
            AO = np.sqrt(AB*AB-L*L*(1+np.cos(alpha)*np.cos(alpha)))

            # chord avec beta nul
            #P = chord(L,B,A,C,beta=0.0) # attention ma methode . attention au arg input

            y=np.nan_to_num(np.divide((J-I),np.linalg.norm(J-I)))
            x=np.cross(y,K-I)
            x=np.nan_to_num(np.divide((x),np.linalg.norm(x)))
            z=np.cross(x,y)

            matR=np.array([x,y,z]).T
            ori=(J+I)/2.0

            d=np.linalg.norm(I-J)
            theta=np.arcsin(offset/d)*2.0
            v_r=np.array([0, -d/2.0, 0])

            rot=np.array([[1,0,0],[0,np.cos(theta),-1.0*np.sin(theta)],[0,np.sin(theta),np.cos(theta)] ])


            P= np.dot(np.dot(matR,rot),v_r)+ori
            # fin chord 0


            Salpha = 0
            diffBeta = np.abs(beta)
            alphaincr = beta # in degree


            # define P research circle in T plan
            n = np.nan_to_num(np.divide((A-B),AB))
            O = A - np.dot(n, AO)
            r = L*np.cos(alpha) #OK


            # build segment
            #T = BuildSegment(O,n,P-O,'zyx');
            Z=np.nan_to_num(np.divide(n,np.linalg.norm(n)))
            Y=np.nan_to_num(np.divide(np.cross(Z,P-O),np.linalg.norm(np.cross(Z,P-O))))
            X=np.nan_to_num(np.divide(np.cross(Y,Z),np.linalg.norm(np.cross(Y,Z))))
            Origin= O

            # erreur ici, il manque les norm
            T=np.array([[ X[0],Y[0],Z[0],Origin[0] ],
                        [ X[1],Y[1],Z[1],Origin[1] ],
                        [ X[2],Y[2],Z[2],Origin[2] ],
                        [    0,   0,   0,       1.0  ]])

            count = 0
            while diffBeta > eps or count > 100:
                if count > 100:
                    raise Exception("count boundary of Chord achieve")


                count = count + 1
                idiff = diffBeta

                Salpha = Salpha + alphaincr
                Salpharad = Salpha * np.pi / 180.0
                Pplan = np.array([  [r*np.cos(Salpharad)],
                                    [ r*np.sin(Salpharad)],
                                     [0],
                                    [1]])
                P = np.dot(T,Pplan)

                P = P[0:3,0]
                nBone = A-P

                ProjC = np.cross(nBone,np.cross(C-P,nBone))
                ProjB = np.cross(nBone,np.cross(B-P,nBone))


                sens = np.dot(np.cross(ProjC,ProjB).T,nBone)


                Betai = np.nan_to_num(np.divide(sens,np.linalg.norm(sens))) * np.arccos(np.nan_to_num(np.divide(np.dot(ProjC.T,ProjB),
                                                     np.linalg.norm(ProjC)*np.linalg.norm(ProjB))))*180.0/np.pi

                diffBeta = np.abs(beta - Betai)

                if (diffBeta - idiff) > 0:
                    if count == 1:
                        Salpha = Salpha - alphaincr
                        alphaincr = -alphaincr
                    else:
                        alphaincr = -alphaincr / 2.0;

        returnValues[i,:]=P

    if arrayDim ==1:
        out = returnValues[0,:]
    else:
        out = returnValues

    return out

def midPoint(acq,lateralMarkerLabel,medialMarkerLabel,offset=0):
    """return the mid point trajectory

    Args:
        acq (btk.Acquisition): An acquisition
        lateralMarkerLabel (str): label of the lateral marker
        medialMarkerLabel (str):  label of the medial marker
        offset (double,Optional[0]): offset

    **Note**

    if `offset` is different to 0, the mid point is computed from the statement:
        lateral + offset*(Vector[medial->lateral])
    """

    midvalues = np.zeros((acq.GetPointFrameNumber(),3))

    for i in range(0,acq.GetPointFrameNumber()):
        lateral = acq.GetPoint(lateralMarkerLabel).GetValues()[i,:]
        medial = acq.GetPoint(medialMarkerLabel).GetValues()[i,:]

        if offset !=0:
            v = medial-lateral
            v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

            midvalues[i,:] = lateral + (offset)*v
        else:
            midvalues[i,:] = (lateral + medial)/2.0

    return midvalues

def calibration2Dof(proxMotionRef,distMotionRef,indexFirstFrame,indexLastFrame,jointRange,sequence="YXZ",index=1,flexInd=0):
    """Calibration 2DOF ( aka dynakad)

    Args:
        proxMotionRef (pyCGM2.Model.Referential.motion): motion attribute of the proximal referential
        distMotionRef (pyCGM2.Model.Referential.motion): motion attribute of the distal referential
        indexFirstFrame (int): start frame
        indexLastFrame (int): end frame
        jointRange (list): minimum and maximum joint angle to process
        sequence (str,Optional[YXZ]): Euler sequence
        index (int,Optional[1]): coronal plane index
        flexInd (int,Optional[0]): sagital plane index

    """

    # onjective function : minimize variance of the knee varus valgus angle
    def objFun(x, proxMotionRef, distMotionRef,indexFirstFrame,indexLastFrame, sequence,index,jointRange):
        #nFrames= len(proxMotionRef)
        frames0 = range(0,len(proxMotionRef))

        if indexFirstFrame and indexLastFrame:
            frames = frames0[indexFirstFrame:indexLastFrame+1]

        elif  not indexFirstFrame and indexLastFrame:
            frames = frames0[:indexLastFrame+1]

        elif  indexFirstFrame and not indexLastFrame:
            frames = frames0[indexFirstFrame:]

        else:
            frames = frames0

        nFrames = len(frames)

        angle=np.deg2rad(x)
        rotZ = np.eye(3,3)
        rotZ[0,0] = np.cos(angle)
        rotZ[0,1] = - np.sin(angle)
        rotZ[1,0] = np.sin(angle)
        rotZ[1,1] = np.cos(angle)

        jointValues = np.zeros((nFrames,3))

        i=0
        for f in frames:
            Rprox = np.dot(proxMotionRef[f].getRotation(),rotZ)
            Rdist = distMotionRef[f].getRotation()

            Rrelative= np.dot(Rprox.T, Rdist)

            if sequence == "XYZ":
                Euler1,Euler2,Euler3 = euler.euler_xyz(Rrelative)
            elif sequence == "XZY":
                Euler1,Euler2,Euler3 = euler.euler_xzy(Rrelative)
            elif sequence == "YXZ":
                Euler1,Euler2,Euler3 = euler.euler_yxz(Rrelative)
            elif sequence == "YZX":
                Euler1,Euler2,Euler3 = euler.euler_yzx(Rrelative)
            elif sequence == "ZXY":
                Euler1,Euler2,Euler3 = euler.euler_zxy(Rrelative)
            elif sequence == "ZYX":
                Euler1,Euler2,Euler3 = euler.euler_zyx(Rrelative)
            else:
                raise Exception("[pyCGM2] joint sequence unknown ")

            jointValues[i,0] = Euler1
            jointValues[i,1] = Euler2
            jointValues[i,2] = Euler3
            i+=1

        if  jointRange is None:
            variance = np.var(jointValues[:,index])
        else:
            flexExt = jointValues[:,flexInd]
            indexes = np.where(np.logical_and(flexExt>=np.deg2rad(jointRange[0]), flexExt<=np.deg2rad(jointRange[1])))[0].tolist()

            if indexes == []:
                raise Exception ("[pyCGM2]. Calibration2-dof : There is no frames included in inputed joint limits")

            varVal = jointValues[:,index]
            varValExtract = varVal[indexes]

            variance = np.var(varValExtract)



        return variance

    x0 = 0.0 # deg
    res = least_squares(objFun, x0, args=(proxMotionRef, distMotionRef,indexFirstFrame,indexLastFrame,sequence,index,jointRange), verbose=2)

    return res.x[0]




def saraCalibration(proxMotionRef,distMotionRef,indexFirstFrame,indexLastFrame, offset = 100, method = "1"):
    """Computation of a joint center position from SARA.

    Args:
        proxMotionRef (pyCGM2.Model.Referential.motion): motion attribute of the proximal referential
        distMotionRef (pyCGM2.Model.Referential.motion): motion attribute of the distal referential
        offset (double,Optional[100]): distance in mm for positionning  axis boundaries
        method (int,Optional[1]): method index . Affect the objective function (see Ehrig et al.).



    **Reference**

    Ehrig, R., Taylor, W. R., Duda, G., & Heller, M. (2007). A survey of formal methods for determining functional joint axes. Journal of Biomechanics, 40(10), 2150–7.

    """
    frames0 = range(0,len(proxMotionRef))

    if indexFirstFrame and indexLastFrame:
        frames = frames0[indexFirstFrame:indexLastFrame+1]

    elif  not indexFirstFrame and indexLastFrame:
        frames = frames0[:indexLastFrame+1]

    elif  indexFirstFrame and not indexLastFrame:
        frames = frames0[indexFirstFrame:]

    nFrames = len(frames)

    if method =="1":

        A = np.zeros((nFrames*3,6))
        b = np.zeros((nFrames*3,1))

        i=0
        for f in frames:
            A[i*3:i*3+3,0:3] = proxMotionRef[f].getRotation()
            A[i*3:i*3+3,3:6] = -1.0 * distMotionRef[f].getRotation()
            b[i*3:i*3+3,:] = (distMotionRef[f].getTranslation() - proxMotionRef[f].getTranslation()).reshape(3,1)
            i+=1




        U,s,V = np.linalg.svd(A,full_matrices=False)
        V = V.T # beware of V ( there is a difference between numpy and matlab)
        invDiagS = np.identity(6) * (1/s) #s from sv is a line array not a matrix

        diagS=np.identity(6) * (s)

        CoR = V.dot(invDiagS).dot(U.T).dot(b)
        AoR = V[:,5]



    elif method =="2": # idem morgan's code

        SR = np.zeros((3,3))
        Sd = np.zeros((3,1))
        SRd = np.zeros((3,1))

        # For each frame compute the transformation matrix of the distal
        # segment in the proximal reference system

        for f in frames:
            Rprox = proxMotionRef[f].getRotation()
            tprox = proxMotionRef[f].getTranslation()

            Rdist = distMotionRef[f].getRotation()
            tdist = distMotionRef[f].getTranslation()


            P = np.concatenate((np.concatenate((Rprox,tprox.reshape(1,3).T),axis=1),np.array([[0,0,0,1]])),axis=0)
            D = np.concatenate((np.concatenate((Rdist,tdist.reshape(1,3).T),axis=1),np.array([[0,0,0,1]])),axis=0)

            T = np.dot(np.linalg.pinv(P),D)
            R = T[0:3,0:3 ]
            d= T[0:3,3].reshape(3,1)


            SR = SR + R
            Sd = Sd + d
            SRd = SRd + np.dot(R.T,d)



        A0 = np.concatenate((nFrames*np.eye(3),-SR),axis=1)
        A1 = np.concatenate((-SR.T,nFrames*np.eye(3)),axis=1)

        A = np.concatenate((A0,A1))
        b = np.concatenate((Sd,-SRd))

        # CoR
        CoR = np.dot(np.linalg.pinv(A),b)

        # AoR
        U,s,V = np.linalg.svd(A,full_matrices=False)
        V = V.T # beware of V ( there is a difference between numpy and matlab)

        diagS = np.identity(6) * (s)

        AoR = V[:,5]

    CoR_prox = CoR[0:3]
    CoR_dist = CoR[3:6]

    prox_axisNorm= np.nan_to_num(np.divide(AoR[0:3],np.linalg.norm(AoR[0:3])))
    dist_axisNorm= np.nan_to_num(np.divide(AoR[3:6],np.linalg.norm(AoR[3:6])))

    prox_origin = CoR_prox +  offset * prox_axisNorm.reshape(3,1)
    prox_axisLim = CoR_prox - offset * prox_axisNorm.reshape(3,1)

    dist_origin = CoR_dist + offset * dist_axisNorm.reshape(3,1)
    dist_axisLim = CoR_dist - offset * dist_axisNorm.reshape(3,1)


    S = diagS[3:6,3:6]
    coeffDet = S[2,2]/(np.trace(S)-S[2,2])

    return prox_origin.reshape(3),prox_axisLim.reshape(3),dist_origin.reshape(3),dist_axisLim.reshape(3),prox_axisNorm,dist_axisNorm,coeffDet





def haraRegression(mp_input,mp_computed,markerDiameter = 14.0,  basePlate = 2.0):
    """Hip joint centre regression from Hara et al, 2016

    Args
        mp_input (dict):  dictionary of the measured anthropometric parameters
        mp_computed (dict):  dictionary of the cgm-computed anthropometric parameters
        markerDiameter (double,Optional[14.0]):  diameter of the marker
        basePlate (double,Optional[2.0]): thickness of the base plate

    **Reference**

    Hara, R., Mcginley, J. L., C, B., Baker, R., & Sangeux, M. (2016). Generation of age and sex specific regression equations to locate the Hip Joint Centres. Gait & Posture

    """
    #TODO : remove mp_computed


    HJCx_L= 11.0 -0.063*mp_input["LeftLegLength"] - markerDiameter/2.0 - basePlate
    HJCy_L=8.0+0.086*mp_input["LeftLegLength"]
    HJCz_L=-9.0-0.078*mp_input["LeftLegLength"]

    LOGGER.logger.debug("Left HJC position from Hara [ X = %s, Y = %s, Z = %s]" %(HJCx_L,HJCy_L,HJCz_L))
    HJC_L_hara=np.array([HJCx_L,HJCy_L,HJCz_L])

    HJCx_R= 11.0 -0.063*mp_input["RightLegLength"]- markerDiameter/2.0 - basePlate
    HJCy_R=-1.0*(8.0+0.086*mp_input["RightLegLength"])
    HJCz_R=-9.0-0.078*mp_input["RightLegLength"]

    LOGGER.logger.debug("Right HJC position from Hara [ X = %s, Y = %s, Z = %s]" %(HJCx_R,HJCy_R,HJCz_R))
    HJC_R_hara=np.array([HJCx_R,HJCy_R,HJCz_R])

    HJC_L = HJC_L_hara
    HJC_R = HJC_R_hara

    return HJC_L,HJC_R



def harringtonRegression(mp_input,mp_computed, predictors, markerDiameter = 14.0, basePlate = 2.0, cgmReferential=True):
    """ Hip joint centre regression from Harrington et al, 2007

    Args:
        mp_input (dict):  dictionary of the measured anthropometric parameters
        mp_computed (dict):  dictionary of the cgm-computed anthropometric parameters
        predictors (str): predictor choice of the regression (full,PWonly,LLonly)
        markerDiameter (double,Optional[14.0]):  diameter of the marker
        basePlate (double,Optional[2.0]): thickness of the base plate
        cgmReferential (boolOptional[True]) - flag indicating HJC position will be expressed in the CGM pelvis Coordinate system


    **Notes**

      - Predictor choice allows using modified Harrington's regression from Sangeux 2015
      - `pelvisDepth`,`asisDistance` and `meanlegLength`  are automaticaly computed from CGM calibration


    **References**

      - Harrington, M., Zavatsky, A., Lawson, S., Yuan, Z., & Theologis, T. (2007). Prediction of the hip joint centre in adults, children, and patients with cerebral palsy based on magnetic resonance imaging. Journal of Biomechanics, 40(3), 595–602
      - Sangeux, M. (2015). On the implementation of predictive methods to locate the hip joint centres. Gait and Posture, 42(3), 402–405.

    """
    #TODO : how to work without CGM calibration

    if predictors.value == "full":
        HJCx_L=-0.24*mp_computed["PelvisDepth"]-9.9  - markerDiameter/2.0 - basePlate # post/ant
        HJCy_L=-0.16*mp_computed["InterAsisDistance"]-0.04*mp_computed["MeanlegLength"]-7.1
        HJCz_L=-1*(0.28*mp_computed["PelvisDepth"]+0.16*mp_computed["InterAsisDistance"]+7.9)
        HJC_L_har=np.array([HJCx_L,HJCy_L,HJCz_L])

        HJCx_R=-0.24*mp_computed["PelvisDepth"]-9.9 - markerDiameter/2.0 - basePlate# post/ant
        HJCy_R=-0.16*mp_computed["InterAsisDistance"]-0.04*mp_computed["MeanlegLength"]-7.1
        HJCz_R=1*(0.28*mp_computed["PelvisDepth"]+0.16*mp_computed["InterAsisDistance"]+7.9)
        HJC_R_har=np.array([HJCx_R,HJCy_R,HJCz_R])

    elif predictors.value=="PWonly":
        HJCx_L=-0.138*mp_computed["InterAsisDistance"]-10.4 - markerDiameter/2.0 - basePlate
        HJCy_L=-0.305*mp_computed["InterAsisDistance"]-10.9
        HJCz_L=-1*(0.33*mp_computed["InterAsisDistance"]+7.3)

        HJC_L_har=np.array([HJCx_L,HJCy_L,HJCz_L])

        HJCx_R=-0.138*mp_computed["InterAsisDistance"]-10.4 - markerDiameter/2.0 - basePlate
        HJCy_R=-0.305*mp_computed["InterAsisDistance"]-10.9
        HJCz_R=1*(0.33*mp_computed["InterAsisDistance"]+7.3)

        HJC_R_har=np.array([HJCx_R,HJCy_R,HJCz_R])


    elif predictors.value=="LLonly":
        HJCx_L=-0.041*mp_computed["MeanlegLength"]-6.3 - markerDiameter/2.0 - basePlate
        HJCy_L=-0.083*mp_computed["MeanlegLength"]-7.9
        HJCz_L=-1*(0.0874*mp_computed["MeanlegLength"]+5.4)

        HJC_L_har=np.array([HJCx_L,HJCy_L,HJCz_L])

        HJCx_R=-0.041*mp_computed["MeanlegLength"]-6.3 - markerDiameter/2.0 - basePlate
        HJCy_R=-0.083*mp_computed["MeanlegLength"]-7.9
        HJCz_R=1*(0.0874*mp_computed["MeanlegLength"]+5.4)

        HJC_R_har=np.array([HJCx_R,HJCy_R,HJCz_R])

    else:
        raise Exception("[pyCGM2] Predictor is unknown choixe possible : full, PWonly, LLonly")

    if cgmReferential :
        Rhar_cgm1=np.array([[1, 0, 0],[0, 0, -1], [0, 1, 0]])
        HJC_L = np.dot(Rhar_cgm1,HJC_L_har)
        HJC_R = np.dot(Rhar_cgm1,HJC_R_har)
        LOGGER.logger.debug("computation in cgm pelvis referential")
        LOGGER.logger.debug("Left HJC position from Harrington [ X = %s, Y = %s, Z = %s]" %(HJC_L[0],HJC_L[1],HJC_L[2]))
        LOGGER.logger.debug("Right HJC position from Harrington [ X = %s, Y = %s, Z = %s]" %(HJC_L[0],HJC_L[1],HJC_L[2]))
    else:
        HJC_L = HJC_L_har
        HJC_R = HJC_R_har

    return HJC_L,HJC_R


def davisRegression(mp_input,mp_computed, markerDiameter = 14.0, basePlate = 2.0):
    """Hip joint centre regression according Davis et al, 1991

    Args
        mp_input (dict):  dictionary of the measured anthropometric parameters
        mp_computed (dict):  dictionary of the cgm-computed anthropometric parameters
        markerDiameter (double,Optional[14.0]):  diameter of the marker
        basePlate (double,Optional[2.0]): thickness of the base plate

    **Reference:**

    Davis, R., Ounpuu, S., Tyburski, D., & Gage, J. (1991). A gait analysis data collection and reduction technique. Human Movement Science, 10, 575–587.


    """

    C=mp_computed["MeanlegLength"] * 0.115 - 15.3

    HJCx_L= C * np.cos(0.5) * np.sin(0.314) - (mp_computed["LeftAsisTrocanterDistance"] + markerDiameter/2.0) * np.cos(0.314)
    HJCy_L=-1*(C * np.sin(0.5) - (mp_computed["InterAsisDistance"] / 2.0))
    HJCz_L= - C * np.cos(0.5) * np.cos(0.314) - (mp_computed["LeftAsisTrocanterDistance"] + markerDiameter/2.0) * np.sin(0.314)

    HJC_L=np.array([HJCx_L,HJCy_L,HJCz_L])

    HJCx_R= C * np.cos(0.5) * np.sin(0.314) - (mp_computed["RightAsisTrocanterDistance"] + markerDiameter/2.0) * np.cos(0.314)
    HJCy_R=+1*(C * np.sin(0.5) - (mp_computed["InterAsisDistance"] / 2.0))
    HJCz_R= -C * np.cos(0.5) * np.cos(0.314) - (mp_computed["RightAsisTrocanterDistance"] + markerDiameter/2.0) * np.sin(0.314)

    HJC_R=np.array([HJCx_R,HJCy_R,HJCz_R])

    return HJC_L,HJC_R

def bellRegression(mp_input,mp_computed,  markerDiameter = 14.0, basePlate = 2.0, cgmReferential=True):
    """Hip joint centre regression from Bell and Brand et al, 2007

    Args
        mp_input (dict):  dictionary of the measured anthropometric parameters
        mp_computed (dict):  dictionary of the cgm-computed anthropometric parameters
        markerDiameter (double,Optional[14.0]):  diameter of the marker
        basePlate (double,Optional[2.0]): thickness of the base plate
        cgmReferential (bool,optional[True]) - flag indicating HJC position will be expressed in the CGM pelvis Coordinate system

    **Reference:**

      - Bell AL, Pederson DR, and Brand RA (1989) Prediction of hip joint center location from external landmarks.
        Human Movement Science. 8:3-16:

      - Bell AL, Pedersen DR, Brand RA (1990) A Comparison of the Accuracy of Several hip Center Location Prediction Methods.
        J Biomech. 23, 617-621.
    """


    HJCx_L= 0.36*mp_computed["InterAsisDistance"] # ML
    HJCy_L= -0.19*mp_computed["InterAsisDistance"] #AP
    HJCz_L=-0.19*mp_computed["InterAsisDistance"] # IS
    HJC_L_bell=np.array([HJCx_L,HJCy_L,HJCz_L])

    HJCx_R= -0.36*mp_computed["InterAsisDistance"] # ML
    HJCy_R= -0.19*mp_computed["InterAsisDistance"] #AP
    HJCz_R=-0.19*mp_computed["InterAsisDistance"] # IS
    HJC_R_bell=np.array([HJCx_R,HJCy_R,HJCz_R])



    if cgmReferential :
        Rbell_cgm1=np.array([[0, 1, 0],
                             [1, 0, 0],
                             [0, 0, 1]])
        HJC_L = np.dot(Rbell_cgm1,HJC_L_bell)
        HJC_R = np.dot(Rbell_cgm1,HJC_R_bell)
        LOGGER.logger.debug("computation in cgm pelvis referential")
        LOGGER.logger.debug("Left HJC position from Bell [ X = %s, Y = %s, Z = %s]" %(HJC_L[0],HJC_L[1],HJC_L[2]))
        LOGGER.logger.debug("Right HJC position from Bell [ X = %s, Y = %s, Z = %s]" %(HJC_L[0],HJC_L[1],HJC_L[2]))
    else:
        HJC_L = HJC_L_bell
        HJC_R = HJC_R_bell

    return HJC_L,HJC_R


# -------- ABSTRACT DECORATOR MODEL : INTERFACE ---------

class DecoratorModel(model.Model):
    def __init__(self, iModel):
        super(DecoratorModel,self).__init__()
        self.model = iModel

#-------- CONCRETE DECORATOR MODEL ---------
class Kad(DecoratorModel):
    """ A concrete CGM decorator altering the knee joint centre from the Knee Aligment device

    Args
      iModel (CGM2.cgm.CGM): a CGM instance
      iAcq (btk.Acquisition): btk aquisition instance of a static c3d with the KAD
    """
    def __init__(self, iModel,iAcq):
        super(Kad,self).__init__(iModel)
        self.acq = iAcq

    def compute(self,side="both",markerDiameter = 14):
        """ Run the KAD processing

        Args:
            side (str,Optional[both]): body side
            markerDiameter (double,Optional[14]): diameter of the marker
        """
        distSkin = 0

        ff = self.acq.GetFirstFrame()

        frameInit =  self.acq.GetFirstFrame()-ff
        frameEnd = self.acq.GetLastFrame()-ff+1

        #self.model.nativeCgm1 = False
        self.model.decoratedModel = True

        LKJCvalues =  np.zeros((self.acq.GetPointFrameNumber(),3))
        LKNEvalues =  np.zeros((self.acq.GetPointFrameNumber(),3))
        LAJCvalues = np.zeros((self.acq.GetPointFrameNumber(),3))

        RKJCvalues = np.zeros((self.acq.GetPointFrameNumber(),3))
        RKNEvalues =  np.zeros((self.acq.GetPointFrameNumber(),3))
        RAJCvalues = np.zeros((self.acq.GetPointFrameNumber(),3))


        if side == "both" or side == "left":

            self.model.setCalibrationProperty("LeftKAD",True)

            if isinstance(self.model,pyCGM2.Model.CGM2.cgm.CGM):
                # cancel shankRotation and thighRotation offset if contain a previous non-zero values
                if "LeftThighRotation" in self.model.mp : self.model.mp["LeftThighRotation"] =0 # look out, it's mp, not mp_computed.
                if "LeftShankRotation" in self.model.mp : self.model.mp["LeftShankRotation"] =0

            for i in range(0,self.acq.GetPointFrameNumber()):
                #  compute points left and right lateral condyle
                LKAX = self.acq.GetPoint("LKAX").GetValues()[i,:]
                LKD1 = self.acq.GetPoint("LKD1").GetValues()[i,:]
                LKD2 = self.acq.GetPoint("LKD2").GetValues()[i,:]

                dist = np.array([np.linalg.norm(LKAX-LKD1), np.linalg.norm(LKAX-LKD2),np.linalg.norm(LKD1-LKD2)] )
                dist =  dist / np.sqrt(2)
                res = np.array([np.mean(dist), np.var(dist)])
                n = np.cross(LKD2-LKD1 , LKAX-LKD1)
                n= np.nan_to_num(np.divide(n,np.linalg.norm(n)))

                I = (LKD1+LKAX)/2
                PP1 = 2/3.0*(I-LKD2)+LKD2
                O = PP1 - n*np.sqrt(3)*res[0]/3.0
                LKAXO = np.nan_to_num(np.divide((O-LKAX),np.linalg.norm(O-LKAX)))

                LKNEvalues[i,:] = O + LKAXO * distSkin

                # locate KJC
    #            LKJC = LKNE + LKAXO * (self.model.mp["leftKneeWidth"]+markerDiameter )/2.0
                if btkTools.isPointExist(self.acq,"LHJC"):
                    LHJC = self.acq.GetPoint("LHJC").GetValues()[i,:]
                    LKJCvalues[i,:] = VCMJointCentre( (self.model.mp["LeftKneeWidth"]+markerDiameter )/2.0 ,LKNEvalues[i,:],LHJC,LKAX, beta= 0.0 )
                else:
                    LKJCvalues[i,:] = LKNEvalues[i,:] + LKAXO * (self.model.mp["LeftKneeWidth"]+markerDiameter )/2.0

                # locate AJC
                beta = 0
                ajcDesc = "KAD"
                if "LeftTibialTorsion" in self.model.mp and self.model.mp["LeftTibialTorsion"] !=0:
                    beta = -1.0 * self.model.mp["LeftTibialTorsion"]
                    ajcDesc = "KAD-manualTT"


                LANK = self.acq.GetPoint("LANK").GetValues()[i,:]
                LAJCvalues[i,:] = VCMJointCentre( (self.model.mp["LeftAnkleWidth"]+markerDiameter )/2.0 ,LANK,LKJCvalues[i,:],LKAX,beta= beta )

            tf_prox = self.model.getSegment("Left Thigh").getReferential("TF")
            tf_dist = self.model.getSegment("Left Shank").getReferential("TF")
            tf_dist2 = self.model.getSegment("Left Foot").getReferential("TF")
            # add nodes to referential

            # nodes
            tf_prox.static.addNode("LKNE_kad",LKNEvalues.mean(axis=0), positionType="Global", desc = "KAD" )
            tf_prox.static.addNode("LKJC_kad",LKJCvalues.mean(axis=0), positionType="Global", desc = "KAD" )
            tf_dist.static.addNode("LKJC_kad",LKJCvalues.mean(axis=0), positionType="Global", desc = "KAD")

            tf_prox.static.addNode("LKJC",LKJCvalues.mean(axis=0), positionType="Global", desc = "KAD" )
            tf_dist.static.addNode("LKJC",LKJCvalues.mean(axis=0), positionType="Global", desc = "KAD")

            # add nodes to referential
            tf_dist.static.addNode("LAJC_kad",LAJCvalues.mean(axis=0), positionType="Global", desc = ajcDesc)
            tf_dist.static.addNode("LAJC",LAJCvalues.mean(axis=0), positionType="Global", desc = ajcDesc)

            tf_dist2.static.addNode("LAJC_kad",LAJCvalues.mean(axis=0), positionType="Global", desc = ajcDesc)
            tf_dist2.static.addNode("LAJC",LAJCvalues.mean(axis=0), positionType="Global", desc = ajcDesc)


        if side == "both" or side == "right":

            self.model.setCalibrationProperty("RightKAD",True)

            if isinstance(self.model,pyCGM2.Model.CGM2.cgm.CGM):

                # cancel shankRotation and thighRotation offset if contain a previous non-zero values
                if "RightThighRotation" in self.model.mp : self.model.mp["RightThighRotation"] =0 # look out, it's mp, not mp_computed.
                if "RightShankRotation" in self.model.mp : self.model.mp["RightShankRotation"] =0


            for i in range(0,self.acq.GetPointFrameNumber()):
                #  compute points left and right lateral condyle
                RKAX = self.acq.GetPoint("RKAX").GetValues()[i,:]
                RKD1 = self.acq.GetPoint("RKD1").GetValues()[i,:]
                RKD2 = self.acq.GetPoint("RKD2").GetValues()[i,:]

                dist = np.array([np.linalg.norm(RKAX-RKD1), np.linalg.norm(RKAX-RKD2),np.linalg.norm(RKD1-RKD2)] )
                dist =  dist / np.sqrt(2)
                res = np.array([np.mean(dist), np.var(dist)])
                n = np.cross(RKD2-RKD1 , RKAX-RKD1)
                n= np.nan_to_num(np.divide(n,np.linalg.norm(n)))

                n=-n # look out the negative sign


                I = (RKD1+RKAX)/2
                PP1 = 2/3.0*(I-RKD2)+RKD2
                O = PP1 - n*np.sqrt(3)*res[0]/3.0
                RKAXO = np.nan_to_num(np.divide((O-RKAX),np.linalg.norm(O-RKAX)))
                RKNEvalues[i,:] = O + RKAXO * distSkin

                # locate KJC
    #            RKJC = RKNE + RKAXO * (self.model.mp["rightKneeWidth"]+markerDiameter )/2.0
                if btkTools.isPointExist(self.acq,"RHJC"):
                    RHJC = self.acq.GetPoint("RHJC").GetValues()[frameInit:frameEnd,:].mean(axis=0)
                    RKJCvalues[i,:] = VCMJointCentre( (self.model.mp["RightKneeWidth"]+markerDiameter )/2.0 ,RKNEvalues[i,:],RHJC,RKAX,beta= 0.0 )
                else:
                    RKJCvalues[i,:] = RKNEvalues[i,:] + RKAXO * (self.model.mp["RightKneeWidth"]+markerDiameter )/2.0

                beta = 0
                ajcDesc = "KAD"
                if "RightTibialTorsion" in self.model.mp and self.model.mp["RightTibialTorsion"] !=0:
                    beta = self.model.mp["RightTibialTorsion"]
                    ajcDesc = "KAD-manualTT"

                # locate AJC
                RANK = self.acq.GetPoint("RANK").GetValues()[i,:]
                RAJCvalues[i,:] = VCMJointCentre( (self.model.mp["RightAnkleWidth"]+markerDiameter )/2.0 ,RANK,RKJCvalues[i,:],RKAX,beta= beta )

            tf_prox = self.model.getSegment("Right Thigh").getReferential("TF")
            tf_dist = self.model.getSegment("Right Shank").getReferential("TF")
            tf_dist2 = self.model.getSegment("Right Foot").getReferential("TF")
            # add nodes to referential

            # nodes
            tf_prox.static.addNode("RKNE_kad",RKNEvalues.mean(axis=0), positionType="Global", desc = "KAD" )
            tf_prox.static.addNode("RKJC_kad",RKJCvalues.mean(axis=0), positionType="Global", desc = "KAD" )
            tf_dist.static.addNode("RKJC_kad",RKJCvalues.mean(axis=0), positionType="Global", desc = "KAD")

            tf_prox.static.addNode("RKJC",RKJCvalues.mean(axis=0), positionType="Global", desc = "KAD" )
            tf_dist.static.addNode("RKJC",RKJCvalues.mean(axis=0), positionType="Global", desc = "KAD")



            # add nodes to referential
            tf_dist.static.addNode("RAJC_kad",RAJCvalues.mean(axis=0), positionType="Global", desc = ajcDesc)
            tf_dist.static.addNode("RAJC",RAJCvalues.mean(axis=0), positionType="Global", desc = ajcDesc)

            tf_dist2.static.addNode("RAJC_kad",RAJCvalues.mean(axis=0), positionType="Global", desc = ajcDesc)
            tf_dist2.static.addNode("RAJC",RAJCvalues.mean(axis=0), positionType="Global", desc = ajcDesc)

        # add KNE markers to static c3d
        if side == "both" or side == "left":
            btkTools.smartAppendPoint(self.acq,"LKNE",LKNEvalues, desc="KAD")
            btkTools.smartAppendPoint(self.acq,"LKJC_KAD",LKJCvalues, desc="KAD")
            btkTools.smartAppendPoint(self.acq,"LAJC_KAD",LAJCvalues, desc="KAD")

        if side == "both" or side == "right":
            btkTools.smartAppendPoint(self.acq,"RKNE",RKNEvalues, desc="KAD") # KNE updated.
            btkTools.smartAppendPoint(self.acq,"RKJC_KAD",RKJCvalues, desc="KAD")
            btkTools.smartAppendPoint(self.acq,"RAJC_KAD",RAJCvalues, desc="KAD")


        #btkTools.smartWriter(self.acq, "tmp-static-KAD.c3d")


class Cgm1ManualOffsets(DecoratorModel):
    """ Replicate behaviour of the CGM1 if segmental offset manaully altered

    Args
      iModel (CGM2.cgm.CGM): a CGM instance

    """
    def __init__(self, iModel):
        super(Cgm1ManualOffsets,self).__init__(iModel)


    def compute(self,acq,side,thighoffset,markerDiameter,tibialTorsion,shankoffset):
        """ run the processing

        Args
            acq (btk.Acquisition): an aquisition instance of a static c3d
            side (str): body side
            thighoffset (double): thigh offset
            markerDiameter (double): diameter of marker
            shankoffset (double): shank offset
            tibialTorsion (double): tibial torsion
        """


        self.model.decoratedModel = True

        ff = acq.GetFirstFrame()
        frameInit = acq.GetFirstFrame()-ff
        frameEnd = acq.GetLastFrame()-ff+1

        if side == "left":

            # zeroing of shankRotation if non-zero
            if shankoffset!=0:
                if isinstance(self.model,pyCGM2.Model.CGM2.cgm.CGM):
                    if "LeftShankRotation" in self.model.mp :
                        self.model.mp["LeftShankRotation"] = 0
                        LOGGER.logger.debug("Special CGM1 case - shank offset cancelled")

            # location of KJC and AJC depends on thighRotation and tibial torsion
            HJC = acq.GetPoint("LHJC").GetValues()[frameInit:frameEnd,:].mean(axis=0)
            KNE = acq.GetPoint("LKNE").GetValues()[frameInit:frameEnd,:].mean(axis=0)
            THI = acq.GetPoint("LTHI").GetValues()[frameInit:frameEnd,:].mean(axis=0)

            KJC = VCMJointCentre((self.model.mp["LeftKneeWidth"]+markerDiameter )/2.0 ,KNE,HJC,THI, beta= -1*thighoffset )


            # locate AJC
            ANK = acq.GetPoint("LANK").GetValues()[frameInit:frameEnd,:].mean(axis=0)

            if thighoffset !=0 :
                ajcDesc = "manual ThighOffset" if tibialTorsion == 0 else "manualTHIoffset-manualTT"
                AJC = VCMJointCentre( (self.model.mp["LeftAnkleWidth"]+markerDiameter )/2.0 ,ANK,KJC,KNE,beta= -1.*tibialTorsion )

            else:
                TIB = acq.GetPoint("LTIB").GetValues()[frameInit:frameEnd,:].mean(axis=0)
                AJC = VCMJointCentre( (self.model.mp["LeftAnkleWidth"]+markerDiameter )/2.0 ,ANK,KJC,TIB,beta= 0 )
                ajcDesc = ""

            # add nodes to referential
            # create and add nodes to the technical referential
            tf_prox = self.model.getSegment("Left Thigh").getReferential("TF")
            tf_dist = self.model.getSegment("Left Shank").getReferential("TF")
            tf_dist2 = self.model.getSegment("Left Foot").getReferential("TF")
            # add nodes to referential

            # nodes
            tf_prox.static.addNode("LKJC_mto",KJC, positionType="Global", desc = "manual ThighOffset" )
            tf_dist.static.addNode("LKJC_mto",KJC, positionType="Global", desc = "manual ThighOffset")

            tf_prox.static.addNode("LKJC",KJC, positionType="Global", desc = "manual ThighOffset" )
            tf_dist.static.addNode("LKJC",KJC, positionType="Global", desc = "manual ThighOffset")


            # add nodes to referential
            tf_dist.static.addNode("LAJC_mto",AJC, positionType="Global", desc = ajcDesc)
            tf_dist.static.addNode("LAJC",AJC, positionType="Global", desc = ajcDesc)

            tf_dist2.static.addNode("LAJC_mto",AJC, positionType="Global", desc = ajcDesc)
            tf_dist2.static.addNode("LAJC",AJC, positionType="Global", desc = ajcDesc)

            # enable tibialTorsion flag
            if thighoffset !=0 and tibialTorsion !=0:
                self.model.m_useLeftTibialTorsion=True


        if side == "right":

            # zeroing of shankRotation if non-zero
            if shankoffset!=0:
                if isinstance(self.model,pyCGM2.Model.CGM2.cgm.CGM):
                    if "RightShankRotation" in self.model.mp :
                        self.model.mp["RightShankRotation"] = 0
                        LOGGER.logger.debug("Special CGM1 case - shank offset cancelled")

            # location of KJC and AJC depends on thighRotation and tibial torsion
            HJC = acq.GetPoint("RHJC").GetValues()[frameInit:frameEnd,:].mean(axis=0)
            KNE = acq.GetPoint("RKNE").GetValues()[frameInit:frameEnd,:].mean(axis=0)
            THI = acq.GetPoint("RTHI").GetValues()[frameInit:frameEnd,:].mean(axis=0)

            KJC = VCMJointCentre((self.model.mp["RightKneeWidth"]+markerDiameter )/2.0 ,KNE,HJC,THI, beta= thighoffset )

            # locate AJC
            ANK = acq.GetPoint("RANK").GetValues()[frameInit:frameEnd,:].mean(axis=0)
            if thighoffset != 0 :
                AJC = VCMJointCentre( (self.model.mp["RightAnkleWidth"]+markerDiameter )/2.0 ,ANK,KJC,KNE,beta= tibialTorsion )
                ajcDesc = "manual ThighOffset" if tibialTorsion == 0 else "manualTHIoffset-manualTT"
            else:

                TIB = acq.GetPoint("RTIB").GetValues()[frameInit:frameEnd,:].mean(axis=0)
                AJC = VCMJointCentre( (self.model.mp["RightAnkleWidth"]+markerDiameter )/2.0 ,ANK,KJC,TIB,beta= 0 )
                ajcDesc = ""


            # create and add nodes to the technical referential
            tf_prox = self.model.getSegment("Right Thigh").getReferential("TF")
            tf_dist = self.model.getSegment("Right Shank").getReferential("TF")
            tf_dist2 = self.model.getSegment("Right Foot").getReferential("TF")
            # add nodes to referential

            # nodes
            tf_prox.static.addNode("RKJC_mto",KJC, positionType="Global", desc = "manual ThighOffset" )
            tf_dist.static.addNode("RKJC_mto",KJC, positionType="Global", desc = "manual ThighOffset")

            tf_prox.static.addNode("RKJC",KJC, positionType="Global", desc = "manual ThighOffset" )
            tf_dist.static.addNode("RKJC",KJC, positionType="Global", desc = "manual ThighOffset")


            # add nodes to referential
            tf_dist.static.addNode("RAJC_mto",AJC, positionType="Global", desc = ajcDesc)
            tf_dist.static.addNode("RAJC",AJC, positionType="Global", desc = ajcDesc)

            tf_dist2.static.addNode("RAJC_mto",AJC, positionType="Global", desc = ajcDesc)
            tf_dist2.static.addNode("RAJC",AJC, positionType="Global", desc = ajcDesc)


            # enable tibialTorsion flag
            if thighoffset !=0 and tibialTorsion!=0:
                self.model.m_useRightTibialTorsion=True





class HipJointCenterDecorator(DecoratorModel):
    """ Concrete CGM decorators altering the hip joint centre

    Args:
      iModel (CGM2.cgm.CGM): a CGM instance
    """
    def __init__(self, iModel):
        super(HipJointCenterDecorator,self).__init__(iModel)

    def custom(self,position_Left=0,position_Right=0,side = "both", methodDesc="custom"):
        """ Locate hip joint centres manually

        Args:
           position_Left (np.array(3,)): position of the left hip center in the pelvis referential
           position_Right (np.array(3,)): position of the right hip center in the pelvis referential
           side (str,Optional[both]): body side
           methodDesc (str,Optional[Custom]): short description of the method

        **Note:**

          - look out the pelvis referential. It has to be similar to the cgm1 pelvis referential.

        """

        self.model.decoratedModel = True

        if side == "both" or side == "left":
            if position_Left.shape ==(3,):
                LHJC_pos = position_Left

                nodeLabel= "LHJC_"+ methodDesc
                tf_prox = self.model.getSegment("Pelvis").getReferential("TF")
                tf_dist = self.model.getSegment("Left Thigh").getReferential("TF")

                # nodes
                tf_prox.static.addNode("LHJC_"+ methodDesc ,LHJC_pos, positionType="Local", desc = methodDesc)
                tf_prox.static.addNode("LHJC",LHJC_pos, positionType="Local", desc = methodDesc)

                glob = tf_prox.static.getNode_byLabel("LHJC_"+ methodDesc).m_global
                tf_dist.static.addNode("LHJC_"+ methodDesc, glob, positionType="Global", desc = methodDesc)
                tf_dist.static.addNode("LHJC",glob, positionType="Global", desc = methodDesc)

        if side == "both" or side == "right":
            if position_Right.shape ==(3,):
                nodeLabel= "RHJC_"+ methodDesc

                RHJC_pos = position_Right

                tf_prox = self.model.getSegment("Pelvis").getReferential("TF")
                tf_dist = self.model.getSegment("Right Thigh").getReferential("TF")

                # nodes
                tf_prox.static.addNode("RHJC_"+ methodDesc ,RHJC_pos, positionType="Local", desc = methodDesc)
                tf_prox.static.addNode("RHJC",RHJC_pos, positionType="Local", desc = methodDesc)

                glob = tf_prox.static.getNode_byLabel("RHJC_"+ methodDesc).m_global
                tf_dist.static.addNode("RHJC_"+ methodDesc, glob, positionType="Global", desc = methodDesc)
                tf_dist.static.addNode("RHJC",glob, positionType="Global", desc = methodDesc)


    def harrington(self,predictors= enums.HarringtonPredictor.Native, side="both"):
        """  Use of the Harrington's regressions

        Args:
           predictors (pyCGM2.enums,Optional[enums.HarringtonPredictor.Native]): harrington's predictors to use
           side (str,Optional[both]): body side

        """

        self.model.decoratedModel = True

        LHJC_pos,RHJC_pos = harringtonRegression(self.model.mp,self.model.mp_computed,predictors)

        if side == "both" or side == "left":

            tf_prox = self.model.getSegment("Pelvis").getReferential("TF")
            tf_dist = self.model.getSegment("Left Thigh").getReferential("TF")
            # nodes
            tf_prox.static.addNode("LHJC_Harrington",LHJC_pos, positionType="Local", desc = "Harrington")
            tf_prox.static.addNode("LHJC",LHJC_pos, positionType="Local", desc = "Harrington")

            glob = tf_prox.static.getNode_byLabel("LHJC_Harrington").m_global
            tf_dist.static.addNode("LHJC_Harrington",glob, positionType="Global", desc = "Harrington")
            tf_dist.static.addNode("LHJC",glob, positionType="Global", desc = "Harrington")


        if  side == "both" or side == "right":

            tf_prox = self.model.getSegment("Pelvis").getReferential("TF")
            tf_dist = self.model.getSegment("Right Thigh").getReferential("TF")

            # nodes
            tf_prox.static.addNode("RHJC_Harrington",RHJC_pos, positionType="Local", desc = "Harrington")
            tf_prox.static.addNode("RHJC",RHJC_pos, positionType="Local", desc = "Harrington")

            glob = tf_prox.static.getNode_byLabel("RHJC_Harrington").m_global
            tf_dist.static.addNode("RHJC_Harrington",glob, positionType="Global", desc = "Harrington")
            tf_dist.static.addNode("RHJC",glob, positionType="Global", desc = "Harrington")

    def hara(self, side="both"):
        """Use of the Hara's regressions

        Args:
            side (str,Optional[both]): body side

        """

        self.model.decoratedModel = True

        LHJC_pos,RHJC_pos = haraRegression(self.model.mp,self.model.mp_computed)

        if side == "both" or side == "left":

            tf_prox = self.model.getSegment("Pelvis").getReferential("TF")
            tf_dist = self.model.getSegment("Left Thigh").getReferential("TF")
            # nodes
            tf_prox.static.addNode("LHJC_Hara",LHJC_pos, positionType="Local", desc = "Hara")
            tf_prox.static.addNode("LHJC",LHJC_pos, positionType="Local", desc = "Hara")

            glob = tf_prox.static.getNode_byLabel("LHJC_Hara").m_global
            tf_dist.static.addNode("LHJC_Hara",glob, positionType="Global", desc = "Hara")
            tf_dist.static.addNode("LHJC",glob, positionType="Global", desc = "Hara")


        if  side == "both" or side == "right":

            tf_prox = self.model.getSegment("Pelvis").getReferential("TF")
            tf_dist = self.model.getSegment("Right Thigh").getReferential("TF")

            # nodes
            tf_prox.static.addNode("RHJC_Hara",RHJC_pos, positionType="Local", desc = "Hara")
            tf_prox.static.addNode("RHJC",RHJC_pos, positionType="Local", desc = "Hara")

            glob = tf_prox.static.getNode_byLabel("RHJC_Hara").m_global
            tf_dist.static.addNode("RHJC_Hara",glob, positionType="Global", desc = "Hara")
            tf_dist.static.addNode("RHJC",glob, positionType="Global", desc = "Hara")

    def davis(self, side="both"):
        """ Use of the Davis's regressions

        Args:
           side (str,Optional[both]): body side

        """

        self.model.decoratedModel = True

        LHJC_pos,RHJC_pos = davisRegression(self.model.mp,self.model.mp_computed)

        if side == "both" or side == "left":

            tf_prox = self.model.getSegment("Pelvis").getReferential("TF")
            tf_dist = self.model.getSegment("Left Thigh").getReferential("TF")
            # nodes
            tf_prox.static.addNode("LHJC_Davis",LHJC_pos, positionType="Local", desc = "Davis")
            tf_prox.static.addNode("LHJC",LHJC_pos, positionType="Local", desc = "Davis")

            glob = tf_prox.static.getNode_byLabel("LHJC_Davis").m_global
            tf_dist.static.addNode("LHJC_Davis",glob, positionType="Global", desc = "Davis")
            tf_dist.static.addNode("LHJC",glob, positionType="Global", desc = "Davis")


        if  side == "both" or side == "right":

            tf_prox = self.model.getSegment("Pelvis").getReferential("TF")
            tf_dist = self.model.getSegment("Right Thigh").getReferential("TF")

            # nodes
            tf_prox.static.addNode("RHJC_Davis",RHJC_pos, positionType="Local", desc = "Davis")
            tf_prox.static.addNode("RHJC",RHJC_pos, positionType="Local", desc = "Davis")

            glob = tf_prox.static.getNode_byLabel("RHJC_Davis").m_global
            tf_dist.static.addNode("RHJC_Davis",glob, positionType="Global", desc = "Davis")
            tf_dist.static.addNode("RHJC",glob, positionType="Global", desc = "Davis")

    def bell(self, side="both"):
        """Use of the Bell's regressions

        Args:
            side (str,Optional[both]): body side

        """

        self.model.decoratedModel = True

        LHJC_pos,RHJC_pos = bellRegression(self.model.mp,self.model.mp_computed)

        if side == "both" or side == "left":

            tf_prox = self.model.getSegment("Pelvis").getReferential("TF")
            tf_dist = self.model.getSegment("Left Thigh").getReferential("TF")
            # nodes
            tf_prox.static.addNode("LHJC_Bell",LHJC_pos, positionType="Local", desc = "Bell")
            tf_prox.static.addNode("LHJC",LHJC_pos, positionType="Local", desc = "Bell")

            glob = tf_prox.static.getNode_byLabel("LHJC_Bell").m_global
            tf_dist.static.addNode("LHJC_Bell",glob, positionType="Global", desc = "Bell")
            tf_dist.static.addNode("LHJC",glob, positionType="Global", desc = "Bell")


        if  side == "both" or side == "right":

            tf_prox = self.model.getSegment("Pelvis").getReferential("TF")
            tf_dist = self.model.getSegment("Right Thigh").getReferential("TF")

            # nodes
            tf_prox.static.addNode("RHJC_Bell",RHJC_pos, positionType="Local", desc = "Bell")
            tf_prox.static.addNode("RHJC",RHJC_pos, positionType="Local", desc = "Bell")

            glob = tf_prox.static.getNode_byLabel("RHJC_Bell").m_global
            tf_dist.static.addNode("RHJC_Bell",glob, positionType="Global", desc = "Bell")
            tf_dist.static.addNode("RHJC",glob, positionType="Global", desc = "Bell")

    def greatTrochanterOffset(self,acq, offset = 89.0,side="both",
                    leftGreatTrochLabel="LGTR", rightGreatTrochLabel="LKNM",
                    markerDiameter = 14):

        self.model.decoratedModel = True

        if side=="both" or side=="left":

            LKNM = acq.GetPoint("LKNM").GetValues()
            LKNE = acq.GetPoint("LKNE").GetValues()
            LGTR = acq.GetPoint("LGTR").GetValues()

            LHJCvalues = VCMJointCentre ((offset+markerDiameter/2.0),LGTR,LKNM,LKNE,beta=0.0)

            tf_prox = self.model.getSegment("Pelvis").getReferential("TF")
            tf_dist = self.model.getSegment("Left Thigh").getReferential("TF")

            # nodes
            tf_dist.static.addNode("LHJC_gt",LHJCvalues.mean(axis=0), positionType="Global", desc = "from gt")
            tf_prox.static.addNode("LHJC_gt",LHJCvalues.mean(axis=0), positionType="Global", desc = "from gt")

            tf_dist.static.addNode("LHJC",LHJCvalues.mean(axis=0), positionType="Global", desc = "from gt")
            tf_prox.static.addNode("LHJC",LHJCvalues.mean(axis=0), positionType="Global", desc = "from gt")

            # nodes
            self.model.getSegment("Left Thigh").getReferential("TF").static.addNode("LHJC_gt",LHJCvalues.mean(axis=0), positionType="Global")
            self.model.getSegment("Pelvis").getReferential("TF").static.addNode("LHJC_gt",LHJCvalues.mean(axis=0), positionType="Global")

            # marker
            btkTools.smartAppendPoint(acq,"LHJC_GT",LHJCvalues, desc="GT")

        if side=="both" or side=="right":

            RKNM = acq.GetPoint("RKNM").GetValues()
            RKNE = acq.GetPoint("RKNE").GetValues()
            RGTR = acq.GetPoint("RGTR").GetValues()

            RHJCvalues = VCMJointCentre ((offset+markerDiameter/2.0),RGTR,RKNM,RKNE,beta=0.0)


            tf_prox = self.model.getSegment("Pelvis").getReferential("TF")
            tf_dist = self.model.getSegment("Right Thigh").getReferential("TF")


            # nodes
            tf_dist.static.addNode("RHJC_gt",RHJCvalues.mean(axis=0), positionType="Global", desc = "from gt")
            tf_prox.static.addNode("RHJC_gt",RHJCvalues.mean(axis=0), positionType="Global", desc = "from gt")

            tf_dist.static.addNode("RHJC",RHJCvalues.mean(axis=0), positionType="Global", desc = "from gt")
            tf_prox.static.addNode("RHJC",RHJCvalues.mean(axis=0), positionType="Global", desc = "from gt")

            # marker
            btkTools.smartAppendPoint(acq,"RHJC_GT",RHJCvalues, desc="GT")


    def fromHjcMarker(self,acq, leftHJC_label = "LHJC",rightHJC_label = "RHJC" ,side="both"):
        """HJC positioned from a virtual HJC marker trajectory computed from an other process

        Args:
            acq (btk.Acquisition): an aquisition instance with  the virtual HJC marker trajectories
            leftHJC_label (str,Optional["LHJC"]): label of the left vritual HJC marker
            rightHJC_label (str,Optional["RHJC"]): label of the right vritual HJC marker
            side (str,Optional[both]): body side
        """
        self.model.decoratedModel = True

        if side=="both" or side=="left":

            LHJCvalues = acq.GetPoint(leftHJC_label).GetValues()

            tf_prox = self.model.getSegment("Pelvis").getReferential("TF")
            tf_dist = self.model.getSegment("Left Thigh").getReferential("TF")

            # nodes
            tf_dist.static.addNode("LHJC_mrk",LHJCvalues.mean(axis=0), positionType="Global", desc = "from marker")
            tf_prox.static.addNode("LHJC_mrk",LHJCvalues.mean(axis=0), positionType="Global", desc = "from marker")

            tf_dist.static.addNode("LHJC",LHJCvalues.mean(axis=0), positionType="Global", desc = "from marker")
            tf_prox.static.addNode("LHJC",LHJCvalues.mean(axis=0), positionType="Global", desc = "from marker")

            # marker
            #btkTools.smartAppendPoint(acq,"LHJC_MRK",LHJCvalues, desc="fromMarker")

        if side=="both" or side=="right":

            RHJCvalues = acq.GetPoint(rightHJC_label).GetValues()

            tf_prox = self.model.getSegment("Pelvis").getReferential("TF")
            tf_dist = self.model.getSegment("Right Thigh").getReferential("TF")

            # nodes
            tf_dist.static.addNode("RHJC_mkr",RHJCvalues.mean(axis=0), positionType="Global", desc = "from marker")
            tf_prox.static.addNode("RHJC_mkr",RHJCvalues.mean(axis=0), positionType="Global", desc = "from marker")

            tf_dist.static.addNode("RHJC",RHJCvalues.mean(axis=0), positionType="Global", desc = "from marker")
            tf_prox.static.addNode("RHJC",RHJCvalues.mean(axis=0), positionType="Global", desc = "from marker")

            # marker
            #btkTools.smartAppendPoint(acq,"LHJC_MRK",RHJCvalues, desc="from marker")

class KneeCalibrationDecorator(DecoratorModel):
    """ Concrete cgm decorator altering the knee joint

    Args:
      iModel (CGM2.cgm.CGM): a CGM instance

    """
    def __init__(self, iModel):
        """

        """

        super(KneeCalibrationDecorator,self).__init__(iModel)

    def midCondyles_KAD(self,acq, side="both",
                    leftLateralKneeLabel="LKNE", leftMedialKneeLabel="LKNM",rightLateralKneeLabel="RKNE", rightMedialKneeLabel="RKNM",
                    markerDiameter = 14):

        """Compute Knee joint centre from mid condyles and relocate AJC like KAD process.

        Args:
            acq (btkAcquisition): a btk acquisition instance of a static c3d
            side (str,Optional[both]): body side
            leftLateralKneeLabel (str,Optional[LKNE]):  label of the left lateral knee marker
            leftMedialKneeLabel (str,Optional[LKNM]):  label of the left medial knee marker
            rightLateralKneeLabel (str,Optional[RKNE]):  label of the left lateral knee marker
            rightMedialKneeLabel (str,Optional[RKNM]):  label of the left medial knee marker
            markerDiameter (double,Optional[14]):  marker diameter
        """

        self.model.decoratedModel = True

        if side=="both" or side=="left":

            # CGM specification
            if isinstance(self.model,pyCGM2.Model.CGM2.cgm.CGM):
                # cancel shankRotation and thighRotation offset if contain a previous non-zero values
                if "LeftThighRotation" in self.model.mp : self.model.mp["LeftThighRotation"] =0

            LKJCvalues = midPoint(acq,leftLateralKneeLabel,leftMedialKneeLabel,offset=(self.model.mp["LeftKneeWidth"]+markerDiameter)/2.0)

            LKNM = acq.GetPoint(leftMedialKneeLabel).GetValues()
            LKNE = acq.GetPoint(leftLateralKneeLabel).GetValues()
            LANK = acq.GetPoint("LANK").GetValues()

            LAJCvalues = VCMJointCentre ((self.model.mp["LeftAnkleWidth"]+markerDiameter )/2.0,LANK,LKJCvalues,LKNE,beta=0.0)

            tf_prox = self.model.getSegment("Left Thigh").getReferential("TF")
            tf_dist = self.model.getSegment("Left Shank").getReferential("TF")
            tf_dist2 = self.model.getSegment("Left Foot").getReferential("TF")

            # nodes
            tf_prox.static.addNode("LKJC_mid",LKJCvalues.mean(axis=0), positionType="Global", desc = "mid" )
            tf_dist.static.addNode("LKJC_mid",LKJCvalues.mean(axis=0), positionType="Global", desc = "mid")

            tf_prox.static.addNode("LKJC",LKJCvalues.mean(axis=0), positionType="Global", desc = "mid" )
            tf_dist.static.addNode("LKJC",LKJCvalues.mean(axis=0), positionType="Global", desc = "mid")

            # marker
            btkTools.smartAppendPoint(acq,"LKJC_MID",LKJCvalues, desc="MID")
            btkTools.smartAppendPoint(acq,"LAJC_midKnee",LAJCvalues, desc="kad-like")

            # add nodes to referential
            tf_dist.static.addNode("LAJC_midKnee",LAJCvalues.mean(axis=0), positionType="Global", desc = "midKJC")
            tf_dist.static.addNode("LAJC",LAJCvalues.mean(axis=0), positionType="Global", desc = "midKJC")

            tf_dist2.static.addNode("LAJC_midKnee",LAJCvalues.mean(axis=0), positionType="Global", desc = "midKJC")
            tf_dist2.static.addNode("LAJC",LAJCvalues.mean(axis=0), positionType="Global", desc = "midKJC")


        if side=="both" or side=="right":

            # CGM specification
            if isinstance(self.model,pyCGM2.Model.CGM2.cgm.CGM):
                # cancel shankRotation and thighRotation offset if contain a previous non-zero values
                if "RightThighRotation" in self.model.mp : self.model.mp["RightThighRotation"] =0


            RKJCvalues = midPoint(acq,rightLateralKneeLabel,rightMedialKneeLabel,offset=(self.model.mp["RightKneeWidth"]+markerDiameter)/2.0)

            RKNM = acq.GetPoint(rightMedialKneeLabel).GetValues()
            RKNE = acq.GetPoint(rightLateralKneeLabel).GetValues()
            RANK = acq.GetPoint("RANK").GetValues()

            RAJCvalues = VCMJointCentre ((self.model.mp["RightAnkleWidth"]+markerDiameter )/2.0,RANK,RKJCvalues,RKNE,beta=0.0)

            tf_prox = self.model.getSegment("Right Thigh").getReferential("TF")
            tf_dist = self.model.getSegment("Right Shank").getReferential("TF")
            tf_dist2 = self.model.getSegment("Right Foot").getReferential("TF")

            # nodes
            tf_prox.static.addNode("RKJC_mid",RKJCvalues.mean(axis=0), positionType="Global", desc = "mid" )
            tf_dist.static.addNode("RKJC_mid",RKJCvalues.mean(axis=0), positionType="Global", desc = "mid")

            tf_prox.static.addNode("RKJC",RKJCvalues.mean(axis=0), positionType="Global", desc = "mid" )
            tf_dist.static.addNode("RKJC",RKJCvalues.mean(axis=0), positionType="Global", desc = "mid")

            # marker
            btkTools.smartAppendPoint(acq,"RKJC_MID",RKJCvalues, desc="MID")
            btkTools.smartAppendPoint(acq,"RAJC_midKnee",RAJCvalues, desc="kad-like")

            # add nodes to referential
            tf_dist.static.addNode("RAJC_midKnee",RAJCvalues.mean(axis=0), positionType="Global", desc = "midKJC")
            tf_dist.static.addNode("RAJC",RAJCvalues.mean(axis=0), positionType="Global", desc = "midKJC")

            tf_dist2.static.addNode("RAJC_midKnee",RAJCvalues.mean(axis=0), positionType="Global", desc = "midKJC")
            tf_dist2.static.addNode("RAJC",RAJCvalues.mean(axis=0), positionType="Global", desc = "midKJC")

    def midCondyles(self,acq, side="both",
                    leftLateralKneeLabel="LKNE", leftMedialKneeLabel="LKNM",rightLateralKneeLabel="RKNE", rightMedialKneeLabel="RKNM",
                    markerDiameter = 14,widthFromMp=True):
        """Compute Knee joint centre from mid condyles.


        Args:
            acq (btkAcquisition): a btk acquisition instance of a static c3d
            side (str,Optional[True]): body side
            leftLateralKneeLabel (str,Optional[LKNE]):  label of the left lateral knee marker
            leftMedialKneeLabel (str,Optional[LKNM]):  label of the left medial knee marker
            rightLateralKneeLabel (str,Optional[RKNE]):  label of the right lateral knee marker
            rightMedialKneeLabel (str,Optional[RKNM]):  label of the right medial knee marker
            markerDiameter (double,Optional[14]):  marker diameter
            widthFromMp (bool,Optional[True]): knee with from model anthropometric parameters

        """
        # TODO : coding exception if label doesn t find.

        self.model.decoratedModel = True

        if side=="both" or side=="left":

            # CGM specification
            if isinstance(self.model,pyCGM2.Model.CGM2.cgm.CGM):
                # cancel shankRotation and thighRotation offset if contain a previous non-zero values
                if "LeftThighRotation" in self.model.mp : self.model.mp["LeftThighRotation"] =0

            if widthFromMp:
                LKJCvalues = midPoint(acq,leftLateralKneeLabel,leftMedialKneeLabel,offset=(self.model.mp["LeftKneeWidth"]+markerDiameter)/2.0)
            else:
                LKJCvalues = midPoint(acq,leftLateralKneeLabel,leftMedialKneeLabel)


            tf_prox = self.model.getSegment("Left Thigh").getReferential("TF")
            tf_dist = self.model.getSegment("Left Shank").getReferential("TF")

            # nodes
            tf_prox.static.addNode("LKJC_mid",LKJCvalues.mean(axis=0), positionType="Global", desc = "mid" )
            tf_dist.static.addNode("LKJC_mid",LKJCvalues.mean(axis=0), positionType="Global", desc = "mid")

            tf_prox.static.addNode("LKJC",LKJCvalues.mean(axis=0), positionType="Global", desc = "mid" )
            tf_dist.static.addNode("LKJC",LKJCvalues.mean(axis=0), positionType="Global", desc = "mid")

            # marker
            btkTools.smartAppendPoint(acq,"LKJC_MID",LKJCvalues, desc="MID")



        if side=="both" or side=="right":

            # CGM specification
            if isinstance(self.model,pyCGM2.Model.CGM2.cgm.CGM):
                # cancel shankRotation and thighRotation offset if contain a previous non-zero values
                if "RightThighRotation" in self.model.mp : self.model.mp["RightThighRotation"] =0

            if widthFromMp:
                RKJCvalues = midPoint(acq,rightLateralKneeLabel,rightMedialKneeLabel,offset=(self.model.mp["RightKneeWidth"]+markerDiameter)/2.0)
            else:
                RKJCvalues = midPoint(acq,rightLateralKneeLabel,rightMedialKneeLabel)

            tf_prox = self.model.getSegment("Right Thigh").getReferential("TF")
            tf_dist = self.model.getSegment("Right Shank").getReferential("TF")

            # nodes
            tf_prox.static.addNode("RKJC_mid",RKJCvalues.mean(axis=0), positionType="Global", desc = "mid" )
            tf_dist.static.addNode("RKJC_mid",RKJCvalues.mean(axis=0), positionType="Global", desc = "mid")

            tf_prox.static.addNode("RKJC",RKJCvalues.mean(axis=0), positionType="Global", desc = "mid" )
            tf_dist.static.addNode("RKJC",RKJCvalues.mean(axis=0), positionType="Global", desc = "mid")

            # marker
            btkTools.smartAppendPoint(acq,"RKJC_MID",RKJCvalues, desc="MID")

    def sara(self,side,**kwargs):
        """Compute Knee flexion axis, relocate knee joint centre from SARA functional calibration

        Args:
            side (str): body side

        Kargs:
            indexFirstFrame (int): start frame
            indexLastFrame (int): last frame

        """
        self.model.decoratedModel = True

        iff = kwargs["indexFirstFrame"] if "indexFirstFrame" in kwargs else None
        ilf = kwargs["indexLastFrame"] if "indexLastFrame" in kwargs else None

        if side == "Left":
            proxSegmentLabel = "Left Thigh"
            distSegmentlabel = "Left Shank"
            HJClabel = "LHJC"
            KJClabel = "LKJC"
            KNElabel = "LKNE"
            sequence = "ZXiY"
        elif side == "Right":
            proxSegmentLabel = "Right Thigh"
            distSegmentlabel = "Right Shank"
            HJClabel = "RHJC"
            KJClabel = "RKJC"
            KNElabel = "RKNE"
            sequence = "ZXY"
        else:
            raise Exception("[pyCGM2] side doesn t recongnize")



        proxMotion = self.model.getSegment(proxSegmentLabel).getReferential("TF").motion
        distMotion = self.model.getSegment(distSegmentlabel).getReferential("TF").motion

        # -- main function -----
        prox_ori,prox_axisLim,dist_ori,dist_axisLim,axis_prox,axis_dist,quality = saraCalibration(proxMotion,distMotion,iff, ilf, method="2")
        # end function -----


        # add nodes in TF
        self.model.getSegment(proxSegmentLabel).getReferential("TF").static.addNode("KneeFlexionOri",prox_ori,positionType="Local")
        self.model.getSegment(proxSegmentLabel).getReferential("TF").static.addNode("KneeFlexionAxis",prox_axisLim,positionType="Local")

        self.model.getSegment(distSegmentlabel).getReferential("TF").static.addNode("KneeFlexionOri",dist_ori,positionType="Local")
        self.model.getSegment(distSegmentlabel).getReferential("TF").static.addNode("KneeFlexionAxis",dist_axisLim,positionType="Local")

        # compute error
        xp = self.model.getSegment(proxSegmentLabel).getReferential("TF").getNodeTrajectory("KneeFlexionOri")
        xd = self.model.getSegment(distSegmentlabel).getReferential("TF").getNodeTrajectory("KneeFlexionOri")

        ferr = xp-xd
        Merr = numeric.rms(ferr)
        LOGGER.logger.debug( " sara rms error : %s " % str(Merr))

        # --- registration of the Knee center ---

        # longitudinal axis of the femur
        HJC = self.model.getSegment(proxSegmentLabel).getReferential("TF").static.getNode_byLabel(HJClabel).m_global
        KJC0 = self.model.getSegment(proxSegmentLabel).getReferential("TF").static.getNode_byLabel(KJClabel).m_global

        # middle of origin in Global
        p_proxOri = self.model.getSegment(proxSegmentLabel).getReferential("TF").static.getNode_byLabel("KneeFlexionOri").m_global
        p_distOri = self.model.getSegment(distSegmentlabel).getReferential("TF").static.getNode_byLabel("KneeFlexionOri").m_global
        meanOri = np.mean((p_distOri,p_proxOri),axis= 0)

        # axis lim
        p_proxAxis = self.model.getSegment(proxSegmentLabel).getReferential("TF").static.getNode_byLabel("KneeFlexionAxis").m_global
        p_distAxis = self.model.getSegment(distSegmentlabel).getReferential("TF").static.getNode_byLabel("KneeFlexionAxis").m_global
        meanAxis = np.mean((p_proxAxis,p_distAxis),axis= 0)

        # intersection beetween midcenter-axis and logitudinal axis
        proxIntersect,pb1 = geometry.LineLineIntersect(p_proxOri,p_proxAxis,HJC,KJC0)
        distIntersect,pb2 = geometry.LineLineIntersect(p_distOri,p_distAxis,HJC,KJC0)#


        # shortest distance
        shortestDistance_prox =  np.linalg.norm(proxIntersect-pb1)
        LOGGER.logger.debug(" 3d line intersect : shortest distance beetween logidudinal axis and flexion axis in Proximal  : %s  " % str(shortestDistance_prox))
        shortestDistance_dist =  np.linalg.norm(distIntersect-pb2)
        LOGGER.logger.debug( " 3d line intersect : shortest distance beetween logidudinal axis and flexion axis in Distal  : %s  " % str(shortestDistance_dist))

        # mean of the intersection point
        KJC = np.mean((proxIntersect,distIntersect), axis=0)

        # Node manager
        #  the node KJC_sara is added in all "Referentials.

        self.model.getSegment(proxSegmentLabel).getReferential("TF").static.addNode("KJC_Sara",KJC, positionType="Global", desc = "SARA")
        self.model.getSegment(distSegmentlabel).getReferential("TF").static.addNode("KJC_Sara",KJC, positionType="Global", desc = "SARA")

        self.model.getSegment(proxSegmentLabel).getReferential("TF").static.addNode("KJC_SaraAxis",meanAxis, positionType="Global", desc = "SARA")
        self.model.getSegment(distSegmentlabel).getReferential("TF").static.addNode("KJC_SaraAxis",meanAxis, positionType="Global", desc = "SARA")


         # Comparison of local position of KJCs
        localKJC = self.model.getSegment(proxSegmentLabel).getReferential("TF").static.getNode_byLabel(KJClabel).m_local
        saraKJC = self.model.getSegment(proxSegmentLabel).getReferential("TF").static.getNode_byLabel("KJC_Sara").m_local

        LOGGER.logger.debug(" former KJC position in the proximal segment : [ %f, %f,%f]   " % (localKJC[0],localKJC[1],localKJC[2]))
        LOGGER.logger.debug(" new KJC position in the proximal segment : [ %f, %f,%f]   " % (saraKJC[0],saraKJC[1],saraKJC[2]))

        # update KJC node
        self.model.getSegment(proxSegmentLabel).getReferential("TF").static.addNode(KJClabel,KJC, positionType="Global", desc = "SARA")
        self.model.getSegment(distSegmentlabel).getReferential("TF").static.addNode(KJClabel,KJC, positionType="Global", desc = "SARA")

        # thight rotation offset
        KNE = self.model.getSegment(proxSegmentLabel).getReferential("TF").static.getNode_byLabel(KNElabel).m_global

        # --- Construction of the anatomical Referential

        a1=(HJC-KJC)
        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

        v=(KNE-KJC)
        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

        a2=np.cross(a1,v)
        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

        x,y,z,R=frame.setFrameData(a1,a2,sequence)

        # projection of saraAxis
        y_sara = (meanAxis-KJC)
        y_sara = np.nan_to_num(np.divide(y_sara , np.linalg.norm(y_sara)))

        saraAxisLocal =    np.dot(R.T,y_sara)
        proj_saraAxis = np.array([ saraAxisLocal[0],
                                   saraAxisLocal[1],
                                 0])

        v_saraAxis = np.nan_to_num(np.divide(proj_saraAxis,np.linalg.norm(proj_saraAxis)))

        angle=np.rad2deg(geometry.angleFrom2Vectors(np.array([0,1,0]), v_saraAxis, z))

        self.model.getSegment(proxSegmentLabel).anatomicalFrame.static.addNode("proj_saraAxis",proj_saraAxis, positionType="Local")
        self.model.getSegment(distSegmentlabel).anatomicalFrame.static.addNode("proj_saraAxis",proj_saraAxis, positionType="Local")

        if angle > 90.0:
            LOGGER.logger.debug("left flexion axis point laterally")
            angle = 180-angle
        LOGGER.logger.debug(angle)

        #if np.abs(angle) > 30.0:
            #raise Exception ("[pyCGM2] : suspected left functional knee flexion axis. check Data")

        if side == "Left":
            self.model.setCalibrationProperty("LeftFuncKneeMethod","SARA")
            self.model.mp_computed["LeftKneeFuncCalibrationOffset"] = angle
        if side == "Right":
            self.model.setCalibrationProperty("RightFuncKneeMethod","SARA")
            self.model.mp_computed["RightKneeFuncCalibrationOffset"] = angle


    def calibrate2dof(self, side, **kwargs):
        """run the calibration2Dof method

        Args:
            side (str): body side

        Kargs:
            indexFirstFrame (int): start frame
            indexLastFrame (int): last frame
            sequence (str): Euler sequence
            jointRange (list): flexion angle boundaries
        """

        self.model.decoratedModel = True
        iff = kwargs["indexFirstFrame"] if "indexFirstFrame" in kwargs else None
        ilf = kwargs["indexLastFrame"] if "indexLastFrame" in kwargs else None
        sequence = kwargs["sequence"] if "sequence" in kwargs else None
        jointRange = kwargs["jointRange"] if "jointRange" in kwargs else None

        if side == "Left":
            proxSegmentLabel = "Left Thigh"
            distSegmentlabel = "Left Shank"
        elif side == "Right":
            proxSegmentLabel = "Right Thigh"
            distSegmentlabel = "Right Shank"
        else:
            raise Exception("[pyCGM2] side doesn t recongnize")


        proxMotion = self.model.getSegment(proxSegmentLabel).anatomicalFrame.motion
        distMotion = self.model.getSegment(distSegmentlabel).anatomicalFrame.motion

        # -- main function -----
        if sequence is None:
            longRot = calibration2Dof(proxMotion,distMotion,iff,ilf,jointRange=jointRange)
        else:
            longRot = calibration2Dof(proxMotion,distMotion,iff,ilf,jointRange,sequence=sequence)
        # end function -----

        if side == "Left":
            self.model.setCalibrationProperty("LeftFuncKneeMethod","2DOF")
            self.model.mp_computed["LeftKneeFuncCalibrationOffset"] = longRot
        if side == "Right":
            self.model.setCalibrationProperty("RightFuncKneeMethod","2DOF")
            self.model.mp_computed["RightKneeFuncCalibrationOffset"] = longRot

    def fromKjcMarker(self,acq, leftKJC_label = "LKJC",rightKJC_label = "RKJC" ,side="both"):
        """KJC positioned from a virtual KJC marker trajectory computed from an other process

        Args:
            acq (btkAcquisition): an acquisition with the virtual KJC marker trajectoriy
            leftKJC_label (str,Optional[LKJC]) : left virtual KJC label
            rightKJC_label (str,Optional[RKJC]) : right virtual KJC label
            side (str,Optional[True]): body side
        """


        self.model.decoratedModel = True

        if side=="both" or side=="left":

            LKJCvalues = acq.GetPoint(leftKJC_label).GetValues()

            tf_prox = self.model.getSegment("Left Thigh").getReferential("TF")
            tf_dist = self.model.getSegment("Left Shank").getReferential("TF")

            # nodes
            tf_dist.static.addNode("LKJC_mrk",LKJCvalues.mean(axis=0), positionType="Global", desc = "from marker")
            tf_prox.static.addNode("LKJC_mrk",LKJCvalues.mean(axis=0), positionType="Global", desc = "from marker")

            tf_dist.static.addNode("LKJC",LKJCvalues.mean(axis=0), positionType="Global", desc = "from marker")
            tf_prox.static.addNode("LKJC",LKJCvalues.mean(axis=0), positionType="Global", desc = "from marker")

            # marker
            #btkTools.smartAppendPoint(acq,"LKJC_MRK",LKJCvalues, desc="fromMarker")

        if side=="both" or side=="right":

            RKJCvalues = acq.GetPoint(rightKJC_label).GetValues()

            tf_prox = self.model.getSegment("Right Thigh").getReferential("TF")
            tf_dist = self.model.getSegment("Right Shank").getReferential("TF")

            # nodes
            tf_dist.static.addNode("RKJC_mkr",RKJCvalues.mean(axis=0), positionType="Global", desc = "from marker")
            tf_prox.static.addNode("RKJC_mkr",RKJCvalues.mean(axis=0), positionType="Global", desc = "from marker")

            tf_dist.static.addNode("RKJC",RKJCvalues.mean(axis=0), positionType="Global", desc = "from marker")
            tf_prox.static.addNode("RKJC",RKJCvalues.mean(axis=0), positionType="Global", desc = "from marker")

            # marker
            #btkTools.smartAppendPoint(acq,"RHJC_MRK",RKJCvalues, desc="from marker")

class AnkleCalibrationDecorator(DecoratorModel):
    """Concrete cgm decorator altering the ankle joint

    Args:
      iModel (CGM2.cgm.CGM): a CGM instance
    """

    def __init__(self, iModel):
        super(AnkleCalibrationDecorator,self).__init__(iModel)

    def midMaleolus(self,acq, side="both",
                    leftLateralAnkleLabel="LANK", leftMedialAnkleLabel="LMED",
                    rightLateralAnkleLabel="RANK", rightMedialAnkleLabel="RMED", markerDiameter= 14,widthFromMp=True):

        """Compute the ankle joint centre from mid maleolus.


        Args:
            acq (btkAcquisition): a btk acquisition instance of a static c3d
            side (str,Optional[True]): body side
            leftLateralAnkleLabel (str,Optional[LANK]):  label of the left lateral ankle marker
            leftMedialAnkleLabel (str,Optional[LMED]):  label of the left medial ankle marker
            rightLateralAnkleLabel (str,Optional[RANK]):  label of the right lateral ankle marker
            rightMedialAnkleLabel (str,Optional[RMED]):  label of the right medial ankle marker
            markerDiameter (double,Optional[14]):  marker diameter
            widthFromMp (bool,Optional[True]): knee with from model anthropometric parameters

        """


        #self.model.nativeCgm1 = False
        self.model.decoratedModel = True


        if side=="both" or side=="left":

            # CGM specification
            if isinstance(self.model,pyCGM2.Model.CGM2.cgm.CGM):
                self.model.m_useLeftTibialTorsion=True
                if "LeftTibialTorsion" in self.model.mp : self.model.mp["LeftTibialTorsion"] =0

            if widthFromMp:
                LAJCvalues = midPoint(acq,leftLateralAnkleLabel,leftMedialAnkleLabel,offset=(self.model.mp["LeftAnkleWidth"]+markerDiameter)/2.0)
            else:
                LAJCvalues = midPoint(acq,leftLateralAnkleLabel,leftMedialAnkleLabel)

            tf_prox = self.model.getSegment("Left Shank").getReferential("TF")
            tf_dist = self.model.getSegment("Left Foot").getReferential("TF")

            # nodes
            tf_prox.static.addNode("LAJC_mid",LAJCvalues.mean(axis=0), positionType="Global", desc = "mid" )
            tf_dist.static.addNode("LAJC_mid",LAJCvalues.mean(axis=0), positionType="Global", desc = "mid")

            tf_prox.static.addNode("LAJC",LAJCvalues.mean(axis=0), positionType="Global", desc = "mid" )
            tf_dist.static.addNode("LAJC",LAJCvalues.mean(axis=0), positionType="Global", desc = "mid")


            btkTools.smartAppendPoint(acq,"LAJC_MID",LAJCvalues, desc="MID")

        if side=="both" or side=="right":

            # CGM specification
            if isinstance(self.model,pyCGM2.Model.CGM2.cgm.CGM):
                self.model.m_useRightTibialTorsion=True
                if "RightTibialTorsion" in self.model.mp : self.model.mp["RightTibialTorsion"] =0

            if widthFromMp:
                RAJCvalues = midPoint(acq,rightLateralAnkleLabel,rightMedialAnkleLabel,offset=(self.model.mp["RightAnkleWidth"]+markerDiameter)/2.0)
            else:
                RAJCvalues = midPoint(acq,rightLateralAnkleLabel,rightMedialAnkleLabel)

            tf_prox = self.model.getSegment("Right Shank").getReferential("TF")
            tf_dist = self.model.getSegment("Right Foot").getReferential("TF")

            # nodes
            tf_prox.static.addNode("RAJC_mid",RAJCvalues.mean(axis=0), positionType="Global", desc = "mid" )
            tf_dist.static.addNode("RAJC_mid",RAJCvalues.mean(axis=0), positionType="Global", desc = "mid")

            tf_prox.static.addNode("RAJC",RAJCvalues.mean(axis=0), positionType="Global", desc = "mid" )
            tf_dist.static.addNode("RAJC",RAJCvalues.mean(axis=0), positionType="Global", desc = "mid")

            btkTools.smartAppendPoint(acq,"RAJC_MID",RAJCvalues, desc="MID")

    def fromAjcMarker(self,acq, leftAJC_label = "LAJC",rightAJC_label = "RAJC" ,side="both"):
        """AJC positioned from  virtual AJC marker trajectory computed from an other process

        Args:
            acq (btkAcquisition): an acquisition with the virtual AJC marker trajectory
            leftAJC_label (str,Optional[LAJC]) : left virtual AJC label
            rightAJC_label (str,Optional[RAJC]) : right virtual AJC label
            side (str,Optional[True]): body side
        """


        self.model.decoratedModel = True

        if side=="both" or side=="left":

            LAJCvalues = acq.GetPoint(leftAJC_label).GetValues()

            tf_prox = self.model.getSegment("Left Shank").getReferential("TF")
            tf_dist = self.model.getSegment("Left Foot").getReferential("TF")

            # nodes
            tf_dist.static.addNode("LAJC_mrk",LAJCvalues.mean(axis=0), positionType="Global", desc = "from marker")
            tf_prox.static.addNode("LAJC_mrk",LAJCvalues.mean(axis=0), positionType="Global", desc = "from marker")

            tf_dist.static.addNode("LAJC",LAJCvalues.mean(axis=0), positionType="Global", desc = "from marker")
            tf_prox.static.addNode("LAJC",LAJCvalues.mean(axis=0), positionType="Global", desc = "from marker")

            # marker
            #btkTools.smartAppendPoint(acq,"LAJC_MRK",LAJCvalues, desc="fromMarker")

        if side=="both" or side=="right":

            RAJCvalues = acq.GetPoint(rightAJC_label).GetValues()

            tf_prox = self.model.getSegment("Right Shank").getReferential("TF")
            tf_dist = self.model.getSegment("Right Foot").getReferential("TF")

            # nodes
            tf_dist.static.addNode("RAJC_mkr",RAJCvalues.mean(axis=0), positionType="Global", desc = "from marker")
            tf_prox.static.addNode("RAJC_mkr",RAJCvalues.mean(axis=0), positionType="Global", desc = "from marker")

            tf_dist.static.addNode("RAJC",RAJCvalues.mean(axis=0), positionType="Global", desc = "from marker")
            tf_prox.static.addNode("RAJC",RAJCvalues.mean(axis=0), positionType="Global", desc = "from marker")

            # marker
            #btkTools.smartAppendPoint(acq,"RHJC_MRK",RAJCvalues, desc="from marker")



# class FootCalibrationDecorator(DecoratorModel):
#     """
#         Concrete cgm decorator altering the ankle joint
#     """
#     def __init__(self, iModel):
#         """
#             :Parameters:
#               - `iModel` (CGM2.cgm.CGM) - a CGM instance
#         """
#         super(AnkleCalibrationDecorator,self).__init__(iModel)
#
#     def footJointCentreFromForeFoot(self,acq, side="both"):
#
#         """
#         """
#
#
#         #self.model.nativeCgm1 = False
#         self.model.decoratedModel = True
