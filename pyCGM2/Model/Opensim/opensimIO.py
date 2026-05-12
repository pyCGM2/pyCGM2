import pyCGM2; LOGGER = pyCGM2.LOGGER
import opensim


import numpy as np
import pandas as pd
pd.set_option("display.precision", 8)

class ImuStorageFile(object):
    def __init__(self,DATA_PATH, filename, freq):

        self.m_DATA_PATH = DATA_PATH
        self.m_filename = filename
        self.m_freq = freq

        # Define the outputs
        self.m_header = pd.DataFrame([f'DataRate={freq}', 
                                    f'DataType=Quaternion',
                                    f'version=3', 
                                    f'OpenSimVersion=4.4', 
                                    f'endheader'])

        self.m_data = {}


    def setData(self, imuName, quaternionArray):
        self.m_data[imuName] = quaternionArray

         

    def construct(self,static=False):
        column_header = pd.DataFrame([f'time'] + [f"{name}_imu" for name in self.m_data]).T
        
        time = [0]
        if static:
            data_output = pd.DataFrame([f"{time[0]}"] + \
                                    [ f'{self.m_data[name][0,0]},  {self.m_data[name][0,1]}, {self.m_data[name][0,2]},  {self.m_data[name][0,3]}'
                                    for name in self.m_data]).T
        else: 
            time = np.array([np.divide([range(self.m_data[name].shape[0]) for name in self.m_data], self.m_freq)[0]]).T

            data = np.array([[f'{self.m_data[name][frame,0]}, {self.m_data[name][frame,1]}, '
                            f'{self.m_data[name][frame,2]}, {self.m_data[name][frame,3]}'
                            for frame in range(self.m_data[name].shape[0])] for name in self.m_data]).T

            data_output = pd.DataFrame(np.append(time, data, axis=1))

        with open(self.m_DATA_PATH+self.m_filename, 'w') as fp:
                fp.write(self.m_header.to_csv(index=False, header=False, lineterminator='\n'))
                fp.write(column_header.to_csv(index=False, sep='\t', header=False, lineterminator='\n'))
                fp.write(data_output.to_csv(index=False, sep='\t', header=False, lineterminator='\n'))


class OpensimDataFrame(object):
    def __init__(self, DATA_PATH, filename):

        self.m_DATA_PATH = DATA_PATH
        self.m_filename = filename

        storageObject = opensim.Storage(DATA_PATH+filename)
        lastTime = storageObject.getLastTime()

        osimlabels = storageObject.getColumnLabels()

        data = {}

        self.m_header = ""
        with open(DATA_PATH+filename) as f:
            contents = f.readlines()
            for line in contents:
                if "endheader" in line:
                    break
                else:
                    self.m_header = self.m_header + line
        self.m_header = self.m_header + "endheader\n"

        for index in range(1, osimlabels.getSize()):  # 1 because 0 is time
            label = osimlabels.get(index)
            index_x = storageObject.getStateIndex(osimlabels.get(index))
            array_x = opensim.ArrayDouble()
            storageObject.getDataColumn(index_x, array_x)
            n = array_x.getSize()
            values = np.zeros((n))
            for i in range(0, n):
                values[i] = array_x.getitem(i)
            data[label] = values

        self.m_dataframe = pd.DataFrame(data)

        index_xTime = storageObject.getStateIndex("time")
        array_xTime = opensim.ArrayDouble()
        storageObject.getTimeColumn(array_xTime)
        freq = 1/(array_xTime.getitem(1)-array_xTime.getitem(0))

        timevalues = np.arange(
            0, self.m_dataframe.shape[0], 1)/freq
        
        self.m_dataframe["time"] = timevalues

        first_column = self.m_dataframe.pop('time')
        self.m_dataframe.insert(0, 'time', first_column)

    def getDataFrame(self):
        return self.m_dataframe

    def dataFrameToDict(self):

        result = {col: self.m_dataframe[col].values for col in self.m_dataframe.columns if col != "time"}
        return result

    def save(self, outDir = None, filename=None):

        directory = self.m_DATA_PATH if outDir is None else  outDir
        filename = self.m_filename if filename is None else  filename

        file1 = open(directory + filename,"w")

        for it in self.m_header:
            file1.write(it)

        columns = self.m_dataframe.columns.to_list()
        for i in range(0, len(columns)):
            if i == len(columns)-1:
                file1.write(columns[i]+"\n")
            else:
                file1.write(columns[i]+"\t")

        for j in range(0, self.m_dataframe.shape[0]):
            li = self.m_dataframe.iloc[j].to_list()
            for k in range(0, len(li)):
                if k == len(li)-1:
                    file1.write("      %.8f\n" % (li[k])) if li[k] >= 0 else file1.write(
                        "     %.8f\n" % (li[k]))
                else:
                    file1.write("      %.8f\t" % (li[k])) if li[k] >= 0 else file1.write(
                        "     %.8f\t" % (li[k]))

                    # file1.write("      "+str(li[k])+"\n")
        file1.close()




class TrcDataFrame(object):
    """
    TRC reader/writer using a pandas DataFrame as the internal representation.

    - Robust to empty fields in TRC rows (common when markers are missing).
    - Uses TAB splitting to preserve empty columns (do NOT use default split()).
    - Provides duplicateLastRow() to append a copy of the last frame.

    DataFrame columns:
        - 'Frame#'
        - 'Time'
        - for each marker M: 'M_X', 'M_Y', 'M_Z'
    """

    def __init__(self, DATA_PATH, filename):
        self.m_DATA_PATH = DATA_PATH
        self.m_filename = filename
        self.m_fullpath = DATA_PATH + filename

        # Read file as raw lines (keep exact header formatting as much as possible)
        with open(self.m_fullpath, "r", encoding="utf-8", errors="replace") as f:
            self._lines = f.readlines()

        # Locate key TRC lines
        self._idx_meta_cols = None   # line: DataRate CameraRate ...
        self._idx_meta_vals = None   # numeric values line below
        self._idx_markers = None     # line: Frame# Time <marker names ...>
        self._idx_xyz = None         # line: X1 Y1 Z1 ...
        self._idx_data0 = None       # first data line

        for i, line in enumerate(self._lines):
            s = line.strip()
            if s.startswith("DataRate") and "NumFrames" in s:
                self._idx_meta_cols = i
                self._idx_meta_vals = i + 1

            if s.startswith("Frame#") and "Time" in s:
                self._idx_markers = i
                self._idx_xyz = i + 1
                self._idx_data0 = i + 2
                break

        if self._idx_markers is None or self._idx_data0 is None:
            raise ValueError("TRC parsing failed: cannot find 'Frame# Time' section.")

        # Store header (everything before 'Frame# Time ...')
        self.m_header = "".join(self._lines[:self._idx_markers])

        # Keep meta lines split (to update NumFrames / OrigNumFrames)
        self._meta_cols = self._lines[self._idx_meta_cols].split() if self._idx_meta_cols is not None else []
        self._meta_vals = self._lines[self._idx_meta_vals].split() if self._idx_meta_vals is not None else []

        # ---- Parse marker names (TAB-based, keep empty columns then filter)
        marker_fields = self._lines[self._idx_markers].rstrip("\n").split("\t")

        # If the file uses spaces instead of tabs (rare), fallback gently
        if len(marker_fields) < 3:
            marker_fields = self._lines[self._idx_markers].split()

        if len(marker_fields) < 3 or marker_fields[0].strip() != "Frame#" or marker_fields[1].strip() != "Time":
            raise ValueError("Unexpected TRC marker line format (expected 'Frame#\\tTime\\t...').")

        # TRC marker line often has marker name then 2 empty tab fields per marker -> filter empties
        self.m_markerNames = [f.strip() for f in marker_fields[2:] if f.strip() != ""]

        if len(self.m_markerNames) == 0:
            raise ValueError("No marker names found in TRC marker line.")

        # ---- Parse data lines (TAB-based; empty fields are allowed -> NaN)
        rows = []
        for line in self._lines[self._idx_data0:]:
            if not line.strip():
                continue

            # MUST split on TAB to preserve empty columns
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                # fallback if not tab-delimited
                fields = line.split()

            expected = 2 + 3 * len(self.m_markerNames)
            if len(fields) < expected:
                raise ValueError(
                    "TRC data row has %d values; expected %d. Line starts with: %r"
                    % (len(fields), expected, line[:120])
                )

            frame = int(float(fields[0]))
            time = float(fields[1])

            coord_fields = fields[2:expected]
            coords = np.array([float(x) if x.strip() != "" else np.nan for x in coord_fields], dtype=float)

            row = {"Frame#": frame, "Time": time}
            k = 0
            for m in self.m_markerNames:
                row["%s_X" % m] = coords[k]
                row["%s_Y" % m] = coords[k + 1]
                row["%s_Z" % m] = coords[k + 2]
                k += 3
            rows.append(row)

        self.m_dataframe = pd.DataFrame(rows)

        # Ensure columns order
        cols = ["Frame#", "Time"]
        for m in self.m_markerNames:
            cols += ["%s_X" % m, "%s_Y" % m, "%s_Z" % m]
        self.m_dataframe = self.m_dataframe[cols]

        # Ensure NumFrames consistency right away (optional but useful)
        self._updateNumFrames(self.m_dataframe.shape[0])

    def getDataFrame(self):
        return self.m_dataframe

    def duplicateLastRow(self, dt=None):
        """
        Duplicate the last frame and append it.
        - Frame# increments by 1
        - Time increments by dt

        If dt is None:
            - if >=2 frames: dt = last_time - previous_time (fallback to 0.01 if <=0)
            - if 1 frame: dt = 0.01
        """
        if self.m_dataframe.shape[0] < 1:
            raise RuntimeError("DataFrame is empty. Nothing to duplicate.")

        if dt is None:
            if self.m_dataframe.shape[0] >= 2:
                dt = float(self.m_dataframe["Time"].iloc[-1] - self.m_dataframe["Time"].iloc[-2])
                if dt <= 0:
                    dt = 0.01
            else:
                dt = 0.01

        last_row = self.m_dataframe.iloc[-1].copy()
        last_row["Frame#"] = int(last_row["Frame#"]) + 1
        last_row["Time"] = float(last_row["Time"]) + float(dt)

        self.m_dataframe = pd.concat([self.m_dataframe, pd.DataFrame([last_row])], ignore_index=True)

        self._updateNumFrames(self.m_dataframe.shape[0])

    def _updateNumFrames(self, nframes):
        """
        Update NumFrames and OrigNumFrames in the metadata line if present.
        Preserves original header lines except this numeric metadata line.
        """
        if (not self._meta_cols) or (not self._meta_vals) or (self._idx_meta_vals is None):
            return

        def set_field(field, value):
            if field in self._meta_cols:
                j = self._meta_cols.index(field)
                if j < len(self._meta_vals):
                    self._meta_vals[j] = value

        set_field("NumFrames", str(nframes))
        set_field("OrigNumFrames", str(nframes))

        # Write back metadata values line with tabs (safe)
        self._lines[self._idx_meta_vals] = "\t".join(self._meta_vals) + "\n"

    def save(self, outDir=None, filename=None):
        """
        Write TRC back to disk.
        - Header preserved as much as possible (original lines up to marker line).
        - Marker names line is rewritten using the canonical TRC convention:
              Frame# <tab> Time <tab> M1 <tab><tab><tab> M2 <tab><tab><tab> ...
        - Data lines written as TAB-separated values.
        - NaN values are written as empty fields.
        """
        directory = self.m_DATA_PATH if outDir is None else outDir
        filename = self.m_filename if filename is None else filename
        outpath = directory + filename

        with open(outpath, "w", encoding="utf-8") as f:
            # Write original header up to (but excluding) marker names line
            for line in self._lines[:self._idx_markers]:
                f.write(line)

            # Marker names line: TRC convention = marker name followed by two empty columns
            f.write("Frame#\tTime\t" + "\t\t\t".join(self.m_markerNames) + "\n")

            # XYZ header line: keep original if available
            if self._idx_xyz is not None and self._idx_xyz < len(self._lines):
                f.write(self._lines[self._idx_xyz])
            else:
                # fallback rebuild
                xyz = []
                for i in range(1, len(self.m_markerNames) + 1):
                    xyz += ["X%d" % i, "Y%d" % i, "Z%d" % i]
                f.write("\t" + "\t".join(xyz) + "\n")

            # Write data lines
            for _, r in self.m_dataframe.iterrows():
                toks = [str(int(r["Frame#"])), "%.6f" % float(r["Time"])]
                for m in self.m_markerNames:
                    for ax in ("X", "Y", "Z"):
                        v = r["%s_%s" % (m, ax)]
                        toks.append("" if pd.isna(v) else "%.5f" % float(v))
                f.write("\t".join(toks) + "\n")
