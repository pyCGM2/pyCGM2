

from pyCGM2.Lib.Processing import spatioTemp


def compute_spatio_temporal_parametersOperation(group, path, modelledFilenames):

    stpData = spatioTemp.computeSpatioTemporalParameters(path, modelledFilenames)

    if not group.exists_group("Preparation/SpatioTemporalParameters"):
        group.create_group("Preparation/SpatioTemporalParameters")

    group = group.retrieve_group("Preparation/SpatioTemporalParameters")

    for trialname in stpData:
            for context in stpData[trialname]:
                for parameter in stpData[trialname][context]:
                    values = stpData[trialname][context][parameter]
                    for i, value in enumerate(values):
                        ds_path = f"{trialname}/{context[0]}{parameter}/Cycle{i}"
                        if group.exists_set(ds_path):
                            group.retrieve_set(ds_path).write([value])
                        else:
                            group.create_set(ds_path, [value])

