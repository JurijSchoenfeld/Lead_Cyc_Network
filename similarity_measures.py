import numpy as np

# %% EVENT SYNCHRONIZATION 1D

'''event_tsi and event_tsj are boolean lists indicating 
the positions of the events.
timei and timej are lists with the time of the events 
in event_tsi and event_tsj
countij is the number of synchronized events where an event at 
j precedes an event at i by at least tau_min time units but by 
no more than \tau_max time units'''


def eventsync_1D(event_tsi, event_tsj, timei=None, timej=None, tau_min=0, tau_max=np.inf, identify_events=False):
    # Get time indices
    if timei is None:
        ei = np.where(event_tsi)[0]
    else:
        ei = timei[event_tsi]

    if timej is None:
        ej = np.where(event_tsj)[0]
    else:
        ej = timej[event_tsj]

    # print(ei, '\n',ej)
    # Number of events
    si = ei.size
    sj = ej.size

    if si == 0 or sj == 0:  # Division by zero in output
        return np.nan, np.nan
    if si in [1, 2] or sj in [1, 2]:  # Too few events to calculate
        return np.nan, np.nan

    # Calculate the time difference (ti_l − tj_m)
    dstij = (np.array([ei[1:-1]] * (sj - 2), dtype='int32').T -
             np.array([ej[1:-1]] * (si - 2), dtype='int32'))

    # Dynamical delay
    diffi = np.diff(ei)  # (ti_l+1 − ti_l, ti_l − ti_l−1)
    diffj = np.diff(ej)  # (tj_m+1 − tj_m, tj_m − tj_m−1)

    diffi_min = np.minimum(diffi[1:], diffi[:-1])  # min(ti_l+1 − ti_l, ti_l − ti_l−1)
    diffj_min = np.minimum(diffj[1:], diffj[:-1])  # min(tj_m+1 − tj_m, tj_m − tj_m−1)

    # τij_lm 
    tau = 0.5 * np.minimum(np.array([diffi_min] * (sj - 2), dtype='float32').T,
                           np.array([diffj_min] * (si - 2), dtype='float32'))
    # τij_lm = 0.5*min(ti_l+1 − ti_l, ti_l − ti_l−1, tj_m+1 − tj_m, tj_m − tj_m−1)

    eff_tau_max = np.minimum(tau, tau_max)  # τij_lm <= τ_max
    eff_tau_min = np.maximum(0, np.ones(np.shape(eff_tau_max)) * tau_min)  # τij_lm >= τ_min

    # Synchronization condition
    sigmaij = (dstij >= eff_tau_min) & (dstij <= eff_tau_max)  # σij_lm: 0 <= τ_min <= ti_l − tj_m <= τij_lm <= τ_max

    # Synchronised events with σij_lm = 1  
    countij = np.sum(sigmaij)

    if identify_events:

        if (countij > 0):
            # Event arrays
            matrix_timei = np.array([ei[1:-1]] * (sj - 2), dtype='int32').T
            matrix_timej = np.array([ej[1:-1]] * (si - 2), dtype='int32')

            # Direction ij (time): events in j (first column) precede events in i (second column)
            if countij > 0:
                directionij = np.column_stack((matrix_timej[sigmaij > 0], matrix_timei[sigmaij > 0]))
            else:
                directionij = np.nan

            return countij, directionij

    return countij, np.nan


# %% EVENT SYNCHRONIZATION 2D

'''event_tsi and event_tsj are boolean lists indicating 
the positions of the events.
timei and timej are lists with the time of the events 
in event_tsi and event_tsj'''


def eventsync(event_tsi, event_tsj, timei=None, timej=None, tau_min=0, tau_max=np.inf, identify_events=False):
    # Get time indices
    if timei is None:
        ei = np.where(event_tsi)[0]
    else:
        ei = timei[event_tsi]

    if timej is None:
        ej = np.where(event_tsj)[0]
    else:
        ej = timej[event_tsj]

    # Number of events
    si = ei.size
    sj = ej.size

    if si == 0 or sj == 0:  # Division by zero in output
        return np.nan, np.nan, np.nan, np.nan
    if si in [1, 2] or sj in [1, 2]:  # Too few events to calculate
        return np.nan, np.nan, np.nan, np.nan

    # Calculate the time difference (ti_l − tj_m)
    dstij = (np.array([ei[1:-1]] * (sj - 2), dtype='int32').T -
             np.array([ej[1:-1]] * (si - 2), dtype='int32'))

    # Dynamical delay
    diffi = np.diff(ei)  # (ti_l+1 − ti_l, ti_l − ti_l−1)
    diffj = np.diff(ej)  # (tj_m+1 − tj_m, tj_m − tj_m−1)

    diffi_min = np.minimum(diffi[1:], diffi[:-1])  # min(ti_l+1 − ti_l, ti_l − ti_l−1)
    diffj_min = np.minimum(diffj[1:], diffj[:-1])  # min(tj_m+1 − tj_m, tj_m − tj_m−1)

    # τij_lm 
    tau = 0.5 * np.minimum(np.array([diffi_min] * (sj - 2), dtype='float32').T,
                           np.array([diffj_min] * (si - 2), dtype='float32'))
    # τij_lm = 0.5*min(ti_l+1 − ti_l, ti_l − ti_l−1, tj_m+1 − tj_m, tj_m − tj_m−1)

    eff_tau_max = np.minimum(tau, tau_max)  # τij_lm <= τ_max
    eff_tau_min = np.maximum(0, np.ones(np.shape(eff_tau_max)) * tau_min)
    # note that τij_lm = τji_ml and dstji = -dstij

    # Synchronization condition
    sigmaij = (dstij >= eff_tau_min) & (dstij <= eff_tau_max)  # σij_lm: 0 <= τ_min <= ti_l − tj_m <= τij_lm <= τ_max
    sigmaji = (-dstij >= eff_tau_min) & (-dstij <= eff_tau_max)  # σji_ml: 0 <= τ_min <= tj_m − ti_l <= τji_ml <= τ_max

    # Indicator function CHECK!!!   
    sigmaji_up = np.vstack((np.array(np.zeros((1, sj - 2)), dtype='bool'), sigmaji[:-1, :]))  # σji_m,l−1
    sigmaji_right = np.hstack((sigmaji[:, 1:], np.array(np.zeros((si - 2, 1)), dtype='bool')))  # σji_m+1,l
    indicatorij = (dstij > 0 & sigmaij & ~sigmaji_up & ~sigmaji_right) + 0.5 * (dstij == 0 & sigmaij) + 0.5 * (
                dstij > 0 & sigmaij & (sigmaji_up | sigmaji_right))
    #             (ti_l-tj_m>0 & σij_lm=1 & σji_m,l−1=0 & σji_m+1,l=0),    (ti_l = tj_m & σij_lm=1), (ti_l-tj_m >0 & σij_lm=1 & (σji_m,l−1=1 | σji_m+1,l=1))

    sigmaij_down = np.vstack((sigmaij[1:, :], np.array(np.zeros((1, sj - 2)), dtype='bool')))  # σij_l,m−1
    sigmaij_left = np.hstack((np.array(np.zeros((si - 2, 1)), dtype='bool'), sigmaij[:, :-1]))  # σij_l+1,m
    indicatorji = (-dstij > 0 & sigmaji & ~sigmaij_down & ~sigmaij_left) + 0.5 * (-dstij == 0 & sigmaji) + 0.5 * (
                -dstij > 0 & sigmaji & (sigmaij_down | sigmaij_left))
    #             (tj_m-ti_l>0 & σji_ml=1 & σij_l,m−1=0 & σij_l+1,m=0),       (ti_l = tj_m & σji_ml=1), (tj_m-ti_l>0 & σji_ml=1 & (σij_l,m−1=1 | σij_l+1,m=1))

    # Synchronised events with Jij_lm > 0  
    countij = np.sum(indicatorij)
    countji = np.sum(indicatorji)

    if identify_events:

        if (countij > 0) | (countji > 0):
            # Event arrays
            matrix_timei = np.array([ei[1:-1]] * (sj - 2), dtype='int32').T
            matrix_timej = np.array([ej[1:-1]] * (si - 2), dtype='int32')

            # Direction ij (time): events in j (first column) precede events in i (second column)
            if countij > 0:
                directionij = np.column_stack((matrix_timej[sigmaij > 0], matrix_timei[sigmaij > 0]))
            else:
                directionij = np.nan

            # Direction ji (time): events in i (first column) precede events in j (second column)
            if countji > 0:
                directionji = np.column_stack((matrix_timei[sigmaji > 0], matrix_timej[sigmaji > 0]))
            else:
                directionji = np.nan

            return countij, countji, directionij, directionji  # normalized with respect to the 'total number of events'

    return countij, countji, np.nan, np.nan  # normalized with respect to the 'total number of events'
