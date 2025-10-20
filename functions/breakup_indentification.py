import numpy as np

def getEachBreakupIdx(breakup_idx):
    '''
    Given an boolean array of when the breakup threshold has been met return each breakup event. 
    A breakup event is defined as when continued breakup occurs.

    Input: Boolean array
    Output: Stacked boolean array of each breakup event
    '''

    int_arr = breakup_idx.astype(int)
    # Add an artificial value from the start
    int_arr = np.insert(int_arr, 0, 0)
    diff = np.diff(int_arr)
    transition_indexes = np.where(diff == 1)[0]
    transition_indexes_end = np.where(diff == -1)[0] + 1
    
    # Add a starting point if the first element in True
    # if transition_indexes[0]:
    #     transition_indexes = np.insert(transition_indexes, 0, 0)
    
    breakup_idx_stacked = []
    
    # Stack these breakup events
    for i in np.arange(0, len(transition_indexes), 1):
        new_array = np.full(breakup_idx.shape, False)
        if i >= len(transition_indexes_end):
            new_array[transition_indexes[i]:] = breakup_idx[transition_indexes[i]:]
            # print( breakup_idx[transition_indexes[i]])
        else:
            new_array[transition_indexes[i]:transition_indexes_end[i]] = breakup_idx[transition_indexes[i]:transition_indexes_end[i]]
            # print(breakup_idx[transition_indexes[i]:transition_indexes_end[i]+1])
        breakup_idx_stacked.append(new_array)
    
    # Check to see if there are any errors
    if ~(np.sum(breakup_idx_stacked, axis=0).astype(bool) == breakup_idx).all():
        print("Error: Breakup indexes not retained!")
        print(np.sum(breakup_idx_stacked, axis=0).astype(bool))
        print(breakup_idx)

    return breakup_idx_stacked

def getMajorEvents(breakup_events_array, time_threshold=6):
    # Given a stacked array of all breakup events return those that persist for longer than time_threshold:
    idx_bool = np.sum(breakup_events_array, axis=1) >= time_threshold
    idx = np.where(idx_bool)[0]

    major_events_stacked = []
    for i in idx:
        major_events_stacked.append(breakup_events_array[i])
    return major_events_stacked