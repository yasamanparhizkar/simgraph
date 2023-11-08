def load_grouped_spikes(spikes_dp):
    binned_data = np.load(spikes_dp)
    binned_data = binned_data.reshape(binned_data.shape[0], 1141, 113)
    binned_data = binned_data * 2 - 1     # turn labels from 0,1 to -1,1

    I_order_10 = [54, 35, 10, 60, 74, 9, 61, 56, 91, 104]

    # group all neurons together
    grouped_data = np.zeros((297, 1141, 1))
    for trial in range(297):
        for frame in range(1141):
            grouped_data[trial, frame, :] = 2 * int((binned_data[trial, frame, :] == 1).any()) - 1
    
    return grouped_data