import matplotlib.pyplot as plt

def plot_history(history):
    loss_list = [s for s in history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history[l], 'b', label='Training loss (' + str(str(format(history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history[l], 'g', label='Validation loss (' + str(str(format(history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig('loss_30_with_change.png')
    if False:
        ## Accuracy
        plt.figure(2)
        for l in acc_list:
            plt.plot(epochs, history[l], 'b', label='Training accuracy (' + str(format(history[l][-1],'.5f'))+')')
        for l in val_acc_list:    
            plt.plot(epochs, history[l], 'g', label='Validation accuracy (' + str(format(history[l][-1],'.5f'))+')')

        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
    plt.show()



loss = [1.0, 0.8, 0.6, 0.5, 0.5, 0.4, 0.3, 0.25, 0.23, 0.2]
val_loss = [1.0, 0.9, 0.8, 0.7, 0.65, 0.6, 0.5, 0.45, 0.4, 0.4]

#acc = [10, 40, 50, 55, 60, 65, 70, 72, 73, 73]
#val_acc = [10, 20, 30, 40, 50, 55, 60, 63, 64, 63]

loss = [0.3134, 0.2460, 0.2428, 0.2478, 0.2385, 0.2450, 0.2436, 0.2432, 0.2427, 0.2427]
val_loss = [0.3103, 0.2468, 0.2729, 0.2793, 0.2307, 0.2444, 0.2448, 0.2457, 0.2452, 0.2433]

loss_sm = [0.4950, 0.3767, 0.3535, 0.3335, 0.3204, 0.3199, 0.3058, 0.3002, 0.2990, 0.3006]
val_loss_sm = [0.2938, 0.3582, 0.3225, 0.3116, 0.3187, 0.3094, 0.2945, 0.2995, 0.2940, 0.2885] 

#loss = [0.3894, 0.2639, 0.2515, 0.2449, 0.2448, 0.2463, 0.2442, 0.2385, 0.2414, 0.2398, 0.2375, 0.2384, 0.2376, 0.2342, 0.2293, 0.2306, 0.2400, 0.2358, 0.2357, 0.2343, 0.2342, 0.2299, 0.2327, 0.2312, 0.2250, 0.2247, 0.2421, 0.2377, 0.2352, 0.2333]
#val_loss = [0.3144, 0.2707, 0.2520, 0.2390, 0.2496, 0.2476, 0.2419, 0.2382, 0.2442, 0.2400, 0.2372, 0.2441, 0.2363, 0.2334, 0.2312, 0.2429, 0.2378, 0.2347, 0.2351, 0.2381, 0.2338, 0.2256, 0.2366, 0.2253, 0.2235, 0.2456, 0.2411, 0.2362, 0.2351, 0.2335]


#loss = [0.4695, 0.2796, 0.2698, 0.2630, 0.2574, 0.2439, 0.2408, 0.2467, 0.2439, 0.2395, 0.2359, 0.2335, 0.2325, 0.2339, 0.2312, 0.2301, 0.2323, 0.2317, 0.2306, 0.2303, 0.2281, 0.2287, 0.2276, 0.2283, 0.2277]
#val_loss = [0.3805, 0.4131, 0.2479, 0.2929, 0.2487, 0.2401, 0.2465, 0.2422, 0.2437, 0.2370, 0.2351, 0.2339, 0.2327, 0.2319, 0.2317, 0.2310, 0.2316, 0.2311, 0.2303, 0.2290, 0.2290, 0.2287, 0.2292, 0.2290, 0.2288]


#loss = [0.3483, 0.2632, 0.2453, 0.2440, 0.2364, 0.2363, 0.2319, 0.2281, 0.2422, 0.2480, 0.2340, 0.2310, 0.2286, 0.2281, 0.2275, 0.2265, 0.2250, 0.2247, 0.2237, 0.2249]
#val_loss = [0.3351, 0.2589, 0.2476, 0.2430, 0.2312, 0.2393, 0.2287, 0.2292, 0.2415, 0.2385, 0.2293, 0.2271, 0.2262, 0.2254, 0.2246, 0.2240, 0.2239, 0.2222, 0.2217, 0.2270]


history = {}
history["loss"] = loss
history["val_loss"] = val_loss
#history["acc"] = acc
#history["val_acc"] = val_acc

plot_history(history)
