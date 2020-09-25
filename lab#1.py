import numpy as np

class Sorting:
    def __init__(self, arr: np.ndarray, sort: str):
        self.comparsions = np.uint64(0)
        self.swaps = np.uint64(0)
        if sort == 'B':
            self.bubble_sort(arr)
        elif sort == 'M':
            self.merge_sort(arr)
        elif sort == 'R':
            self.radix_sort(arr)

    def bubble_sort(self, arr: np.ndarray):
        #Із фіксацією останнього місця пересування
        saved_arr = arr
        
        n = len(arr)
        for i in range(n-1):
            for j in range(0, n-i-1): 
                if arr[j] > arr[j+1] : 
                    arr[[j, j+1]] = arr[[j+1, j]]
                    self.swaps += 1
                # self.comparsions += 1
        
        arr = saved_arr

    def merge_sort(self, arr: np.ndarray):
        saved_arr = arr
 
        if len(arr) > 1:
            self.comparsions += 1
            mid = len(arr)//2 # Finding the mid of the array 
            L = arr[:mid] # Dividing the array elements  
            R = arr[mid:] # into 2 halves 
    
            self.merge_sort(L) # Sorting the first half 
            self.merge_sort(R) # Sorting the second half 
    
            i = j = k = 0
            
            # Copy data to temp arrays L[] and R[] 
            while i < len(L) and j < len(R): 
                if L[i] < R[j]: 
                    arr[k] = L[i] 
                    i+= 1
                else: 
                    arr[k] = R[j] 
                    j+= 1
                self.comparsions += 1
                self.swaps += 1
                k+= 1
            
            # Checking if any element was left 
            while i < len(L): 
                self.comparsions += 1
                arr[k] = L[i] 
                i+= 1
                k+= 1
                self.swaps += 1
            
            while j < len(R): 
                self.comparsions += 1
                arr[k] = R[j] 
                j+= 1
                k+= 1
                self.swaps += 1

        arr = saved_arr

    def radix_sort(self, arr: np.ndarray):
        saved_arr = arr

        def countingSort(arr: np.ndarray, exp1: int):
            n = len(arr) 
        
            # The output array elements that will have sorted arr 
            output = np.zeros(n, dtype=int)
        
            # initialize count array as 0 
            count = np.zeros(10, dtype=int)
        
            # Store count of occurrences in count[] 
            for i in range(0, n): 
                index = arr[i] // exp1
                count[ index % 10 ] += 1
                self.swaps += 1
        
            # Change count[i] so that count[i] now contains actual 
            #  position of this digit in output array 
            for i in range(1,10): 
                count[i] += count[i-1]
                self.swaps += 1
        
            # Build the output array 
            i = n-1
            while i >= 0:
                self.comparsions += 1
                index = arr[i] // exp1
                output[ count[ index % 10 ] - 1] = arr[i]
                self.swaps += 1
                count[ index % 10 ] -= 1
                i -= 1
        
            # Copying the output array to arr[], 
            # so that arr now contains sorted numbers 
            i = 0
            for i in range(0,len(arr)): 
                arr[i] = output[i] 
        
        # Method to do Radix Sort 
        
        # Find the maximum number to know number of digits 
        max1 = np.max(arr)
    
        # Do counting sort for every digit. Note that instead 
        # of passing digit number, exp is passed. exp is 10^i 
        # where i is current digit number 
        exp = 1
        while max1//exp > 0:
            self.comparsions += 1
            countingSort(arr, exp) 
            exp *= 10

        arr = saved_arr        

if __name__ == "__main__":
    SL1 = np.arange(1000)   #sorted list 1000
    SL2 = np.arange(10000)  #sorted list 10000
    SL3 = np.arange(100000) #sorted list 100000
    
    USL1 = np.arange(1000)[::-1]    #sorted reversed list 1000
    USL2 = np.arange(10000)[::-1]   #sorted reversed list 10000
    USL3 = np.arange(100000)[::-1]  #sorted reversed list 100000

    ALs1 = np.array((np.random.rand(1000, 1000)*10000)%1000, np.int32)
    ALs2 = np.array((np.random.rand(1000, 10000)*100000)%10000, np.int32)
    ALs3 = np.array((np.random.rand(1000, 100000)*1000000)%100000, np.int32)

    # Best1000_Bubble = Sorting(SL1, 'B')
    # b1000_B_swaps = Best1000_Bubble.swaps
    # b1000_B_comps = Best1000_Bubble.comparsions

    # Worst1000_Bubble = Sorting(USL1, 'B')
    # w1000_B_swaps = Worst1000_Bubble.swaps
    # w1000_B_comps = Worst1000_Bubble.comparsions

    # Best10000_Bubble = Sorting(SL2, 'B')
    # b10000_B_swaps = Best10000_Bubble.swaps
    # b10000_B_comps = Best10000_Bubble.comparsions

    # Worst10000_Bubble = Sorting(USL2, 'B')
    # w10000_B_swaps = Worst10000_Bubble.swaps
    # w10000_B_comps = Worst10000_Bubble.comparsions

    # Best100000_Bubble = Sorting(SL3, 'B')
    # b100000_B_swaps = Best100000_Bubble.swaps
    # b100000_B_comps = Best100000_Bubble.comparsions

    # Worst100000_Bubble = Sorting(USL3, 'B')
    # w100000_B_swaps = Worst100000_Bubble.swaps
    # w100000_B_comps = Worst100000_Bubble.comparsions

    
    Best1000_Merge = Sorting(SL1, 'M')
    b1000_M_swaps = Best1000_Merge.swaps
    b1000_M_comps = Best1000_Merge.comparsions

    Worst1000_Merge = Sorting(USL1, 'M')
    w1000_M_swaps = Worst1000_Merge.swaps
    w1000_M_comps = Worst1000_Merge.comparsions

    Best10000_Merge = Sorting(SL2, 'M')
    b10000_M_swaps = Best10000_Merge.swaps
    b10000_M_comps = Best10000_Merge.comparsions

    Worst10000_Merge = Sorting(USL2, 'M')
    w10000_M_swaps = Worst10000_Merge.swaps
    w10000_M_comps = Worst10000_Merge.comparsions

    Best100000_Merge = Sorting(SL3, 'M')
    b100000_M_swaps = Best100000_Merge.swaps
    b100000_M_comps = Best100000_Merge.comparsions

    Worst100000_Merge = Sorting(USL3, 'M')
    w100000_M_swaps = Worst100000_Merge.swaps
    w100000_M_comps = Worst100000_Merge.comparsions

    USL1 = np.arange(1000)[::-1]    #sorted reversed list 1000
    USL2 = np.arange(10000)[::-1]   #sorted reversed list 10000
    USL3 = np.arange(100000)[::-1]  #sorted reversed list 100000

    Best1000_Radix = Sorting(SL1, 'R')
    b1000_R_swaps = Best1000_Radix.swaps
    b1000_R_comps = Best1000_Radix.comparsions

    Worst1000_Radix = Sorting(USL1, 'R')
    w1000_R_swaps = Worst1000_Radix.swaps
    w1000_R_comps = Worst1000_Radix.comparsions

    Best10000_Radix = Sorting(SL2, 'R')
    b10000_R_swaps = Best10000_Radix.swaps
    b10000_R_comps = Best10000_Radix.comparsions

    Worst10000_Radix = Sorting(USL2, 'R')
    w10000_R_swaps = Worst10000_Radix.swaps
    w10000_R_comps = Worst10000_Radix.comparsions

    Best100000_Radix = Sorting(SL3, 'R')
    b100000_R_swaps = Best100000_Radix.swaps
    b100000_R_comps = Best100000_Radix.comparsions

    Worst100000_Radix = Sorting(USL3, 'R')
    w100000_R_swaps = Worst100000_Radix.swaps
    w100000_R_comps = Worst100000_Radix.comparsions
    
    
    swaps_B_1000 = np.zeros(1000, int)
    # comps_B_1000 = np.zeros(1000, int)
    swaps_B_10000 = np.zeros(1000, int)
    # comps_B_10000 = np.zeros(1000, int)
    swaps_B_100000 = np.zeros(1000, int)
    # comps_B_100000 = np.zeros(1000, int)

    swaps_M_1000 = np.zeros(1000, int)
    comps_M_1000 = np.zeros(1000, int)
    swaps_M_10000 = np.zeros(1000, int)
    comps_M_10000 = np.zeros(1000, int)
    swaps_M_100000 = np.zeros(1000, int)
    comps_M_100000 = np.zeros(1000, int)

    swaps_R_1000 = np.zeros(1000, int)
    comps_R_1000 = np.zeros(1000, int)
    swaps_R_10000 = np.zeros(1000, int)
    comps_R_10000 = np.zeros(1000, int)
    swaps_R_100000 = np.zeros(1000, int)
    comps_R_100000 = np.zeros(1000, int)

    for i in range(1000):
        Avg1000_Bubble = Sorting(ALs1[i], 'B')
        swaps_B_1000[i] = Avg1000_Bubble.swaps
        # comps_B_1000[i] = Avg1000_Bubble.comparsions
        Avg10000_Bubble = Sorting(ALs2[i], 'B')
        swaps_B_10000[i] = Avg10000_Bubble.swaps
        # comps_B_10000[i] = Avg10000_Bubble.comparsions
        Avg100000_Bubble = Sorting(ALs3[i], 'B')
        swaps_B_100000[i] = Avg100000_Bubble.swaps
        # comps_B_100000[i] = Avg100000_Bubble.comparsions

        Avg1000_Merge = Sorting(ALs1[i], 'M')
        swaps_M_1000[i] = Avg1000_Merge.swaps
        comps_M_1000[i] = Avg1000_Merge.comparsions
        Avg10000_Merge = Sorting(ALs2[i], 'M')
        swaps_M_10000[i] = Avg10000_Merge.swaps
        comps_M_10000[i] = Avg10000_Merge.comparsions
        Avg100000_Merge = Sorting(ALs3[i], 'M')
        swaps_M_100000[i] = Avg100000_Merge.swaps
        comps_M_100000[i] = Avg100000_Merge.comparsions

        Avg1000_Radix = Sorting(ALs1[i], 'R')
        swaps_R_1000[i] = Avg1000_Radix.swaps
        comps_R_1000[i] = Avg1000_Radix.comparsions
        Avg10000_Radix = Sorting(ALs2[i], 'R')
        swaps_R_10000[i] = Avg10000_Radix.swaps
        comps_R_10000[i] = Avg10000_Radix.comparsions
        Avg100000_Radix = Sorting(ALs3[i], 'R')
        swaps_R_100000[i] = Avg100000_Radix.swaps
        comps_R_100000[i] = Avg100000_Radix.comparsions

    a1000_B_swaps = np.average(swaps_B_1000)
    # a1000_B_comps = np.average(comps_B_1000)
    a10000_B_swaps = np.average(swaps_B_10000)
    # a10000_B_comps = np.average(comps_B_10000)
    a100000_B_swaps = np.average(swaps_B_100000)
    # a100000_B_comps = np.average(comps_B_100000)

    a1000_M_swaps = np.average(swaps_M_1000)
    a1000_M_comps = np.average(comps_M_1000)
    a10000_M_swaps = np.average(swaps_M_10000)
    a10000_M_comps = np.average(comps_M_10000)
    a100000_M_swaps = np.average(swaps_M_100000)
    a100000_M_comps = np.average(comps_M_100000)
    
    a1000_R_swaps = np.average(swaps_R_1000)
    a1000_R_comps = np.average(comps_R_1000)
    a10000_R_swaps = np.average(swaps_R_10000)
    a10000_R_comps = np.average(comps_R_10000)
    a100000_R_swaps = np.average(swaps_R_100000)
    a100000_R_comps = np.average(comps_R_100000)

    print("/------------------------------------/")
    print("Bubble Sort")
    # print("1000  :: B -- comps =", b1000_B_comps, "\tswaps =", b1000_B_swaps)
    print("1000  :: A -- swaps =", a1000_B_swaps)
    # print("1000  :: W -- comps =", w1000_B_comps, "\tswaps =", w1000_B_swaps)
    # print("10000 :: B -- comps =", b10000_B_comps, "\tswaps =", b10000_B_swaps)
    print("10000 :: A -- swaps =", a10000_B_swaps)
    # print("10000 :: W -- comps =", w10000_B_comps, "\tswaps =", w10000_B_swaps)
    # print("100000:: B -- comps =", b100000_B_comps, "\tswaps =", b100000_B_swaps)
    print("100000:: A -- swaps=", a100000_B_swaps)
    # print("100000:: W -- comps =", w100000_B_comps, "\tswaps =", w100000_B_swaps)
    print("/------------------------------------/")
    print("Merge Sort")
    print("1000  :: B -- comps =", b1000_M_comps, "\tswaps =", b1000_M_swaps)
    print("1000  :: A -- comps =", a1000_M_comps, "\tswaps =", a1000_M_swaps)
    print("1000  :: W -- comps =", w1000_M_comps, "\tswaps =", w1000_M_swaps)
    print("10000 :: B -- comps =", b10000_M_comps, "\tswaps =", b10000_M_swaps)
    print("10000 :: A -- comps =", a10000_M_comps, "\tswaps =", a10000_M_swaps)
    print("10000 :: W -- comps =", w10000_M_comps, "\tswaps =", w10000_M_swaps)
    print("100000:: B -- comps =", b100000_M_comps, "\tswaps =", b100000_M_swaps)
    print("100000:: A -- comps =", a100000_M_comps, "\tswaps =", a100000_M_swaps)
    print("100000:: W -- comps =", w100000_M_comps, "\tswaps =", w100000_M_swaps)
    print("/------------------------------------/")
    print("Radix Sort")
    print("1000  :: B -- comps =", b1000_R_comps, "\tswaps =", b1000_R_swaps)
    print("1000  :: A -- comps =", a1000_R_comps, "\tswaps =", a1000_R_swaps)
    print("1000  :: W -- comps =", w1000_R_comps, "\tswaps =", w1000_R_swaps)
    print("10000 :: B -- comps =", b10000_R_comps, "\tswaps =", b10000_R_swaps)
    print("10000 :: A -- comps =", a10000_R_comps, "\tswaps =", a10000_R_swaps)
    print("10000 :: W -- comps =", w10000_R_comps, "\tswaps =", w10000_R_swaps)
    print("100000:: B -- comps =", b100000_R_comps, "\tswaps =", b100000_R_swaps)
    print("100000:: A -- comps =", a100000_R_comps, "\tswaps =", a100000_R_swaps)
    print("100000:: W -- comps =", w100000_R_comps, "\tswaps =", w100000_R_swaps)
    print("/------------------------------------/")