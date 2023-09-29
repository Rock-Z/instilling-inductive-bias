from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import torch
import torch.multiprocessing as mp

import cv2

import os
import time

from utils import ImageNet16Class, mean_pool, ImageNetKaggle

def sam_process(sam, in_queue, rank):
    try:
        device = torch.device(f"cuda:{rank}")
    except:
        raise IndexError(f"Rank {rank} is out of range of available GPUs")
    
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam, 
                                            points_per_side = 16,
                                            points_per_batch = 64,
                                            crop_n_layers = 1,
                                            crop_n_points_downscale_factor = 2,
                                            min_mask_region_area = 100)
    
    while True:
        inputs = in_queue.get()
        if inputs is None:
            # Put sentinel value to out queue
            print("Process", rank, "exiting")
            break
        (image, filename) = inputs
        anns = mask_generator.generate(image)        
        pooled_image = mean_pool(image, anns)        
        print("Process", rank, "finished processing", filename)
        
        
        # Write image
        cv2.imwrite(os.path.join("MeanPooledEval", filename), cv2.cvtColor(pooled_image, cv2.COLOR_RGB2BGR))
        print("wrote image to", os.path.join("MeanPooledEval", filename))

def main():
    output_dir = "MeanPooledEval"
    os.mkdir(output_dir)
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    
    # Check number of GPUs and spawn one process per GPU
    num_workers = torch.cuda.device_count()
    print("Loaded Sam! Distributing work over", num_workers, "GPUs")
    
    st = time.time()
    
    in_queue = mp.Queue()
    processes = []
    
    for rank in range(num_workers):
        p = mp.Process(target=sam_process, args=(sam, in_queue, rank))
        p.start()
        processes.append(p)
    
    # Use downsampled 16 Class ImageNet
    imnet = ImageNetKaggle("ImageNet", "val")

    print(f"Starting conversion of {len(imnet)} images to mean-pooled images")
    for i in range(len(imnet)):
        image, _ = imnet[i]
        filename = imnet.samples[i].split("/")[-1]
        in_queue.put((image, filename))
    # Give each queue a sentinel value
    for _ in range(num_workers):
        in_queue.put(None)
    
    # Cleanup
    for p in processes:
        p.join()  

    et = time.time()
    print("Finished! Time taken:", et - st)
    
if __name__ == "__main__":
    main()