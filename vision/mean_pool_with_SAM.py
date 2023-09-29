from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import torch
import torch.multiprocessing as mp

import cv2

import os
import time

from utils import ImageNet16Class, mean_pool

def sam_process(sam, in_queue, out_queue, rank):
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
            out_queue.put(None)
            print("Process", rank, "exiting")
            break
        (image, filename) = inputs
        anns = mask_generator.generate(image)        
        pooled_image = mean_pool(image, anns)        
        print("Process", rank, "finished processing", filename)
        out_queue.put((pooled_image, filename))

def main():
    output_dir = "MeanPooledImageNet"
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    
    # Check number of GPUs and spawn one process per GPU
    num_workers = torch.cuda.device_count()
    print("Loaded Sam! Distributing work over", num_workers, "GPUs")
    
    st = time.time()
    
    in_queue = mp.Queue()
    out_queue = mp.Queue()
    processes = []
    
    for rank in range(num_workers):
        p = mp.Process(target=sam_process, args=(sam, in_queue, out_queue, rank))
        p.start()
        processes.append(p)
    
    # Use downsampled 16 Class ImageNet
    imnet_16class = ImageNet16Class("ImageNet", "16-class-ImageNet/downsampled_1000")

    print(f"Starting conversion of {len(imnet_16class)} images to mean-pooled images")
    for i in range(len(imnet_16class)):
        image, _ = imnet_16class[i]
        filename = imnet_16class.samples[i].split("/")[-1]
        in_queue.put((image, filename))
    # Give each queue a sentinel value
    for _ in range(num_workers):
        in_queue.put(None)
        
    # Process outputs
    parent_folder = os.path.join(output_dir, "ILSVRC/Data/CLS-LOC/train")
    os.makedirs(parent_folder, exist_ok=True)
    num_finished_workers = 0
    while True:
        outputs = out_queue.get()
        if outputs is None:
            num_finished_workers += 1
            print("Detected process exit, num_finished_workers =", num_finished_workers)
            if num_finished_workers == num_workers:
                break
            else:
                continue
        (pooled_image, filename) = outputs
        image_synset = filename.split("_")[0]

        # Check if the folder for this synset exists, if not create one
        if not os.path.exists(os.path.join(parent_folder, image_synset)):
            os.mkdir(os.path.join(parent_folder, image_synset))
        
        # Write image
        cv2.imwrite(os.path.join(parent_folder, image_synset, filename), cv2.cvtColor(pooled_image, cv2.COLOR_RGB2BGR))
    
    # Cleanup
    for p in processes:
        p.join()  

    et = time.time()
    print("Finished! Time taken:", et - st)
    
if __name__ == "__main__":
    main()