from data_preparation_utils import (
    get_imgs_directory,
    resize_and_save_img_in_directory,
    get_datasets,
    get_img_num,
)


target_size = (256, 256)
burst_image_dir, nburst_image_dir = get_imgs_directory()

resize_and_save_img_in_directory(burst_image_dir, target_size)
resize_and_save_img_in_directory(nburst_image_dir, target_size)
