from dltk.io.augmentation import *
import random
import SimpleITK as sitk
from dltk.io.preprocessing import *
import numpy as np
import os
import tensorflow as tf
import math
import pickle

def get_image_paths(): 
    negative_images_path = "./dataset_dir/"
    positive_images_path = "./dataset_dir/10/"

    neg = np.arange(1,10) # dirs that contain neg images
    negative_images = []
    negative_images_w = []

    # only loop dirs 1/ to 9/ since 9/ and 10/ contain anchor and positive
    for dir_ in neg: 
        for i, f in enumerate(os.listdir(negative_images_path +str(dir_) + '/')):
            negative_images.append(negative_images_path + str(dir_) + '/' + f)
            negative_images_w.append(int(dir_))

    # shuffle negative_images and negative_images_w together
    comb_negatives = list(zip(negative_images, negative_images_w))
    random.shuffle(comb_negatives)
    negative_images,  negative_images_w = zip(*comb_negatives)
    negative_images,  negative_images_w = negative_images[:150], negative_images_w[:150]

    negative_images_w = list(negative_images_w)
    positive_images = [str(positive_images_path + f) for f in os.listdir(positive_images_path)]
    anchor_images = positive_images[len(positive_images)-150:]
    positive_images = positive_images[:150]
    return negative_images, negative_images_w, anchor_images, positive_images

def to_numpy(img):
    return np.asanyarray(img.dataobj)

def load_and_preprocess(img):
    # Read the .nii image containing the volume with SimpleITK:
    t2 = sitk.ReadImage(img)
    t2 = resample_img(t2)
    # and access the numpy array:
    t2 = sitk.GetArrayFromImage(t2)
    t2 = normalise_zero_one(t2)
    return t2
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def serialize_example(img): # , label
    # convert to numpy array and preprocess
    preprocessed = load_and_preprocess(img)
    
    # Notice how the numpy array is converted into a byte_string. That's serialization in memory!
    feature = {
        # 'label': _float_feature(label),
        'image_raw': _bytes_feature(tf.io.serialize_tensor(preprocessed).numpy())
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0]):
    
    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    out_size = [192, 192, 160]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def write_records(niis, labels, n_per_record, outfile):
    n_niis = len(niis)
    n_records = math.ceil(len(niis) / n_per_record)

    for i, shard in enumerate(range(0, n_niis, n_per_record)):
        print(f"writing record {i} of {n_records-1}")
        with tf.io.TFRecordWriter(
                f"{outfile}_{i:0>3}-of-{n_records-1:0>3}.tfrecords", 
                options= tf.io.TFRecordOptions(compression_type="GZIP")
        ) as writer:
            for nii in niis[shard:shard+n_per_record]:
                example = serialize_example(img=nii)
                writer.write(example.SerializeToString())

def main():
    # record names prefixes
    prefix = ["anchor", "positive", "negative"]
    
    negative_images, negative_images_w, anchor_images, positive_images = get_image_paths()
    
    # save weights as pickle to reload later, because shuffle in 'comb_negative' will mess up the order
    with open('./negative_images_w', 'wb') as file:
        pickle.dump(negative_images_w, file)
    
    write_records(anchor_images, [0], len(anchor_images), prefix[0])
    write_records(positive_images, [0], len(positive_images), prefix[1])
    write_records(negative_images, [0], len(negative_images), prefix[2])
    
if __name__ == "__main__":
    main()