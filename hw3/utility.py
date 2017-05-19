import numpy as np


def img2npy(img_path):
    import skimage
    import skimage.io
    import skimage.transform
    img = skimage.io.imread(img_path)
    img_resize = skimage.transform.resize(img, (64, 64), mode='constant')
    
    return img_resize

def read_tags(path='data/tags_clean.csv', min_count=1):
    """
    Arguments:
    path        path to tags csv file.
    min_count   integer, minimum # of posts for tags.
    
    Returns:
    tags        A dictionary, the keys are image_id and
                the values are their corresponding tags.
    """

    tags_dict = {}
    for line in open(path):
        id, all_tag = line.strip().split(',', 1)
        tag_list = [t.split(':', 1) for t in all_tag.split('\t')]
        
        tags = [tag[0].strip() for tag in filter(lambda x: int(x[1])>=min_count, tag_list)]
        tags_dict[int(id)] = tags

    return tags_dict

def read_test_texts(path):
    """
    Arguments:
    path        path to testing text file.

    Returns:
    text        A dictionary, the keys are testing_text_id
                and the values are their corresponding text.
    """

    text_dict = {}
    for line in open(path):
        id, text = line.strip().split(',', 1)
        text_dict[int(id)] = text

    return text_dict
