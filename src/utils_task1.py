import tensorflow as tf
import tensorflow as tf
import xml.etree.ElementTree as ET
import os

def extract_info_from_xml_file(filename:str, anno_xml:str):
    """
    Extracts information from the XML annotation file corresponding to the given image filename.

    Args:
    filename (str): The filename of the image.
    anno_xml (str): The path to the XML annotation file.

    Returns:
    tuple: A tuple containing the extracted information, including width, height, source ID, format,
           bounding box coordinates (xmin, xmax, ymin, ymax), class text, class label, single class,
           difficulty, truncated, and view.
    """
    
    format = filename.split(".")[-1]
    #source_id = filename.split(".")[0].splite("_")[-1]         #If you want to extrat the number from the filename as source_id
        
    tree = ET.parse(anno_xml)                                   #Open XML File with annotations
    root = tree.getroot()
    
    width = int(root.find('size/width').text)
    height = int(root.find('size/width').text)
    source_id = root.find('source/database').text               #If you want to extrat the database name from the annotations as source_id
    class_single = root.find('segmented').text                  #Not quite sure if the rigth feature is extracted

    xmin, xmax, ymin, ymax, truncated, difficult, view, class_text, class_label = [], [], [], [], [], [], [], [], []
    
    for num, obj in enumerate(root.findall('object')):
        xmin.append(int(obj.find('bndbox/xmin').text))
        xmax.append(int(obj.find('bndbox/xmax').text))
        ymin.append(int(obj.find('bndbox/ymin').text))
        ymax.append(int(obj.find('bndbox/ymax').text))
        class_text.append(obj.find('name').text)                #Not quite sure if the rigth feature is extracted
        class_label.append(num)                                 #Not quite sure if the rigth feature is extracted
        truncated.append(int(obj.find('truncated').text))
        difficult.append(int(obj.find('difficult').text))
        view.append(obj.find('pose').text)                      #Not quite sure if the rigth feature is extracted
    
    return width, height, source_id, format, xmin, xmax, ymin, ymax, class_text, class_label, class_single, difficult, truncated, view

def serialize_info(base_path_folder, width, height, filename, source_id, image, format, xmin, xmax, ymin, ymax, class_text, class_label, class_single, difficult, truncated, view):
    """
    Serialize image and associated information into a TFRecord file.

    Args:
    base_path_folder (str): The base path folder to save the TFRecord file.
    width (int): The width of the image.
    height (int): The height of the image.
    filename (str): The filename of the image.
    source_id (str): The source ID of the image.
    image (numpy.ndarray): The image data.
    format (str): The format of the image file.
    xmin (list): List of minimum x-coordinate values of bounding boxes.
    xmax (list): List of maximum x-coordinate values of bounding boxes.
    ymin (list): List of minimum y-coordinate values of bounding boxes.
    ymax (list): List of maximum y-coordinate values of bounding boxes.
    class_text (list): List of class text labels.
    class_label (list): List of class labels.
    class_single (int): Single class label.
    difficult (list): List of difficulty flags.
    truncated (list): List of truncated flags.
    view (list): List of view information.

    Returns:
    bytes: Serialized TFRecord data.
    """
    
    base_path_name = filename.split(".")[0]
    with tf.io.TFRecordWriter(os.path.join(base_path_folder, f'{base_path_name}.tfrecords')) as file_writer:
        
        # Serialize image data
        serialized_image = tf.io.serialize_tensor(image)

        record_bytes = tf.train.Example(features=tf.train.Features(feature={
            "image/width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
            "image/height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
            'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf-8')])),
            'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[source_id.encode('utf-8')])),
            'image/encodedrawdata': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_image.numpy()])),
            'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[format.encode('utf-8')])),
            'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
            'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
            'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
            'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
            'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[text.encode('utf-8') for text in class_text])),
            'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=class_label)),
            'image/object/class/single': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(class_single)])),
            'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult)),
            'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
            'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=[text.encode('utf-8') for text in view])),    
        })).SerializeToString()
        
        file_writer.write(record_bytes)

def decode_info(record_bytes):
    """
    Decode TFRecord data into its features. 
    The features are defined based on the specifications provided by the CV_challenge.

    Args:
    record_bytes (bytes): Serialized TFRecord data.

    Returns:
    dict: Decoded features.
    """
    feature = {
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/encodedrawdata': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
        'image/object/class/single': tf.io.FixedLenFeature([], tf.int64),
        'image/object/difficult': tf.io.VarLenFeature(tf.int64),
        'image/object/truncated': tf.io.VarLenFeature(tf.int64),
        'image/object/view': tf.io.VarLenFeature(tf.string)
    }
    example = tf.io.parse_single_example(record_bytes, feature)
    
    return example