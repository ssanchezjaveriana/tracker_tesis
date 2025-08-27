# COCO class names for your 6 tracked classes
CLASS_NAMES = {
    2: 'car',
    24: 'backpack', 
    28: 'suitcase',
    62: 'tv',
    63: 'laptop',
    72: 'refrigerator'
}

def get_class_name(class_id):
    """Get class name from class ID for tracked classes"""
    return CLASS_NAMES.get(class_id, f"unknown_{class_id}")