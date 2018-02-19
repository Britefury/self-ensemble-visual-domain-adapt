import os

def intens_aug_options(intens_scale_range, intens_offset_range):
    intens_scale_range_lower = intens_scale_range_upper = None
    if intens_scale_range != '':
        if ':' not in intens_scale_range:
            print('Invalid intens_scale_range format; should be float:float')
            return
        l, _, h = intens_scale_range.partition(':')
        try:
            intens_scale_range_lower = float(l)
            intens_scale_range_upper = float(h)
        except ValueError:
            print('Invalid intens_scale_range format; should be float:float')
            return

    intens_offset_range_lower = intens_offset_range_upper = None
    if intens_offset_range != '':
        if ':' not in intens_offset_range:
            print('Invalid intens_offset_range format; should be float:float')
            return
        l, _, h = intens_offset_range.partition(':')
        try:
            intens_offset_range_lower = float(l)
            intens_offset_range_upper = float(h)
        except ValueError:
            print('Invalid intens_offset_range format; should be float:float')
            return

    return intens_scale_range_lower, intens_scale_range_upper, intens_offset_range_lower, intens_offset_range_upper


def colon_separated_range(x):
    lower = upper = None
    if x != '':
        if ':' not in x:
            print('Invalid range format; should be float:float')
            return
        l, _, h = x.partition(':')
        try:
            lower = float(l)
            upper = float(h)
        except ValueError:
            print('Invalid range format; should be float:float')
            return

    return lower, upper


def ensure_containing_dir_exists(path):
    dir_name = os.path.dirname(path)
    if dir_name != '' and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return path
