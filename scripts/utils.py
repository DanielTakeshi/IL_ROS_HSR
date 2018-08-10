

def rgb_baseline_check(h):
    if 'img_rgb' in h:
        return True
    elif 'img_depth' in h:
        return False
    else:
        raise ValueError(h)

def net_check(h):
    if 'grasp_' in h:
        return 'grasp'
    elif 'success_' in h:
        return 'success'
    else:
        raise ValueError(h)


