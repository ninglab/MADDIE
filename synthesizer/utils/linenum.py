from inspect import currentframe, getframeinfo

def get_code_location():
    # .f_back moves up one level to the code that called this function
    caller_frame = currentframe().f_back
    info = getframeinfo(caller_frame)
    return f"[{info.filename}:{caller_frame.f_lineno}]"